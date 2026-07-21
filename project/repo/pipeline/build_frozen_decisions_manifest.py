"""Collect and validate all ticket freezes into one machine-readable manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import SUMMARY_COLUMNS, write_json_artifact
from .metrics import metric_bundle
from .run_ticket5_dev import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "configs" / "frozen_decisions.json"

FREEZE_PATHS = {
    1: PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
    2: PROJECT_ROOT / "experiments" / "ticket-2" / "frozen_decision.json",
    3: PROJECT_ROOT / "experiments" / "ticket-3" / "frozen_decision.json",
    4: PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json",
    5: PROJECT_ROOT / "experiments" / "ticket-5" / "frozen_decision.json",
}

def _get_ticket4_selected_variant() -> str:
    """Dynamically resolve the selected variant from ticket-4 frozen_decision.json."""
    freeze_path = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"
    if freeze_path.exists():
        return json.loads(freeze_path.read_text(encoding="utf-8"))["selected_variant"]
    return "lr_c1_balanced_default"


def _get_dev_predictions() -> dict[int, Path]:
    variant = _get_ticket4_selected_variant()
    return {
        1: PROJECT_ROOT / "experiments" / "step-4-baselines" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv",
        2: PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "predictions" / "normalize_urls_placeholder_dev_predictions.csv",
        3: PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv",
        4: PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "predictions" / f"{variant}_dev_predictions.csv",
        5: PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "predictions" / f"{variant}_dev_predictions.csv",
    }

HELDOUT_PREDICTIONS = {
    1: PROJECT_ROOT / "predictions" / "heldout_predictions.csv",
    2: PROJECT_ROOT / "predictions" / "ticket-2-heldout-predictions.csv",
    3: PROJECT_ROOT / "predictions" / "ticket-3-heldout-predictions.csv",
    4: PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv",
    5: PROJECT_ROOT / "predictions" / "ticket-5-heldout-predictions.csv",
}

COMPLETION_PATHS = {
    ticket: PROJECT_ROOT / "experiments" / f"ticket-{ticket}" / "heldout" / "heldout_evaluation_completed.json"
    for ticket in range(1, 6)
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_dev_only_decisions(freezes: dict[int, dict[str, Any]]) -> None:
    first = freezes[1]
    if first["heldout_observed_at_freeze"] is not False or first["heldout_evaluation_count_at_freeze"] != 0:
        raise RuntimeError("Ticket 1 freeze was not pre-heldout")
    second = freezes[2]
    if second["heldout_observed_at_freeze"] is not False or second["ticket2_heldout_evaluation_count_at_freeze"] != 0:
        raise RuntimeError("Ticket 2 freeze was not pre-heldout")
    third = freezes[3]
    if third["ticket3_heldout_artifact_used_in_decision"] is not False or third["ticket3_heldout_reporting_count_at_freeze"] != 0:
        raise RuntimeError("Ticket 3 freeze used held-out")
    fourth = freezes[4]
    if fourth["ticket4_heldout_artifact_used_in_decision"] is not False or fourth["ticket4_heldout_evaluation_count_at_freeze"] != 0:
        raise RuntimeError("Ticket 4 freeze used held-out")
    fifth = freezes[5]
    if fifth["ticket5_heldout_artifact_used_in_decision"] is not False or fifth["ticket5_heldout_reporting_count_at_freeze"] != 0:
        raise RuntimeError("Ticket 5 freeze used held-out")
    for ticket, freeze in freezes.items():
        decision_split = freeze.get("selection_split", freeze.get("decision_split", ""))
        if "dev" not in decision_split:
            raise RuntimeError(f"Ticket {ticket} does not record dev-based selection")


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("frozen-decision manifest exists; refusing overwrite")
    DEV_PREDICTIONS = _get_dev_predictions()
    freezes = {ticket: _read_json(path) for ticket, path in FREEZE_PATHS.items()}
    _assert_dev_only_decisions(freezes)
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if list(summary.columns) != SUMMARY_COLUMNS or summary["ticket"].tolist() != [f"ticket_{value}" for value in range(1, 6)]:
        raise RuntimeError("summary does not contain exactly one ordered row per ticket")

    # Dynamically read ticket-4 winner to avoid hardcoding
    t4_freeze = freezes[4]
    t4_variant = t4_freeze["selected_variant"]
    t4_C = float(t4_freeze.get("selected_C", t4_freeze.get("C", 1.0)))
    t4_class_weight = t4_freeze.get("selected_class_weight", t4_freeze.get("class_weight", "balanced"))
    t4_threshold = float(t4_freeze.get("selected_threshold", t4_freeze.get("threshold", 0.5)))
    recipes = {
        1: {"factory": "raw_text_reference", "normalization": None, "C": 1.0, "class_weight": None, "threshold": 0.5, "training_label_corrections": []},
        2: {"factory": "url_placeholder_reference", "normalization": "normalize_urls_placeholder", "C": 1.0, "class_weight": None, "threshold": 0.5, "training_label_corrections": []},
        3: {"factory": "raw_text_reference", "normalization": None, "C": 1.0, "class_weight": None, "threshold": 0.5, "training_label_corrections": []},
        4: {"factory": "raw_text_balanced_logistic", "normalization": None, "C": t4_C, "class_weight": t4_class_weight, "threshold": t4_threshold, "training_label_corrections": []},
        5: {"factory": "raw_text_balanced_logistic", "normalization": None, "C": t4_C, "class_weight": t4_class_weight, "threshold": t4_threshold, "training_label_corrections": freezes[5].get("training_label_corrections", [])},
    }
    model_names = {
        1: "raw_text_tfidf_logistic_regression",
        2: "normalize_urls_placeholder",
        3: "raw_text_tfidf_logistic_regression",
        4: t4_variant,
        5: t4_variant,
    }
    decisions: list[dict[str, Any]] = []
    for ticket in range(1, 6):
        freeze = freezes[ticket]
        summary_row = summary.loc[summary["ticket"] == f"ticket_{ticket}"].iloc[0]
        completion = _read_json(COMPLETION_PATHS[ticket])
        archived_heldout = pd.read_csv(HELDOUT_PREDICTIONS[ticket])
        archived_heldout_metrics = metric_bundle(
            archived_heldout["y_true"], archived_heldout["y_pred"]
        )
        dev_metrics = freeze.get("pre_freeze_dev_evidence", freeze.get("selected_dev_evidence"))
        if ticket == 1:
            decision_reason = freeze["decision_basis"]
        else:
            decision_reason = freeze["decision_reason"]
        decisions.append(
            {
                "ticket": ticket,
                "ticket_key": f"ticket_{ticket}",
                "freeze_path": str(FREEZE_PATHS[ticket].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "freeze_sha256": sha256(FREEZE_PATHS[ticket]),
                "freeze_status": freeze["freeze_status"],
                "frozen_at": freeze["frozen_at"],
                "decision_basis_split": freeze.get("selection_split", freeze.get("decision_split")),
                "heldout_used_for_selection": False,
                "heldout_count_at_freeze": 0,
                "selection_reopening_permitted": bool(freeze.get("selection_reopening_permitted", False)),
                "model_name": model_names[ticket],
                "recipe": recipes[ticket],
                "decision": summary_row["decision"],
                "freeze_decision_reason": decision_reason,
                "summary_decision_reason": str(summary_row["decision_reason"]),
                "expected_dev_metrics": {
                    key: dev_metrics[key]
                    for key in ("precision_target_1", "recall_target_1", "f1_target_1", "accuracy", "true_negative", "false_positive", "false_negative", "true_positive")
                },
                "expected_heldout_metrics": archived_heldout_metrics,
                "expected_transitions_vs_ticket1_baseline": {
                    key: int(summary_row[key]) for key in ("fixed_fp", "fixed_fn", "new_fp", "new_fn")
                },
                "archived_dev_predictions": str(DEV_PREDICTIONS[ticket].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "archived_dev_predictions_sha256": sha256(DEV_PREDICTIONS[ticket]),
                "archived_heldout_predictions": str(HELDOUT_PREDICTIONS[ticket].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "archived_heldout_predictions_sha256": sha256(HELDOUT_PREDICTIONS[ticket]),
                "heldout_completion_path": str(COMPLETION_PATHS[ticket].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "heldout_completion_sha256": sha256(COMPLETION_PATHS[ticket]),
            }
        )
    payload = {
        "manifest_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Step 10 consolidated freeze and clean-process reproducibility audit; no ticket selection is reopened",
        "explicit_replay_authorization": "User Step 10 explicitly requested re-running all final configurations from clean processes.",
        "data_path": "data/train.csv",
        "data_sha256": sha256(PROJECT_ROOT / "data" / "train.csv"),
        "split_path": "starter/data/split_indices.json",
        "split_sha256": sha256(PROJECT_ROOT / "starter" / "data" / "split_indices.json"),
        "requirements_lock_sha256": sha256(PROJECT_ROOT / "requirements-lock.txt"),
        "seed": 3102,
        "n_jobs": 1,
        "baseline_comparator_ticket": "ticket_1",
        "clean_process_policy": "Launch pipeline.reproduce_frozen_ticket once per ticket in a distinct interpreter process; write only under experiments/final-reproducibility-audit/replays.",
        "final_submission_decision": "ticket_5 retains the ticket_4 balanced Logistic Regression configuration with no label corrections",
        "final_prediction_path": "predictions/final-heldout-predictions.csv",
        "decisions": decisions,
        "result_table_inputs": {
            "summary": {"path": "results/summary.csv", "sha256": sha256(PROJECT_ROOT / "results" / "summary.csv")},
            "threshold_sweep": {"path": "results/threshold_sweep.csv", "sha256": sha256(PROJECT_ROOT / "results" / "threshold_sweep.csv")},
            "data_quality_audit": {"path": "results/data_quality_audit.csv", "sha256": sha256(PROJECT_ROOT / "results" / "data_quality_audit.csv")},
        },
        "exact_command": subprocess.list2cmdline([sys.executable, "-m", "pipeline.build_frozen_decisions_manifest"]),
    }
    write_json_artifact(payload, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "tickets": len(decisions), "all_dev_based": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
