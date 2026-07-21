"""Freeze the dev-selected Ticket 4 model, hyperparameters, and threshold.

The selected variant is read dynamically from selection_result.json so that
the expanded MODEL_SPECS (C=5, C=10 variants) can win if they score higher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .artifacts import write_json_artifact, write_text_artifact
from .run_ticket4_dev import sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DIR = PROJECT_ROOT / "experiments" / "ticket-4" / "dev"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise RuntimeError("Ticket 4 freeze exists; refusing overwrite")
    heldout_dir = PROJECT_ROOT / "experiments" / "ticket-4" / "heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise RuntimeError("Ticket 4 held-out artifacts exist before freeze")
    stable_prediction = PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv"
    if stable_prediction.exists():
        raise RuntimeError("Ticket 4 stable held-out predictions exist before freeze")
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if (summary["ticket"] == "ticket_4").any():
        raise RuntimeError("Ticket 4 summary exists before freeze")

    selection_path = args.dev_dir / "selection_result.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    configurations_path = args.dev_dir / "model_configurations.json"
    configurations = json.loads(configurations_path.read_text(encoding="utf-8"))
    metrics_path = args.dev_dir / "results" / "dev_model_metrics.csv"
    metrics = pd.read_csv(metrics_path)

    # Read the winning variant dynamically from the selection result
    SELECTED_VARIANT = selection["selected_variant"]
    selected = metrics.loc[metrics["variant"] == SELECTED_VARIANT]
    if len(selected) != 1:
        raise RuntimeError(
            f"selected variant '{SELECTED_VARIANT}' not found in dev metrics; "
            "check that run_ticket4_dev completed successfully"
        )
    row = selected.iloc[0]
    if selection["heldout_rows_loaded"] != 0 or selection["heldout_evaluations_run"] != 0:
        raise RuntimeError("held-out evidence was present during Ticket 4 selection")

    selected_C = float(row["C"])
    selected_class_weight = str(row["class_weight"])
    selected_threshold = float(row["decision_threshold"])

    decision_reason = (
        f"Selected variant '{SELECTED_VARIANT}' maximised target-1 dev F1 under the predeclared "
        f"bounded criterion (dev F1 {float(row['f1_target_1']):.10f}). "
        f"Hyperparameters: C={selected_C}, class_weight={selected_class_weight}, "
        f"threshold={selected_threshold:.2f}."
    )

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.freeze_ticket4", *sys.argv[1:]])
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "decision_rule_source_sha256": PROJECT_ROOT / "pipeline" / "decision_rule.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_heldout.py",
        "experiment_plan_sha256": args.dev_dir / "experiment_plan.json",
        "dev_metrics_sha256": metrics_path,
        "selection_result_sha256": selection_path,
        "model_configurations_sha256": configurations_path,
        "threshold_sweep_sha256": PROJECT_ROOT / "results" / "threshold_sweep.csv",
        "selected_dev_predictions_sha256": args.dev_dir / "predictions" / f"{SELECTED_VARIANT}_dev_predictions.csv",
        "baseline_heldout_predictions_sha256": PROJECT_ROOT / "predictions" / "heldout_predictions.csv",
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    freeze = {
        "ticket": 4,
        "freeze_status": "frozen_before_ticket4_heldout_evaluation",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_split": "dev_ids only",
        "selection_criterion": selection["criterion"],
        "selected_variant": SELECTED_VARIANT,
        "selected_model_family": selection.get("selected_model_family", "logistic_regression"),
        "selected_preprocessing": "raw text through default TfidfVectorizer fitted only on train_ids",
        "selected_C": selected_C,
        "selected_class_weight": selected_class_weight,
        "selected_threshold": selected_threshold,
        "selected_prediction_rule": f"predict target 1 when class-1 probability >= {selected_threshold:.2f}",
        "selected_effective_configuration": configurations[SELECTED_VARIANT],
        "selected_dev_evidence": {
            key: (row[key].item() if hasattr(row[key], "item") else row[key])
            for key in (
                "precision_target_1",
                "recall_target_1",
                "f1_target_1",
                "accuracy",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "prediction_changes",
                "fixed_fp",
                "fixed_fn",
                "new_fp",
                "new_fn",
            )
        },
        "best_threshold_only_evidence": selection["threshold_sweep_best"],
        "decision_reason": decision_reason,
        "prior_ticket_heldout_artifacts_exist": True,
        "ticket4_heldout_artifact_used_in_decision": False,
        "ticket4_heldout_evaluation_count_at_freeze": 0,
        "selection_reopening_permitted": False,
        "exact_freeze_command": command,
        "integrity": {key: sha256(path) for key, path in paths.items()},
    }
    write_json_artifact(freeze, args.output)
    write_text_artifact(
        "\n".join(
            [
                "# Ticket 4 Freeze Decision",
                "",
                f"Frozen at: {freeze['frozen_at']}",
                "",
                f"Selected: {SELECTED_VARIANT} — C={selected_C}, class_weight={selected_class_weight}, threshold={selected_threshold:.2f}.",
                "",
                decision_reason,
                "",
                "No Ticket 4 held-out artifact or metric was used. Model, preprocessing, hyperparameters, and threshold are locked; held-out cannot reopen the decision.",
            ]
        ),
        args.output.with_name("freeze_decision.md"),
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
