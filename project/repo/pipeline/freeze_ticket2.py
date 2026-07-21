"""Freeze the dev-selected Ticket 2 URL-normalization decision before held-out."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .artifacts import write_json_artifact, write_text_artifact
from .normalization import NORMALIZATION_PARAMETERS, URL_PATTERN, URL_TOKEN
from .run_ticket2_heldout import effective_parameters, sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_PATH = PROJECT_ROOT / "experiments" / "ticket-2" / "dev"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-2" / "frozen_decision.json"
SELECTED_VARIANT = "normalize_urls_placeholder"
DECISION_REASON = (
    "Selected on dev only: URL placeholdering had the highest normalization-variant "
    "F1 (0.7403132728771641 versus raw 0.7388120423108218), increased precision and "
    "accuracy, fixed 28 baseline errors while creating 22, and was exactly invariant "
    "on all 767 URL-perturbed dev rows while the raw control changed 275 predictions."
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output.exists():
        raise RuntimeError("Ticket 2 freeze already exists; refusing overwrite")
    heldout_dir = PROJECT_ROOT / "experiments" / "ticket-2" / "heldout"
    if heldout_dir.exists() and any(heldout_dir.iterdir()):
        raise RuntimeError("held-out artifacts exist before Ticket 2 freeze")
    summary = pd.read_csv(PROJECT_ROOT / "results" / "summary.csv")
    if (summary["ticket"] == "ticket_2").any():
        raise RuntimeError("Ticket 2 summary exists before freeze")

    metrics = pd.read_csv(args.dev_dir / "results" / "dev_metrics.csv")
    selected = metrics.loc[metrics["variant"] == SELECTED_VARIANT].iloc[0]
    control = metrics.loc[metrics["variant"] == "raw_text_control"].iloc[0]
    robustness = pd.read_csv(args.dev_dir / "robustness" / "robustness_metrics.csv")
    selected_robustness = robustness.loc[
        (robustness["perturbation_for"] == SELECTED_VARIANT)
        & (robustness["evaluated_variant"] == SELECTED_VARIANT)
    ].iloc[0]
    control_robustness = robustness.loc[
        (robustness["perturbation_for"] == SELECTED_VARIANT)
        & (robustness["evaluated_variant"] == "raw_text_control")
    ].iloc[0]
    if float(selected["f1_target_1"]) <= float(control["f1_target_1"]):
        raise ValueError("selected normalization does not improve dev F1")
    if int(selected_robustness["changed_predictions"]) != 0:
        raise ValueError("selected normalization is not prediction-invariant to URL perturbation")

    command = subprocess.list2cmdline(
        [sys.executable, "-m", "pipeline.freeze_ticket2", *sys.argv[1:]]
    )
    path_by_integrity_key = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "normalization_source_sha256": PROJECT_ROOT / "pipeline" / "normalization.py",
        "ticket2_source_sha256": PROJECT_ROOT / "pipeline" / "ticket2.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket2_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket2_heldout.py",
        "experiment_plan_sha256": args.dev_dir / "experiment_plan.json",
        "dev_run_config_sha256": args.dev_dir / "run_config.json",
        "dev_metrics_sha256": args.dev_dir / "results" / "dev_metrics.csv",
        "control_dev_predictions_sha256": args.dev_dir / "predictions" / "raw_text_control_dev_predictions.csv",
        "selected_dev_predictions_sha256": args.dev_dir / "predictions" / f"{SELECTED_VARIANT}_dev_predictions.csv",
        "robustness_metrics_sha256": args.dev_dir / "robustness" / "robustness_metrics.csv",
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    freeze = {
        "ticket": 2,
        "freeze_status": "frozen_before_ticket2_heldout",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_variant": SELECTED_VARIANT,
        "decision": "Adopt URL placeholder normalization as the single Ticket 2 lever.",
        "decision_reason": DECISION_REASON,
        "decision_split": "dev_ids only",
        "heldout_observed_at_freeze": False,
        "ticket2_heldout_evaluation_count_at_freeze": 0,
        "selection_reopening_permitted": False,
        "normalization": {
            "parameters": NORMALIZATION_PARAMETERS[SELECTED_VARIANT],
            "url_pattern": URL_PATTERN.pattern,
            "replacement_token": URL_TOKEN.strip(),
            "order": "URL replacement occurs before the otherwise unchanged frozen TF-IDF vectorizer.",
        },
        "effective_parameters": effective_parameters(SELECTED_VARIANT),
        "raw_control_dev_evidence": {
            key: control[key].item() if hasattr(control[key], "item") else control[key]
            for key in (
                "precision_target_1",
                "recall_target_1",
                "f1_target_1",
                "accuracy",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            )
        },
        "selected_dev_evidence": {
            key: selected[key].item() if hasattr(selected[key], "item") else selected[key]
            for key in (
                "precision_target_1",
                "recall_target_1",
                "f1_target_1",
                "accuracy",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "f1_delta_vs_frozen_baseline",
                "prediction_changes",
                "fixed_fp",
                "fixed_fn",
                "new_fp",
                "new_fn",
                "converged",
                "n_iter",
            )
        },
        "url_robustness_evidence": {
            "affected_dev_rows": int(selected_robustness["affected_rows"]),
            "selected_changed_predictions": int(selected_robustness["changed_predictions"]),
            "selected_maximum_absolute_score_shift": float(selected_robustness["maximum_absolute_score_shift"]),
            "raw_control_changed_predictions": int(control_robustness["changed_predictions"]),
            "raw_control_prediction_invariance_rate": float(control_robustness["prediction_invariance_rate"]),
        },
        "exact_freeze_command": command,
        "integrity": {
            key: sha256(path) for key, path in path_by_integrity_key.items()
        },
    }
    write_json_artifact(freeze, args.output)
    write_text_artifact(
        "\n".join(
            [
                "# Ticket 2 Freeze Decision",
                "",
                f"Frozen at: {freeze['frozen_at']}",
                "",
                "Selected variant: `normalize_urls_placeholder`.",
                "",
                DECISION_REASON,
                "",
                "Held-out observed for Ticket 2 at freeze: **no**.",
                "",
                "The selection is closed. Held-out may be evaluated once and must not reopen the choice.",
            ]
        ),
        args.output.with_name("freeze_decision.md"),
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
