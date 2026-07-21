"""Evaluate the frozen Ticket 4 model and threshold on held-out exactly once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    SUMMARY_COLUMNS,
    validate_prediction_frame,
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .data import load_labeled_tweets, select_split_by_id
from .decision_rule import MODEL_SPECS, fit_and_evaluate_spec, predictions_at_threshold
from .metrics import metric_bundle
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .run_ticket4_dev import sha256
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import error_rows, prediction_change_rows, transition_counts
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_FREEZE_PATH = PROJECT_ROOT / "experiments" / "ticket-4" / "frozen_decision.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-4" / "heldout"
BASELINE_PREDICTIONS = PROJECT_ROOT / "predictions" / "heldout_predictions.csv"
STABLE_PREDICTIONS = PROJECT_ROOT / "predictions" / "ticket-4-heldout-predictions.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--confirm-single-ticket4-evaluation", action="store_true")
    return parser.parse_args()


def validate_frozen_ticket4_configuration(path: str | Path = DEFAULT_FREEZE_PATH) -> dict[str, Any]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "frozen_before_ticket4_heldout_evaluation":
        raise ValueError("Ticket 4 decision is not frozen")
    if (
        freeze["selected_variant"] != "lr_c1_balanced_default"
        or float(freeze["selected_C"]) != 1.0
        or freeze["selected_class_weight"] != "balanced"
        or float(freeze["selected_threshold"]) != 0.5
    ):
        raise ValueError("Ticket 4 freeze does not contain the dev-selected decision")
    if freeze["ticket4_heldout_evaluation_count_at_freeze"] != 0:
        raise ValueError("Ticket 4 held-out evaluation count was not zero at freeze")
    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "decision_rule_source_sha256": PROJECT_ROOT / "pipeline" / "decision_rule.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket4_heldout.py",
        "experiment_plan_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "experiment_plan.json",
        "dev_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "results" / "dev_model_metrics.csv",
        "selection_result_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "selection_result.json",
        "model_configurations_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "model_configurations.json",
        "threshold_sweep_sha256": PROJECT_ROOT / "results" / "threshold_sweep.csv",
        "selected_dev_predictions_sha256": PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "predictions" / "lr_c1_balanced_default_dev_predictions.csv",
        "baseline_heldout_predictions_sha256": BASELINE_PREDICTIONS,
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    mismatches = {
        key: {"expected": freeze["integrity"][key], "actual": sha256(file)}
        for key, file in paths.items()
        if freeze["integrity"][key] != sha256(file)
    }
    if mismatches:
        raise ValueError(f"Ticket 4 frozen integrity validation failed: {mismatches}")
    return freeze


def main() -> int:
    args = _arguments()
    if not args.confirm_single_ticket4_evaluation:
        raise RuntimeError("explicit single Ticket 4 held-out evaluation confirmation is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Ticket 4 held-out artifacts exist; refusing repeated evaluation")
    if STABLE_PREDICTIONS.exists():
        raise RuntimeError("Ticket 4 stable held-out predictions exist; refusing overwrite")
    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    summary = pd.read_csv(summary_path)
    if list(summary.columns) != SUMMARY_COLUMNS or (summary["ticket"] == "ticket_4").any():
        raise RuntimeError("summary schema invalid or Ticket 4 row already exists")
    freeze = validate_frozen_ticket4_configuration(args.freeze)

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    heldout = select_split_by_id(data, split, "heldout")
    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    validate_prediction_frame(baseline, expected_ids=list(split.heldout_ids))
    spec = next(item for item in MODEL_SPECS if item.name == freeze["selected_variant"])

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket4_heldout", *sys.argv[1:]])
    freeze_hash = sha256(args.freeze)
    write_json_artifact(
        {
            "status": "ticket4_heldout_evaluation_started",
            "ticket4_heldout_evaluation_number": 1,
            "freeze_sha256": freeze_hash,
            "exact_command": command,
            "selection_reopened": False,
        },
        output_dir / "heldout_evaluation_started.json",
    )

    model, evaluation = fit_and_evaluate_spec(train, heldout, spec, settings)
    threshold = float(freeze["selected_threshold"])
    probability_predictions = predictions_at_threshold(evaluation.predictions["score"], threshold)
    if not np.array_equal(probability_predictions, evaluation.predictions["y_pred"].to_numpy(dtype=int)):
        raise AssertionError("frozen threshold rule differs from classifier prediction at 0.50")
    predictions = evaluation.predictions.copy()
    predictions["ticket"] = "ticket_4_decision_rule"
    write_prediction_artifact(predictions, output_dir / "heldout_predictions.csv", expected_ids=list(split.heldout_ids))
    write_prediction_artifact(predictions, STABLE_PREDICTIONS, expected_ids=list(split.heldout_ids))
    metrics = metric_bundle(predictions["y_true"], predictions["y_pred"])
    transitions = transition_counts(baseline, predictions)
    changes = prediction_change_rows(baseline, predictions, heldout)

    write_csv_artifact(
        pd.DataFrame(
            [
                {
                    "split": "heldout",
                    "variant": freeze["selected_variant"],
                    "C": freeze["selected_C"],
                    "class_weight": freeze["selected_class_weight"],
                    "decision_threshold": threshold,
                    **metrics,
                    **transitions,
                    "converged": evaluation.converged,
                    "n_iter": json.dumps(evaluation.n_iter),
                }
            ]
        ),
        output_dir / "heldout_metrics.csv",
    )
    write_csv_artifact(
        pd.DataFrame(
            [
                {"model_name": freeze["selected_variant"], "actual_label": 0, "predicted_0": metrics["true_negative"], "predicted_1": metrics["false_positive"]},
                {"model_name": freeze["selected_variant"], "actual_label": 1, "predicted_0": metrics["false_negative"], "predicted_1": metrics["true_positive"]},
            ]
        ),
        output_dir / "heldout_confusion_matrix.csv",
    )
    write_csv_artifact(changes, output_dir / "heldout_changes_vs_frozen_baseline.csv")
    write_csv_artifact(error_rows(heldout, evaluation, "false_positives"), output_dir / "heldout_false_positives.csv")
    write_csv_artifact(error_rows(heldout, evaluation, "false_negatives"), output_dir / "heldout_false_negatives.csv")
    write_json_artifact(
        {
            "converged": evaluation.converged,
            "n_iter": evaluation.n_iter,
            "warnings": evaluation.warnings,
        },
        output_dir / "warnings.json",
    )
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact(
        {
            "scope": "Single held-out evaluation of the frozen Ticket 4 balanced Logistic Regression decision",
            "exact_command": command,
            "freeze_sha256": freeze_hash,
            "ticket4_heldout_evaluation_number": 1,
            "train_rows": len(train),
            "heldout_rows": len(heldout),
            "selection_or_tuning_on_heldout": False,
            "selection_reopened": False,
        },
        output_dir / "run_config.json",
    )
    write_text_artifact(command, output_dir / "run_command.txt")

    selected_dev = freeze["selected_dev_evidence"]
    row = pd.DataFrame(
        [
            {
                "ticket": "ticket_4",
                "model_name": freeze["selected_variant"],
                "dev_f1_target_1": selected_dev["f1_target_1"],
                "heldout_f1_target_1": metrics["f1_target_1"],
                "heldout_accuracy": metrics["accuracy"],
                "fixed_fp": transitions["fixed_fp"],
                "fixed_fn": transitions["fixed_fn"],
                "new_fp": transitions["new_fp"],
                "new_fn": transitions["new_fn"],
                "decision": "adopt_balanced_logistic_regression_at_default_threshold",
                "decision_reason": freeze["decision_reason"],
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    write_csv_artifact(pd.concat([summary, row], ignore_index=True), summary_path)
    write_json_artifact(
        {
            "status": "ticket4_heldout_evaluation_completed",
            "ticket4_heldout_evaluation_count": 1,
            "freeze_sha256": freeze_hash,
            "prediction_sha256": sha256(output_dir / "heldout_predictions.csv"),
            "stable_prediction_sha256": sha256(STABLE_PREDICTIONS),
            "selection_reopened": False,
            "metrics": metrics,
            "transitions_vs_frozen_baseline": transitions,
        },
        output_dir / "heldout_evaluation_completed.json",
    )
    print(json.dumps({"metrics": metrics, "transitions": transitions}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
