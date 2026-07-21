"""Evaluate the frozen Ticket 2 normalization decision once on held-out."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

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
from .normalization import NORMALIZATION_PARAMETERS
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import (
    error_rows,
    fit_and_evaluate_variant,
    make_ticket2_pipeline,
    prediction_change_rows,
    transition_counts,
)
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_FREEZE_PATH = (
    PROJECT_ROOT / "experiments" / "ticket-2" / "frozen_decision.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-2" / "heldout"
BASELINE_HELDOUT_PREDICTIONS = (
    PROJECT_ROOT / "predictions" / "heldout_predictions.csv"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--confirm-single-heldout-evaluation", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def effective_parameters(variant: str) -> dict[str, Any]:
    settings = load_reproducibility_settings()
    model = make_ticket2_pipeline(variant, settings)
    return {
        "normalizer": _json_value(model.named_steps["normalizer"].get_params(deep=False)),
        "tfidf_vectorizer": _json_value(model.named_steps["features"].get_params(deep=False)),
        "logistic_regression": _json_value(model.named_steps["classifier"].get_params(deep=False)),
    }


def validate_frozen_ticket2_configuration(
    freeze_path: str | Path = DEFAULT_FREEZE_PATH,
) -> dict[str, Any]:
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "frozen_before_ticket2_heldout":
        raise ValueError("Ticket 2 decision is not frozen")
    if freeze["heldout_observed_at_freeze"] is not False:
        raise ValueError("freeze does not record held-out unseen for Ticket 2")
    if freeze["ticket2_heldout_evaluation_count_at_freeze"] != 0:
        raise ValueError("Ticket 2 held-out evaluation count was not zero at freeze")
    variant = freeze["selected_variant"]
    if variant not in NORMALIZATION_PARAMETERS or variant == "raw_text_control":
        raise ValueError("freeze selected an unknown or non-normalized variant")
    if freeze["effective_parameters"] != effective_parameters(variant):
        raise ValueError("current effective model parameters do not match Ticket 2 freeze")

    paths = {
        "data_sha256": PROJECT_ROOT / "data" / "train.csv",
        "split_sha256": PROJECT_ROOT / "starter" / "data" / "split_indices.json",
        "ticket1_freeze_sha256": PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json",
        "normalization_source_sha256": PROJECT_ROOT / "pipeline" / "normalization.py",
        "ticket2_source_sha256": PROJECT_ROOT / "pipeline" / "ticket2.py",
        "dev_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket2_dev.py",
        "heldout_runner_source_sha256": PROJECT_ROOT / "pipeline" / "run_ticket2_heldout.py",
        "experiment_plan_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "experiment_plan.json",
        "dev_run_config_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "run_config.json",
        "dev_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "results" / "dev_metrics.csv",
        "control_dev_predictions_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "predictions" / "raw_text_control_dev_predictions.csv",
        "selected_dev_predictions_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "predictions" / f"{variant}_dev_predictions.csv",
        "robustness_metrics_sha256": PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "robustness" / "robustness_metrics.csv",
        "requirements_lock_sha256": PROJECT_ROOT / "requirements-lock.txt",
    }
    mismatches = {
        key: {"expected": freeze["integrity"][key], "actual": sha256(path)}
        for key, path in paths.items()
        if freeze["integrity"][key] != sha256(path)
    }
    if mismatches:
        raise ValueError(f"Ticket 2 frozen integrity validation failed: {mismatches}")
    return freeze


def main() -> int:
    args = _arguments()
    if not args.confirm_single_heldout_evaluation:
        raise RuntimeError("explicit single Ticket 2 held-out confirmation is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Ticket 2 held-out output exists; refusing a repeated evaluation")

    freeze = validate_frozen_ticket2_configuration(args.freeze)
    command = subprocess.list2cmdline(
        [sys.executable, "-m", "pipeline.run_ticket2_heldout", *sys.argv[1:]]
    )
    freeze_hash = sha256(args.freeze)
    write_json_artifact(
        {
            "status": "ticket2_heldout_evaluation_started",
            "ticket2_heldout_evaluation_number": 1,
            "freeze_sha256": freeze_hash,
            "selected_variant": freeze["selected_variant"],
            "exact_command": command,
            "selection_reopened": False,
        },
        output_dir / "heldout_evaluation_started.json",
    )

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    heldout = select_split_by_id(data, split, "heldout")
    _, evaluation = fit_and_evaluate_variant(
        train, heldout, freeze["selected_variant"], settings
    )
    expected_ids = list(split.heldout_ids)
    selected_path = output_dir / "heldout_predictions.csv"
    write_prediction_artifact(
        evaluation.predictions, selected_path, expected_ids=expected_ids
    )
    write_prediction_artifact(
        evaluation.predictions,
        PROJECT_ROOT / "predictions" / "ticket-2-heldout-predictions.csv",
        expected_ids=expected_ids,
    )

    baseline_predictions = pd.read_csv(BASELINE_HELDOUT_PREDICTIONS)
    validate_prediction_frame(baseline_predictions, expected_ids=expected_ids)
    transitions = transition_counts(baseline_predictions, evaluation.predictions)
    write_csv_artifact(
        prediction_change_rows(baseline_predictions, evaluation.predictions, heldout),
        output_dir / "heldout_changes_vs_frozen_baseline.csv",
    )
    for kind in ("false_positives", "false_negatives"):
        write_csv_artifact(
            error_rows(heldout, evaluation, kind),
            output_dir / f"heldout_{kind}.csv",
        )
    metrics_row = {
        "split": "heldout",
        "variant": freeze["selected_variant"],
        "model_name": evaluation.model_name,
        **evaluation.metrics,
        **transitions,
        "converged": evaluation.converged,
        "n_iter": json.dumps(evaluation.n_iter),
    }
    write_csv_artifact(pd.DataFrame([metrics_row]), output_dir / "heldout_metrics.csv")
    write_csv_artifact(
        pd.DataFrame(
            [
                {
                    "model_name": evaluation.model_name,
                    "actual_label": 0,
                    "predicted_0": evaluation.metrics["true_negative"],
                    "predicted_1": evaluation.metrics["false_positive"],
                },
                {
                    "model_name": evaluation.model_name,
                    "actual_label": 1,
                    "predicted_0": evaluation.metrics["false_negative"],
                    "predicted_1": evaluation.metrics["true_positive"],
                },
            ]
        ),
        output_dir / "heldout_confusion_matrix.csv",
    )
    write_json_artifact(
        {
            "warnings": list(evaluation.warnings),
            "converged": evaluation.converged,
            "n_iter": list(evaluation.n_iter or ()),
        },
        output_dir / "warnings.json",
    )
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact(
        {
            "scope": "single held-out evaluation of the frozen Ticket 2 decision",
            "exact_command": command,
            "freeze_sha256": freeze_hash,
            "data_sha256": sha256(args.data),
            "split_sha256": sha256(args.split),
            "selected_variant": freeze["selected_variant"],
            "fit_rows": len(train),
            "heldout_rows": len(heldout),
            "ticket2_heldout_evaluation_number": 1,
            "selection_or_tuning_on_heldout": False,
        },
        output_dir / "run_config.json",
    )
    write_text_artifact(command, output_dir / "run_command.txt")

    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    summary = pd.read_csv(summary_path)
    if list(summary.columns) != SUMMARY_COLUMNS:
        raise ValueError("existing summary schema does not match required contract")
    if (summary["ticket"] == "ticket_2").any():
        raise RuntimeError("Ticket 2 summary row already exists; refusing duplicate")
    summary_row = pd.DataFrame(
        [
            {
                "ticket": "ticket_2",
                "model_name": evaluation.model_name,
                "dev_f1_target_1": freeze["selected_dev_evidence"]["f1_target_1"],
                "heldout_f1_target_1": evaluation.metrics["f1_target_1"],
                "heldout_accuracy": evaluation.metrics["accuracy"],
                "fixed_fp": transitions["fixed_fp"],
                "fixed_fn": transitions["fixed_fn"],
                "new_fp": transitions["new_fp"],
                "new_fn": transitions["new_fn"],
                "decision": "adopt_url_placeholder_normalization",
                "decision_reason": freeze["decision_reason"],
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    write_csv_artifact(pd.concat([summary, summary_row], ignore_index=True), summary_path)
    write_json_artifact(
        {
            "status": "ticket2_heldout_evaluation_completed",
            "ticket2_heldout_evaluation_count": 1,
            "freeze_sha256": freeze_hash,
            "prediction_sha256": sha256(selected_path),
            "selection_reopened": False,
            "metrics": evaluation.metrics,
            "transitions_vs_frozen_baseline": transitions,
        },
        output_dir / "heldout_evaluation_completed.json",
    )
    print(json.dumps(metrics_row, indent=2, sort_keys=True))
    print("ticket2_heldout_evaluation_count=1")
    print("selection_reopened=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
