"""Run the bounded Ticket 4 model and decision-rule investigation on dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    THRESHOLD_SWEEP_COLUMNS,
    validate_prediction_frame,
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .data import load_labeled_tweets, select_split_by_id
from .decision_rule import (
    BASELINE_VARIANT,
    MODEL_SPECS,
    THRESHOLD_VARIANT,
    evaluation_from_threshold,
    fit_and_evaluate_spec,
    select_best_threshold,
    threshold_sweep_rows,
)
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import error_rows, prediction_change_rows, transition_counts
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "experiments" / "ticket-4" / "dev" / "experiment_plan.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-4" / "dev"
BASELINE_DEV_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "step-4-baselines"
    / "predictions"
    / "raw_text_tfidf_logistic_regression_dev_predictions.csv"
)
ROOT_THRESHOLD_SWEEP = PROJECT_ROOT / "results" / "threshold_sweep.csv"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    return value


def _assert_plan(plan: dict[str, Any]) -> None:
    if plan.get("ticket") != 4 or plan.get("created_before_dev_execution") is not True:
        raise ValueError("Ticket 4 plan is missing its pre-execution declaration")
    sweep = plan["threshold_sweep"]
    if (sweep["inclusive_minimum"], sweep["inclusive_maximum"], sweep["step"], sweep["count"]) != (0.2, 0.8, 0.01, 61):
        raise ValueError("Ticket 4 threshold plan differs from the frozen bounded grid")
    if plan.get("heldout_access_by_dev_command") is not False:
        raise ValueError("dev plan must prohibit held-out access")


def _assert_control_reproduction(control: pd.DataFrame, expected_ids: list[int]) -> pd.DataFrame:
    frozen = pd.read_csv(BASELINE_DEV_PREDICTIONS)
    validate_prediction_frame(frozen, expected_ids=expected_ids)
    validate_prediction_frame(control, expected_ids=expected_ids)
    if not np.array_equal(frozen["y_true"].to_numpy(), control["y_true"].to_numpy()):
        raise AssertionError("Ticket 4 control labels do not reproduce the frozen baseline")
    if not np.array_equal(frozen["y_pred"].to_numpy(), control["y_pred"].to_numpy()):
        raise AssertionError("Ticket 4 control predictions do not reproduce the frozen baseline")
    score_difference = np.abs(
        frozen["score"].to_numpy(dtype=float) - control["score"].to_numpy(dtype=float)
    )
    if not np.all(score_difference <= 1e-12):
        raise AssertionError("Ticket 4 control scores differ materially from the frozen baseline")
    return frozen


def _coefficient_rows(model: Any, variant: str, limit: int = 25) -> list[dict[str, Any]]:
    features = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    order = np.argsort(coefficients)
    rows: list[dict[str, Any]] = []
    for direction, indices in (
        ("negative", order[:limit]),
        ("positive", order[-limit:][::-1]),
    ):
        for rank, index in enumerate(indices, start=1):
            rows.append(
                {
                    "variant": variant,
                    "direction": direction,
                    "rank": rank,
                    "feature": str(features[index]),
                    "coefficient": float(coefficients[index]),
                }
            )
    return rows


def _selection_key(row: dict[str, Any], best_f1: float) -> tuple[Any, ...]:
    is_tied = abs(float(row["f1_target_1"]) - best_f1) <= 1e-12
    if not is_tied:
        return (1,)
    return (
        0,
        int(row["prediction_changes"]),
        abs(float(row["decision_threshold"]) - 0.5),
        str(row["variant"]),
    )


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    existing = [path for path in output_dir.iterdir()] if output_dir.exists() else []
    allowed_existing = {args.plan.resolve()}
    unexpected = [path for path in existing if path.resolve() not in allowed_existing]
    if unexpected:
        raise RuntimeError("Ticket 4 dev artifacts already exist; refusing repeated execution")
    if ROOT_THRESHOLD_SWEEP.exists():
        raise RuntimeError("results/threshold_sweep.csv already exists; refusing overwrite")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    _assert_plan(plan)
    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")
    expected_ids = list(split.dev_ids)

    models: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    warnings_payload: dict[str, Any] = {}
    configurations: dict[str, Any] = {}
    coefficients: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model, evaluation = fit_and_evaluate_spec(train, dev, spec, settings)
        models[spec.name] = model
        evaluations[spec.name] = evaluation
        warnings_payload[spec.name] = {
            "converged": evaluation.converged,
            "n_iter": evaluation.n_iter,
            "warnings": evaluation.warnings,
        }
        configurations[spec.name] = {
            "classifier_family": spec.classifier,
            "intended_lever": spec.intended_lever,
            "native_threshold": spec.native_threshold,
            "effective_tfidf_parameters": _jsonable(model.named_steps["features"].get_params(deep=False)),
            "effective_classifier_parameters": _jsonable(model.named_steps["classifier"].get_params(deep=False)),
        }
        coefficients.extend(_coefficient_rows(model, spec.name))

    control = evaluations[BASELINE_VARIANT]
    frozen_baseline = _assert_control_reproduction(control.predictions, expected_ids)
    sweep_rows = threshold_sweep_rows(control)
    selected_threshold_row = select_best_threshold(sweep_rows)
    selected_threshold = float(selected_threshold_row["threshold"])
    threshold_evaluation = evaluation_from_threshold(control, selected_threshold)
    evaluations[THRESHOLD_VARIANT] = threshold_evaluation

    contract_sweep = pd.DataFrame(sweep_rows).loc[:, THRESHOLD_SWEEP_COLUMNS]
    write_csv_artifact(contract_sweep, ROOT_THRESHOLD_SWEEP)
    detailed_sweep_rows: list[dict[str, Any]] = []
    for row in sweep_rows:
        candidate = evaluation_from_threshold(control, float(row["threshold"]))
        transitions = transition_counts(frozen_baseline, candidate.predictions)
        detailed_sweep_rows.append({**row, **transitions})
    write_csv_artifact(pd.DataFrame(detailed_sweep_rows), output_dir / "results" / "threshold_sweep_detailed.csv")

    spec_by_name = {spec.name: spec for spec in MODEL_SPECS}
    metric_rows: list[dict[str, Any]] = []
    for variant, evaluation in evaluations.items():
        if variant == THRESHOLD_VARIANT:
            family = "logistic_regression"
            c_value = 1.0
            class_weight = None
            threshold = selected_threshold
            lever = "threshold"
        else:
            spec = spec_by_name[variant]
            family = spec.classifier
            c_value = spec.c
            class_weight = spec.class_weight
            threshold = spec.native_threshold
            lever = spec.intended_lever
        transitions = transition_counts(frozen_baseline, evaluation.predictions)
        metric_rows.append(
            {
                "variant": variant,
                "classifier_family": family,
                "C": c_value,
                "class_weight": class_weight,
                "decision_threshold": threshold,
                "intended_lever": lever,
                **evaluation.metrics,
                **transitions,
                "converged": evaluation.converged,
                "n_iter": json.dumps(evaluation.n_iter),
            }
        )
        write_prediction_artifact(
            evaluation.predictions,
            output_dir / "predictions" / f"{variant}_dev_predictions.csv",
            expected_ids=expected_ids,
        )
        write_csv_artifact(
            prediction_change_rows(frozen_baseline, evaluation.predictions, dev),
            output_dir / "changes" / f"{variant}_changes.csv",
        )
        write_csv_artifact(
            error_rows(dev, evaluation, "false_positives"),
            output_dir / "errors" / f"{variant}_false_positives.csv",
        )
        write_csv_artifact(
            error_rows(dev, evaluation, "false_negatives"),
            output_dir / "errors" / f"{variant}_false_negatives.csv",
        )

    best_f1 = max(float(row["f1_target_1"]) for row in metric_rows)
    selected = min(metric_rows, key=lambda row: _selection_key(row, best_f1))
    selection_payload = {
        "selection_split": "dev_ids only",
        "criterion": plan["selection_criterion"],
        "selected_variant": selected["variant"],
        "selected_model_family": selected["classifier_family"],
        "selected_C": selected["C"],
        "selected_class_weight": selected["class_weight"],
        "selected_threshold": selected["decision_threshold"],
        "selected_metrics": {key: selected[key] for key in ("precision_target_1", "recall_target_1", "f1_target_1", "accuracy", "true_negative", "false_positive", "false_negative", "true_positive")},
        "transitions_vs_frozen_baseline": {key: selected[key] for key in ("prediction_changes", "fixed_fp", "fixed_fn", "new_fp", "new_fn")},
        "threshold_sweep_best": selected_threshold_row,
        "heldout_rows_loaded": 0,
        "heldout_evaluations_run": 0,
    }
    write_csv_artifact(pd.DataFrame(metric_rows), output_dir / "results" / "dev_model_metrics.csv")
    write_csv_artifact(pd.DataFrame(coefficients), output_dir / "interpretability" / "top_coefficients.csv")
    write_json_artifact(warnings_payload, output_dir / "warnings.json")
    write_json_artifact(configurations, output_dir / "model_configurations.json")
    write_json_artifact(selection_payload, output_dir / "selection_result.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")

    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket4_dev", *sys.argv[1:]])
    run_config = {
        "scope": "Ticket 4 bounded decision-rule and model comparison on dev only",
        "exact_command": command,
        "data_sha256": sha256(args.data),
        "split_sha256": sha256(args.split),
        "plan_sha256": sha256(args.plan),
        "baseline_dev_predictions_sha256": sha256(BASELINE_DEV_PREDICTIONS),
        "seed": settings.seed,
        "n_jobs": settings.n_jobs,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "heldout_rows_loaded": 0,
        "heldout_evaluations_run": 0,
        "native_model_variants": len(MODEL_SPECS),
        "threshold_candidates": len(sweep_rows),
        "control_reproduces_frozen_baseline_predictions_exactly": True,
        "control_scores_match_frozen_baseline_within_absolute_tolerance": 1e-12,
    }
    write_json_artifact(run_config, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(json.dumps(selection_payload, indent=2, sort_keys=True))
    print("control_reproduction=PASS")
    print("heldout_evaluations_run=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
