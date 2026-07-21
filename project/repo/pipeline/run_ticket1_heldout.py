"""Recover deleted Ticket 1 held-out artifacts from the exact frozen baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .artifacts import SUMMARY_COLUMNS, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .baselines import fit_and_evaluate_reference_baseline, make_reference_pipeline
from .data import load_labeled_tweets, select_split_by_id
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "starter" / "configs" / "project_contract.json"
DEFAULT_FREEZE_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "heldout"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--artifact-recovery", action="store_true")
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


def effective_parameters(model: Any) -> dict[str, Any]:
    return {
        "tfidf_vectorizer": _json_value(model.named_steps["features"].get_params(deep=False)),
        "logistic_regression": _json_value(model.named_steps["classifier"].get_params(deep=False)),
    }


def compare_contract(actual: float, reference: float, tolerance: float) -> dict[str, Any]:
    difference = float(actual - reference)
    return {
        "actual": float(actual),
        "reference": float(reference),
        "difference": difference,
        "absolute_difference": abs(difference),
        "tolerance": float(tolerance),
        "matches_reference": abs(difference) <= tolerance,
    }


def validate_frozen_configuration(
    freeze_path: str | Path,
    *,
    data_path: str | Path,
    split_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze["freeze_status"] != "frozen_before_heldout":
        raise ValueError("Ticket 1 configuration is not frozen")
    if freeze["heldout_observed_at_freeze"] is not False or freeze["heldout_evaluation_count_at_freeze"] != 0:
        raise ValueError("freeze does not record held-out unseen")
    expected = freeze["integrity"]
    actual = {
        "data_sha256": sha256(data_path),
        "split_sha256": sha256(split_path),
        "contract_sha256": sha256(contract_path),
        "step4_run_config_sha256": sha256(PROJECT_ROOT / "experiments" / "step-4-baselines" / "run_config.json"),
        "step4_dev_predictions_sha256": sha256(PROJECT_ROOT / "experiments" / "step-4-baselines" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv"),
        "baseline_source_sha256": sha256(PROJECT_ROOT / "pipeline" / "baselines.py"),
        "requirements_lock_sha256": sha256(PROJECT_ROOT / "requirements-lock.txt"),
        "software_versions_sha256": sha256(PROJECT_ROOT / "experiments" / "step-4-baselines" / "software_versions.json"),
    }
    mismatches = {key: {"expected": expected[key], "actual": value} for key, value in actual.items() if expected[key] != value}
    if mismatches:
        raise ValueError(f"frozen integrity validation failed: {mismatches}")
    settings = load_reproducibility_settings()
    if settings.seed != freeze["seed"] or settings.n_jobs != freeze["n_jobs"]:
        raise ValueError("reproducibility settings do not match freeze")
    if effective_parameters(make_reference_pipeline(settings)) != freeze["effective_parameters"]:
        raise ValueError("model parameters do not match freeze")
    return freeze


def _error_rows(data: pd.DataFrame, predictions: pd.DataFrame, kind: str) -> pd.DataFrame:
    merged = predictions.merge(data.loc[:, ["id", "text", "keyword", "location"]], on="id", validate="one_to_one", sort=False)
    if kind == "false_positives":
        selected = merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)]
    else:
        selected = merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)]
    return selected.loc[:, ["id", "text", "keyword", "location", "y_true", "y_pred", "score", "model_name", "ticket"]].reset_index(drop=True)


def main() -> int:
    args = _arguments()
    if not args.artifact_recovery:
        raise RuntimeError("The primary held-out comparison already occurred; only explicit --artifact-recovery is permitted")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("recovery output directory is not empty; refusing overwrite")
    freeze = validate_frozen_configuration(args.freeze, data_path=args.data, split_path=args.split, contract_path=args.contract)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket1_heldout", *sys.argv[1:]])
    freeze_hash = sha256(args.freeze)
    write_json_artifact({
        "status": "artifact_recovery_started",
        "historical_primary_comparison_count": 1,
        "artifact_recovery_rerun": 1,
        "freeze_sha256": freeze_hash,
        "exact_command": command,
        "purpose": "Recreate files deleted by explicit rollback; not a new comparison or decision."
    }, output_dir / "heldout_evaluation_started.json")

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    heldout = select_split_by_id(data, split, "heldout")
    _, evaluation = fit_and_evaluate_reference_baseline(train, heldout, settings)
    comparison = compare_contract(evaluation.metrics["f1_target_1"], contract["reference_baseline_f1"], contract["tolerance"])
    comparison.update({"metric": contract["metric"], "historical_primary_comparison_count": 1, "artifact_recovery_rerun": 1})
    expected_ids = list(split.heldout_ids)
    prediction_path = output_dir / "heldout_predictions.csv"
    write_prediction_artifact(evaluation.predictions, prediction_path, expected_ids=expected_ids)
    write_prediction_artifact(evaluation.predictions, PROJECT_ROOT / "predictions" / "heldout_predictions.csv", expected_ids=expected_ids)
    write_csv_artifact(pd.DataFrame([{ "split": "heldout", "model_name": evaluation.model_name, **evaluation.metrics, "converged": evaluation.converged, "n_iter": json.dumps(evaluation.n_iter)}]), output_dir / "heldout_metrics.csv")
    write_csv_artifact(pd.DataFrame([
        {"model_name": evaluation.model_name, "actual_label": 0, "predicted_0": evaluation.metrics["true_negative"], "predicted_1": evaluation.metrics["false_positive"]},
        {"model_name": evaluation.model_name, "actual_label": 1, "predicted_0": evaluation.metrics["false_negative"], "predicted_1": evaluation.metrics["true_positive"]},
    ]), output_dir / "heldout_confusion_matrix.csv")
    for kind in ("false_positives", "false_negatives"):
        write_csv_artifact(_error_rows(heldout, evaluation.predictions, kind), output_dir / f"heldout_{kind}.csv")
    write_json_artifact(comparison, output_dir / "primary_contract_comparison.json")
    write_json_artifact({"warnings": list(evaluation.warnings), "converged": evaluation.converged, "n_iter": list(evaluation.n_iter or ())}, output_dir / "warnings.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact({
        "scope": "artifact recovery for the historical frozen Ticket 1 held-out comparison",
        "freeze_sha256": freeze_hash,
        "exact_command": command,
        "data_sha256": sha256(args.data),
        "split_sha256": sha256(args.split),
        "contract_sha256": sha256(args.contract),
        "fit_rows": len(train),
        "heldout_rows": len(heldout),
        "selection_or_tuning_on_heldout": False,
        "historical_primary_comparison_count": 1,
        "artifact_recovery_rerun": 1
    }, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    summary = pd.DataFrame([{
        "ticket": "ticket_1",
        "model_name": evaluation.model_name,
        "dev_f1_target_1": freeze["pre_freeze_dev_evidence"]["f1_target_1"],
        "heldout_f1_target_1": evaluation.metrics["f1_target_1"],
        "heldout_accuracy": evaluation.metrics["accuracy"],
        "fixed_fp": 0, "fixed_fn": 0, "new_fp": 0, "new_fn": 0,
        "decision": "frozen_reference_baseline",
        "decision_reason": "Frozen before the historical held-out comparison from project specification and dev evidence; files later reconstructed without changing the decision."
    }], columns=SUMMARY_COLUMNS)
    write_csv_artifact(summary, PROJECT_ROOT / "results" / "summary.csv")
    write_json_artifact({
        "status": "artifact_recovery_completed",
        "historical_primary_comparison_count": 1,
        "artifact_recovery_rerun": 1,
        "freeze_sha256": freeze_hash,
        "prediction_sha256": sha256(prediction_path),
        "matches_reference": comparison["matches_reference"]
    }, output_dir / "heldout_evaluation_completed.json")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(json.dumps(evaluation.metrics, indent=2, sort_keys=True))
    print("historical_primary_comparison_count=1")
    print("artifact_recovery_rerun=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
