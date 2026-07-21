"""Run only the instructor floor and minimal reference baseline on dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import (
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .baselines import (
    BaselineEvaluation,
    assert_identical_evaluations,
    evaluate_floor_model,
    fit_and_evaluate_reference_baseline,
    make_reference_pipeline,
)
from .data import load_labeled_tweets, select_split_by_id
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "step-4-baselines"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--repetitions", type=int, default=2, choices=[2])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def _effective_parameters(model: Any) -> dict[str, Any]:
    return {
        "tfidf_vectorizer": _json_value(
            model.named_steps["features"].get_params(deep=False)
        ),
        "logistic_regression": _json_value(
            model.named_steps["classifier"].get_params(deep=False)
        ),
    }


def _metrics_table(evaluations: list[BaselineEvaluation]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        rows.append(
            {
                "split": "dev",
                "model_name": evaluation.model_name,
                **evaluation.metrics,
                "converged": evaluation.converged,
                "n_iter": (
                    "" if evaluation.n_iter is None else json.dumps(evaluation.n_iter)
                ),
            }
        )
    return pd.DataFrame(rows)


def _confusion_table(evaluations: list[BaselineEvaluation]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        metrics = evaluation.metrics
        rows.extend(
            [
                {
                    "model_name": evaluation.model_name,
                    "actual_label": 0,
                    "predicted_0": metrics["true_negative"],
                    "predicted_1": metrics["false_positive"],
                },
                {
                    "model_name": evaluation.model_name,
                    "actual_label": 1,
                    "predicted_0": metrics["false_negative"],
                    "predicted_1": metrics["true_positive"],
                },
            ]
        )
    return pd.DataFrame(rows)


def _error_examples(
    dev_frame: pd.DataFrame,
    evaluation: BaselineEvaluation,
    *,
    error_type: str,
) -> pd.DataFrame:
    context_columns = ["id", "text", "keyword", "location"]
    merged = evaluation.predictions.merge(
        dev_frame.loc[:, context_columns],
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if error_type == "false_positives":
        selected = merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)]
    elif error_type == "false_negatives":
        selected = merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)]
    else:
        raise ValueError(f"unknown error type {error_type!r}")
    columns = [
        "id",
        "text",
        "keyword",
        "location",
        "y_true",
        "y_pred",
        "score",
        "model_name",
        "ticket",
    ]
    return selected.loc[:, columns].reset_index(drop=True)


def main() -> int:
    args = _arguments()
    if args.repetitions != 2:
        raise ValueError("this reproducibility run requires exactly two repetitions")

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    if split.seed != settings.seed:
        raise ValueError("reproducibility seed does not match fixed split seed")
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")

    floor_first = evaluate_floor_model(train, dev)
    floor_second = evaluate_floor_model(train, dev)
    assert_identical_evaluations(floor_first, floor_second)

    configure_reproducibility(settings)
    first_model, reference_first = fit_and_evaluate_reference_baseline(
        train, dev, settings
    )
    configure_reproducibility(settings)
    _, reference_second = fit_and_evaluate_reference_baseline(train, dev, settings)
    assert_identical_evaluations(reference_first, reference_second)

    output_dir = args.output_dir.resolve()
    predictions_dir = output_dir / "predictions"
    results_dir = output_dir / "results"
    expected_dev_ids = list(split.dev_ids)
    evaluations = [floor_first, reference_first]
    for evaluation in evaluations:
        write_prediction_artifact(
            evaluation.predictions,
            predictions_dir / f"{evaluation.model_name}_dev_predictions.csv",
            expected_ids=expected_dev_ids,
        )
        for error_type in ("false_positives", "false_negatives"):
            write_csv_artifact(
                _error_examples(dev, evaluation, error_type=error_type),
                results_dir / f"{evaluation.model_name}_{error_type}.csv",
            )

    write_csv_artifact(_metrics_table(evaluations), results_dir / "dev_metrics.csv")
    write_csv_artifact(
        _confusion_table(evaluations), results_dir / "dev_confusion_matrices.csv"
    )
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact(
        {
            "floor_run_1": list(floor_first.warnings),
            "floor_run_2": list(floor_second.warnings),
            "reference_run_1": list(reference_first.warnings),
            "reference_run_2": list(reference_second.warnings),
        },
        output_dir / "warnings.json",
    )

    command = subprocess.list2cmdline(
        [sys.executable, "-m", "pipeline.run_baselines", *sys.argv[1:]]
    )
    template_model = make_reference_pipeline(settings)
    run_config = {
        "scope": "floor and reference baseline; train fit and dev evaluation only",
        "ticket_1_heldout_status": "not_frozen_and_not_evaluated",
        "baseline_selection_rationale": (
            "The project specifies raw-text TF-IDF plus Logistic Regression but no "
            "additional settings, so this run uses the version-locked sklearn defaults "
            "and only supplies the project seed to random_state."
        ),
        "exact_command": command,
        "data_path": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "split_path": str(args.split.resolve()),
        "split_sha256": _sha256(args.split),
        "seed": settings.seed,
        "n_jobs": settings.n_jobs,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "input_feature": "raw text column only",
        "label_column": "target",
        "metadata_features": [],
        "manual_normalization": None,
        "class_weight": None,
        "threshold_selection": None,
        "effective_regularization": (
            "L2: scikit-learn 1.9.0 represents the deprecated penalty default as "
            "'deprecated' and its l1_ratio=0.0 gives pure L2 regularization."
        ),
        "prediction_rule": "sklearn Pipeline.predict with the fitted LogisticRegression",
        "score": "predict_proba probability for classifier class 1",
        "repetitions": args.repetitions,
        "identical_floor_predictions_and_metrics": True,
        "identical_reference_predictions_scores_and_metrics": True,
        "reference_converged": reference_first.converged,
        "reference_n_iter": list(reference_first.n_iter or ()),
        "effective_parameters": _effective_parameters(template_model),
    }
    write_json_artifact(run_config, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    write_json_artifact(
        {
            "hypothesis": (
                "The instructor floor verifies wiring, while an untuned raw-text "
                "TF-IDF plus Logistic Regression pipeline provides a stronger comparator."
            ),
            "interpretation": (
                "The raw-text baseline's dev metrics exceed the all-zero floor; "
                "held-out agreement with the reference contract was not tested."
            ),
            "limitation": (
                "This run is dev-only and cannot establish reproduction of the "
                "held-out reference F1 until the Ticket 1 configuration is explicitly frozen."
            ),
            "heldout_predictions_created": False,
            "heldout_metrics_computed": False,
        },
        output_dir / "run_notes.json",
    )

    print(_metrics_table(evaluations).to_string(index=False))
    print(f"artifacts={output_dir}")
    print("reproducibility_check=PASS")
    print("heldout_evaluated=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
