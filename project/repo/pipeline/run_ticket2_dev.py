"""Run the predeclared one-lever Ticket 2 normalization study on dev only."""

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
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .data import load_labeled_tweets, select_split_by_id
from .normalization import (
    NORMALIZATION_NAMES,
    PERTURBATIONS,
    make_normalizer,
)
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import (
    error_rows,
    fit_and_evaluate_variant,
    prediction_change_rows,
    robustness_comparison,
    transition_counts,
)
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = (
    PROJECT_ROOT / "experiments" / "ticket-2" / "dev" / "experiment_plan.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-2" / "dev"
STEP4_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "step-4-baselines"
    / "predictions"
    / "raw_text_tfidf_logistic_regression_dev_predictions.csv"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confusion_rows(variant: str, evaluation: Any) -> list[dict[str, Any]]:
    metrics = evaluation.metrics
    return [
        {
            "variant": variant,
            "model_name": evaluation.model_name,
            "actual_label": 0,
            "predicted_0": metrics["true_negative"],
            "predicted_1": metrics["false_positive"],
        },
        {
            "variant": variant,
            "model_name": evaluation.model_name,
            "actual_label": 1,
            "predicted_0": metrics["false_negative"],
            "predicted_1": metrics["true_positive"],
        },
    ]


def _interpretation(delta: float, transitions: dict[str, int]) -> str:
    if transitions["prediction_changes"] == 0:
        return (
            "Prediction-equivalent to the raw control on visible dev; inspect the paired "
            "perturbation result to determine whether it adds robustness."
        )
    direction = "increased" if delta > 0 else "decreased"
    return (
        f"Changed {transitions['prediction_changes']} dev labels and {direction} target-1 "
        f"F1 by {abs(delta):.12f}; inspect fixed/new errors before judging the mechanism."
    )


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    planned_names = tuple(item["name"] for item in plan["variants"])
    if planned_names != NORMALIZATION_NAMES:
        raise ValueError("experiment plan variants do not match implementation registry")
    unexpected_existing = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != args.plan.resolve()
    ]
    if unexpected_existing:
        raise RuntimeError("Ticket 2 dev output already exists; refusing overwrite")

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    if split.seed != settings.seed:
        raise ValueError("fixed split and reproducibility seeds differ")
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")

    models: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for variant in NORMALIZATION_NAMES:
        configure_reproducibility(settings)
        model, evaluation = fit_and_evaluate_variant(train, dev, variant, settings)
        models[variant] = model
        evaluations[variant] = evaluation

    control = evaluations["raw_text_control"]
    saved_control = pd.read_csv(STEP4_PREDICTIONS)
    exact_columns = ["id", "y_true", "y_pred", "model_name"]
    if not np.array_equal(
        control.predictions[exact_columns].to_numpy(),
        saved_control[exact_columns].to_numpy(),
    ) or not np.allclose(
        control.predictions["score"],
        saved_control["score"],
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("raw Ticket 2 control does not reproduce frozen dev baseline")

    hypotheses = {item["name"]: item["hypothesis"] for item in plan["variants"]}
    metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    warnings_payload: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    for variant in NORMALIZATION_NAMES:
        evaluation = evaluations[variant]
        transitions = transition_counts(control.predictions, evaluation.predictions)
        delta = float(
            evaluation.metrics["f1_target_1"] - control.metrics["f1_target_1"]
        )
        metrics_rows.append(
            {
                "variant": variant,
                "model_name": evaluation.model_name,
                **evaluation.metrics,
                "f1_delta_vs_frozen_baseline": delta,
                **transitions,
                "converged": evaluation.converged,
                "n_iter": json.dumps(evaluation.n_iter),
            }
        )
        confusion_rows.extend(_confusion_rows(variant, evaluation))
        write_prediction_artifact(
            evaluation.predictions,
            output_dir / "predictions" / f"{variant}_dev_predictions.csv",
            expected_ids=list(split.dev_ids),
        )
        write_csv_artifact(
            prediction_change_rows(control.predictions, evaluation.predictions, dev),
            output_dir / "changes" / f"{variant}_changes.csv",
        )
        for kind in ("false_positives", "false_negatives"):
            write_csv_artifact(
                error_rows(dev, evaluation, kind),
                output_dir / "errors" / f"{variant}_{kind}.csv",
            )
        normalizer = make_normalizer(variant)
        normalized_train = normalizer.transform(train["text"])
        normalized_dev = normalizer.transform(dev["text"])
        count_rows.append(
            {
                "variant": variant,
                "train_changed_rows": int(
                    (np.asarray(normalized_train) != train["text"].to_numpy()).sum()
                ),
                "dev_changed_rows": int(
                    (np.asarray(normalized_dev) != dev["text"].to_numpy()).sum()
                ),
            }
        )
        warnings_payload[variant] = list(evaluation.warnings)
        notes[variant] = {
            "hypothesis": hypotheses[variant],
            "interpretation": _interpretation(delta, transitions),
            "limitation": (
                "One fixed dev split and one normalization lever; not evidence for "
                "combinations and not used after held-out access."
            ),
            "f1_delta_vs_frozen_baseline": delta,
            **transitions,
        }

    robustness_rows: list[dict[str, Any]] = []
    robustness_change_rows: list[dict[str, Any]] = []
    for variant, perturb in PERTURBATIONS.items():
        perturbed_text = dev["text"].map(perturb)
        changed_mask = dev["text"].to_numpy() != perturbed_text.to_numpy()
        for evaluated_variant in ("raw_text_control", variant):
            result = robustness_comparison(
                model=models[evaluated_variant],
                original_text=dev["text"],
                perturbed_text=perturbed_text,
            )
            robustness_rows.append(
                {
                    "perturbation_for": variant,
                    "evaluated_variant": evaluated_variant,
                    **result,
                }
            )
            if result["affected_rows"]:
                original_pred = models[evaluated_variant].predict(dev["text"]).astype(int)
                perturbed_pred = models[evaluated_variant].predict(perturbed_text).astype(int)
                changed_predictions = changed_mask & (original_pred != perturbed_pred)
                for row_index in np.flatnonzero(changed_predictions):
                    robustness_change_rows.append(
                        {
                            "perturbation_for": variant,
                            "evaluated_variant": evaluated_variant,
                            "id": int(dev.iloc[row_index]["id"]),
                            "original_text": dev.iloc[row_index]["text"],
                            "perturbed_text": perturbed_text.iloc[row_index],
                            "original_prediction": int(original_pred[row_index]),
                            "perturbed_prediction": int(perturbed_pred[row_index]),
                        }
                    )

    write_csv_artifact(
        pd.DataFrame(metrics_rows), output_dir / "results" / "dev_metrics.csv"
    )
    write_csv_artifact(
        pd.DataFrame(confusion_rows),
        output_dir / "results" / "dev_confusion_matrices.csv",
    )
    write_csv_artifact(
        pd.DataFrame(count_rows),
        output_dir / "results" / "normalization_changed_rows.csv",
    )
    write_csv_artifact(
        pd.DataFrame(robustness_rows),
        output_dir / "robustness" / "robustness_metrics.csv",
    )
    robustness_columns = [
        "perturbation_for",
        "evaluated_variant",
        "id",
        "original_text",
        "perturbed_text",
        "original_prediction",
        "perturbed_prediction",
    ]
    write_csv_artifact(
        pd.DataFrame(robustness_change_rows, columns=robustness_columns),
        output_dir / "robustness" / "prediction_changes.csv",
    )
    write_json_artifact(warnings_payload, output_dir / "warnings.json")
    write_json_artifact(notes, output_dir / "variant_notes.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    command = subprocess.list2cmdline(
        [sys.executable, "-m", "pipeline.run_ticket2_dev", *sys.argv[1:]]
    )
    write_json_artifact(
        {
            "scope": "Ticket 2 raw control and six one-lever normalization variants on dev only",
            "exact_command": command,
            "data_sha256": _sha256(args.data),
            "split_sha256": _sha256(args.split),
            "plan_sha256": _sha256(args.plan),
            "ticket1_freeze_sha256": _sha256(
                PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json"
            ),
            "baseline_dev_predictions_sha256": _sha256(STEP4_PREDICTIONS),
            "seed": settings.seed,
            "n_jobs": settings.n_jobs,
            "train_rows": len(train),
            "dev_rows": len(dev),
            "variants": list(NORMALIZATION_NAMES),
            "raw_control_reproduces_frozen_baseline": True,
            "heldout_rows_loaded": 0,
            "heldout_evaluations_run": 0,
        },
        output_dir / "run_config.json",
    )
    write_text_artifact(command, output_dir / "run_command.txt")
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(pd.DataFrame(robustness_rows).to_string(index=False))
    print("raw_control_reproduction=PASS")
    print("heldout_evaluations_run=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
