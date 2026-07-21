"""Run the predeclared Ticket 3 feature and shortcut audit on dev only."""

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

from .artifacts import write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .baselines import evaluate_floor_model, fit_and_evaluate_reference_baseline
from .data import load_labeled_tweets, select_split_by_id
from .metrics import metric_bundle
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .shortcut_features import (
    TICKET_NAME,
    VARIANT_COMPONENTS,
    VARIANT_NAMES,
    coefficient_rows,
    evaluate_perturbation,
    fit_and_evaluate_shortcut_variant,
    mask_keyword,
    mask_location,
    neutralize_superficial_text,
)
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .ticket2 import error_rows, prediction_change_rows, transition_counts
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "experiments" / "ticket-3" / "dev" / "experiment_plan.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-3" / "dev"
BASELINE_DEV_PREDICTIONS = PROJECT_ROOT / "experiments" / "step-4-baselines" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv"


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


def _changed_perturbation_rows(
    *,
    model: Any,
    original: pd.DataFrame,
    perturbed: pd.DataFrame,
    variant: str,
    perturbation: str,
) -> list[dict[str, Any]]:
    original_pred = model.predict(original).astype(int)
    perturbed_pred = model.predict(perturbed).astype(int)
    rows = []
    for index in np.flatnonzero(original_pred != perturbed_pred):
        rows.append(
            {
                "variant": variant,
                "perturbation": perturbation,
                "id": int(original.iloc[index]["id"]),
                "y_true": int(original.iloc[index]["target"]),
                "original_prediction": int(original_pred[index]),
                "perturbed_prediction": int(perturbed_pred[index]),
                "text": original.iloc[index]["text"],
                "keyword": original.iloc[index]["keyword"],
                "location": original.iloc[index]["location"],
            }
        )
    return rows


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if tuple(item["name"] for item in plan["variants"]) != VARIANT_NAMES:
        raise ValueError("Ticket 3 plan variants do not match implementation")
    unexpected = [path for path in output_dir.rglob("*") if path.is_file() and path.resolve() != args.plan.resolve()]
    if unexpected:
        raise RuntimeError("Ticket 3 dev output exists; refusing overwrite")

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")

    floor = evaluate_floor_model(train, dev)
    floor.predictions["ticket"] = TICKET_NAME
    configure_reproducibility(settings)
    text_model, text_control = fit_and_evaluate_reference_baseline(train, dev, settings)
    text_control.predictions["ticket"] = TICKET_NAME
    saved = pd.read_csv(BASELINE_DEV_PREDICTIONS)
    exact = ["id", "y_true", "y_pred", "model_name"]
    if not np.array_equal(text_control.predictions[exact].to_numpy(), saved[exact].to_numpy()) or not np.allclose(text_control.predictions["score"], saved["score"], rtol=0.0, atol=1e-15):
        raise AssertionError("Ticket 3 text control does not reproduce frozen baseline")

    models: dict[str, Any] = {"raw_text_tfidf_logistic_regression": text_model}
    evaluations: dict[str, Any] = {
        "train_majority_floor": floor,
        "raw_text_tfidf_logistic_regression": text_control,
    }
    for variant in VARIANT_COMPONENTS:
        configure_reproducibility(settings)
        model, evaluation = fit_and_evaluate_shortcut_variant(train, dev, variant, settings)
        models[variant] = model
        evaluations[variant] = evaluation

    hypothesis = {item["name"]: item["hypothesis"] for item in plan["variants"]}
    metrics_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    warning_payload: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    coefficient_output: list[dict[str, Any]] = []
    for variant in VARIANT_NAMES:
        evaluation = evaluations[variant]
        transitions = transition_counts(text_control.predictions, evaluation.predictions)
        delta = float(evaluation.metrics["f1_target_1"] - text_control.metrics["f1_target_1"])
        metrics_rows.append({"variant": variant, **evaluation.metrics, "f1_delta_vs_frozen_baseline": delta, **transitions, "converged": evaluation.converged, "n_iter": "" if evaluation.n_iter is None else json.dumps(evaluation.n_iter)})
        confusion_rows.extend([
            {"variant":variant,"actual_label":0,"predicted_0":evaluation.metrics["true_negative"],"predicted_1":evaluation.metrics["false_positive"]},
            {"variant":variant,"actual_label":1,"predicted_0":evaluation.metrics["false_negative"],"predicted_1":evaluation.metrics["true_positive"]},
        ])
        write_prediction_artifact(evaluation.predictions, output_dir / "predictions" / f"{variant}_dev_predictions.csv", expected_ids=list(split.dev_ids))
        write_csv_artifact(prediction_change_rows(text_control.predictions, evaluation.predictions, dev), output_dir / "changes" / f"{variant}_changes.csv")
        for kind in ("false_positives", "false_negatives"):
            write_csv_artifact(error_rows(dev, evaluation, kind), output_dir / "errors" / f"{variant}_{kind}.csv")
        warning_payload[variant] = list(evaluation.warnings)
        notes[variant] = {"hypothesis": hypothesis[variant], "interpretation": f"Dev F1 delta versus frozen text baseline is {delta:.12f}; inspect transitions, coefficients, and perturbations before judging legitimacy.", "limitation": "One fixed dev split; correlations are not causal and held-out was not accessed by this command.", **transitions}
        if variant in VARIANT_COMPONENTS:
            coefficient_output.extend(coefficient_rows(models[variant], variant))

    perturbations = {
        "mask_keyword": mask_keyword(dev),
        "mask_location": mask_location(dev),
        "mask_keyword_and_location": mask_location(mask_keyword(dev)),
        "neutralize_superficial_text": neutralize_superficial_text(dev),
    }
    robustness_rows: list[dict[str, Any]] = []
    robustness_changes: list[dict[str, Any]] = []
    for variant, components in VARIANT_COMPONENTS.items():
        relevant = []
        if "keyword" in components or "shallow" in components:
            relevant.append("mask_keyword")
        if "location" in components or "shallow" in components:
            relevant.append("mask_location")
        if ("keyword" in components or "shallow" in components) and ("location" in components or "shallow" in components):
            relevant.append("mask_keyword_and_location")
        if "text" in components or "length" in components or "shallow" in components:
            relevant.append("neutralize_superficial_text")
        for perturbation in relevant:
            result = evaluate_perturbation(models[variant], dev, perturbations[perturbation])
            original_f1 = evaluations[variant].metrics["f1_target_1"]
            robustness_rows.append({"variant":variant,"perturbation":perturbation,**result.__dict__,"f1_delta_vs_original":float(result.f1_target_1-original_f1),"f1_delta_vs_frozen_baseline":float(result.f1_target_1-text_control.metrics["f1_target_1"])})
            robustness_changes.extend(_changed_perturbation_rows(model=models[variant], original=dev, perturbed=perturbations[perturbation], variant=variant, perturbation=perturbation))

    superficial = perturbations["neutralize_superficial_text"]
    original_pred = text_model.predict(dev["text"]).astype(int)
    perturbed_pred = text_model.predict(superficial["text"]).astype(int)
    original_scores = text_model.predict_proba(dev["text"])[:, 1]
    perturbed_scores = text_model.predict_proba(superficial["text"])[:, 1]
    surface_metrics = metric_bundle(dev["target"], perturbed_pred)
    robustness_rows.append({"variant":"raw_text_tfidf_logistic_regression","perturbation":"neutralize_superficial_text","affected_rows":int((dev["text"] != superficial["text"]).sum()),"changed_predictions":int((original_pred != perturbed_pred).sum()),"precision_target_1":surface_metrics["precision_target_1"],"recall_target_1":surface_metrics["recall_target_1"],"f1_target_1":surface_metrics["f1_target_1"],"accuracy":surface_metrics["accuracy"],"mean_absolute_score_shift":float(np.abs(original_scores-perturbed_scores).mean()),"maximum_absolute_score_shift":float(np.abs(original_scores-perturbed_scores).max()),"f1_delta_vs_original":float(surface_metrics["f1_target_1"]-text_control.metrics["f1_target_1"]),"f1_delta_vs_frozen_baseline":float(surface_metrics["f1_target_1"]-text_control.metrics["f1_target_1"])})

    write_csv_artifact(pd.DataFrame(metrics_rows), output_dir / "results" / "dev_metrics.csv")
    write_csv_artifact(pd.DataFrame(confusion_rows), output_dir / "results" / "dev_confusion_matrices.csv")
    write_csv_artifact(pd.DataFrame(coefficient_output), output_dir / "interpretability" / "top_coefficients.csv")
    write_csv_artifact(pd.DataFrame(robustness_rows), output_dir / "robustness" / "robustness_metrics.csv")
    columns = ["variant","perturbation","id","y_true","original_prediction","perturbed_prediction","text","keyword","location"]
    write_csv_artifact(pd.DataFrame(robustness_changes, columns=columns), output_dir / "robustness" / "prediction_changes.csv")
    write_csv_artifact(pd.DataFrame([{"split":"train","rows":len(train),"missing_keyword":int(train["keyword"].isna().sum()),"missing_location":int(train["location"].isna().sum()),"unique_nonmissing_keywords":int(train["keyword"].nunique()),"unique_nonmissing_locations":int(train["location"].nunique())},{"split":"dev","rows":len(dev),"missing_keyword":int(dev["keyword"].isna().sum()),"missing_location":int(dev["location"].isna().sum()),"unique_nonmissing_keywords":int(dev["keyword"].nunique()),"unique_nonmissing_locations":int(dev["location"].nunique())}]), output_dir / "results" / "metadata_profile.csv")
    write_json_artifact(warning_payload, output_dir / "warnings.json")
    write_json_artifact(notes, output_dir / "variant_notes.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket3_dev", *sys.argv[1:]])
    write_json_artifact({"scope":"Ticket 3 controlled feature and shortcut audit on dev only","exact_command":command,"data_sha256":_sha256(args.data),"split_sha256":_sha256(args.split),"plan_sha256":_sha256(args.plan),"ticket1_freeze_sha256":_sha256(PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json"),"baseline_dev_predictions_sha256":_sha256(BASELINE_DEV_PREDICTIONS),"seed":settings.seed,"n_jobs":settings.n_jobs,"train_rows":len(train),"dev_rows":len(dev),"variants":list(VARIANT_NAMES),"raw_control_reproduces_frozen_baseline":True,"heldout_rows_loaded":0,"heldout_evaluations_run":0}, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(pd.DataFrame(robustness_rows).to_string(index=False))
    print("raw_control_reproduction=PASS")
    print("heldout_evaluations_run=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
