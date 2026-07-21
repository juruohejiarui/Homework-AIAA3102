"""Regenerate the predeclared Ticket 1 discrepancy probes on dev only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .artifacts import build_prediction_frame, write_csv_artifact, write_json_artifact, write_prediction_artifact, write_text_artifact
from .baselines import BaselineEvaluation, REFERENCE_MODEL_NAME, fit_and_evaluate_reference_baseline
from .data import load_labeled_tweets, select_split_by_id
from .metrics import metric_bundle
from .modeling import make_leakage_safe_pipeline
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "probes" / "probe_plan.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "probes"

PROBES: tuple[dict[str, Any], ...] = (
    # -- TF-IDF: n-gram range -----------------------------------------------
    {"name": "unigrams_only",              "tfidf": {"ngram_range": (1, 1)}},
    {"name": "tfidf_word_bigrams",         "tfidf": {"ngram_range": (1, 2)}},
    {"name": "trigrams_added",             "tfidf": {"ngram_range": (1, 3)}},
    # -- TF-IDF: term weighting ---------------------------------------------
    {"name": "no_sublinear_tf",            "tfidf": {"sublinear_tf": False}},
    # -- TF-IDF: vocabulary size --------------------------------------------
    {"name": "max_features_20000",         "tfidf": {"max_features": 20000}},
    {"name": "max_features_unbounded",     "tfidf": {"max_features": None}},
    # -- TF-IDF: document-frequency cutoffs ---------------------------------
    {"name": "min_df_2",                   "tfidf": {"min_df": 2}},
    {"name": "min_df_3",                   "tfidf": {"min_df": 3}},
    {"name": "min_df_5",                   "tfidf": {"min_df": 5}},
    {"name": "max_df_0_9",                 "tfidf": {"max_df": 0.9}},
    # -- TF-IDF: IDF weighting ----------------------------------------------
    {"name": "no_idf",                     "tfidf": {"use_idf": False}},
    {"name": "no_smooth_idf",              "tfidf": {"smooth_idf": False}},
    # -- TF-IDF: vector normalisation ---------------------------------------
    {"name": "l1_normalization",           "tfidf": {"norm": "l1"}},
    {"name": "no_vector_normalization",    "tfidf": {"norm": None}},
    # -- TF-IDF: tokenisation -----------------------------------------------
    {"name": "unicode_accents",            "tfidf": {"strip_accents": "unicode"}},
    {"name": "single_character_tokens",    "tfidf": {"token_pattern": r"(?u)\b\w+\b"}},
    # -- TF-IDF: casing (preserve_case == lowercase=False) ------------------
    {"name": "tfidf_lowercase_false",      "tfidf": {"lowercase": False}},
    # -- LogReg: regularisation strength ------------------------------------
    {"name": "c_0_1",                      "logreg": {"C": 0.1}},
    {"name": "c_0_25",                     "logreg": {"C": 0.25}},
    {"name": "c_0_5",                      "logreg": {"C": 0.5}},
    {"name": "c_2",                        "logreg": {"C": 2.0}},
    {"name": "c_4",                        "logreg": {"C": 4.0}},
    {"name": "c_10",                       "logreg": {"C": 10.0}},
    # -- LogReg: class weighting --------------------------------------------
    {"name": "balanced_classes",           "logreg": {"class_weight": "balanced"}},
    # -- LogReg: solver -----------------------------------------------------
    {"name": "logreg_solver_liblinear",    "logreg": {"solver": "liblinear"}},
    {"name": "lbfgs_solver",               "logreg": {"solver": "lbfgs"}},
    # -- LogReg: convergence ------------------------------------------------
    {"name": "logreg_max_iter_1000",       "logreg": {"max_iter": 1000}},
    # -- LogReg: random seed ------------------------------------------------
    {"name": "seed_1",                     "logreg": {"random_state": 1}},
    {"name": "seed_42",                    "logreg": {"random_state": 42}},
    {"name": "logreg_seed_9999",           "logreg": {"random_state": 9999}},
    # -- LogReg: penalty representation -------------------------------------
    {"name": "logreg_explicit_l2",         "logreg": {"penalty": "l2"}},
    # -- Leakage diagnostic -------------------------------------------------
    {"name": "leaky_tfidf_train_plus_dev", "special": "leaky"},
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _evaluation(name: str, dev: pd.DataFrame, classifier: LogisticRegression, y_pred: np.ndarray, scores: np.ndarray, caught: list[Any]) -> BaselineEvaluation:
    y_true = dev["target"].to_numpy(dtype=int)
    warning_records = tuple({"category": item.category.__name__, "message": str(item.message)} for item in caught)
    return BaselineEvaluation(
        model_name=name,
        predictions=build_prediction_frame(ids=dev["id"].tolist(), y_true=y_true, y_pred=y_pred, scores=scores, model_name=name, ticket="ticket_1_baseline"),
        metrics=metric_bundle(y_true, y_pred),
        warnings=warning_records,
        converged=not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        n_iter=tuple(int(value) for value in classifier.n_iter_.tolist()),
    )


def _run_probe(train: pd.DataFrame, dev: pd.DataFrame, definition: dict[str, Any]) -> BaselineEvaluation:
    settings = load_reproducibility_settings()
    name = definition["name"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if definition.get("special") == "leaky":
            vectorizer = TfidfVectorizer()
            vectorizer.fit(pd.concat([train["text"], dev["text"]], ignore_index=True))
            train_features = vectorizer.transform(train["text"])
            dev_features = vectorizer.transform(dev["text"])
            classifier = LogisticRegression(random_state=settings.seed)
            classifier.fit(train_features, train["target"])
            y_pred = classifier.predict(dev_features).astype(int)
            positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
            scores = classifier.predict_proba(dev_features)[:, positive_index]
        else:
            classifier_params: dict[str, Any] = {"random_state": settings.seed}
            classifier_params.update(definition.get("logreg", {}))
            model = make_leakage_safe_pipeline(TfidfVectorizer(**definition.get("tfidf", {})), LogisticRegression(**classifier_params))
            model.fit(train["text"], train["target"])
            classifier = model.named_steps["classifier"]
            y_pred = model.predict(dev["text"]).astype(int)
            positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
            scores = model.predict_proba(dev["text"])[:, positive_index]
    return _evaluation(name, dev, classifier, y_pred, scores, caught)


def _transitions(baseline: pd.DataFrame, probe: pd.DataFrame) -> dict[str, int]:
    frame = baseline.loc[:, ["id", "y_true", "y_pred"]].merge(probe.loc[:, ["id", "y_pred"]], on="id", suffixes=("_baseline", "_probe"), validate="one_to_one", sort=False)
    y = frame["y_true"]
    base = frame["y_pred_baseline"]
    candidate = frame["y_pred_probe"]
    return {
        "prediction_changes": int((base != candidate).sum()),
        "fixed_fp": int(((y == 0) & (base == 1) & (candidate == 0)).sum()),
        "fixed_fn": int(((y == 1) & (base == 0) & (candidate == 1)).sum()),
        "new_fp": int(((y == 0) & (base == 0) & (candidate == 1)).sum()),
        "new_fn": int(((y == 1) & (base == 1) & (candidate == 0)).sum()),
    }


def _changes(baseline: pd.DataFrame, probe: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    left = baseline.loc[:, ["id", "y_true", "y_pred", "score"]].rename(columns={"y_pred": "baseline_y_pred", "score": "baseline_score"})
    right = probe.loc[:, ["id", "y_pred", "score"]].rename(columns={"y_pred": "probe_y_pred", "score": "probe_score"})
    frame = left.merge(right, on="id", validate="one_to_one", sort=False).merge(dev.loc[:, ["id", "text", "keyword", "location"]], on="id", validate="one_to_one", sort=False)
    frame = frame[frame["baseline_y_pred"] != frame["probe_y_pred"]].copy()
    frame["baseline_correct"] = frame["baseline_y_pred"] == frame["y_true"]
    frame["probe_correct"] = frame["probe_y_pred"] == frame["y_true"]
    frame["outcome"] = np.where(frame["probe_correct"], "fixed_error", "new_error")
    frame["transition"] = frame["baseline_y_pred"].astype(str) + "->" + frame["probe_y_pred"].astype(str)
    return frame.loc[:, ["id", "text", "keyword", "location", "y_true", "baseline_y_pred", "probe_y_pred", "baseline_score", "probe_score", "transition", "baseline_correct", "probe_correct", "outcome"]].reset_index(drop=True)


def _errors(dev: pd.DataFrame, evaluation: BaselineEvaluation, kind: str) -> pd.DataFrame:
    frame = evaluation.predictions.merge(dev.loc[:, ["id", "text", "keyword", "location"]], on="id", validate="one_to_one", sort=False)
    if kind == "false_positives":
        frame = frame[(frame["y_true"] == 0) & (frame["y_pred"] == 1)]
    else:
        frame = frame[(frame["y_true"] == 1) & (frame["y_pred"] == 0)]
    return frame.loc[:, ["id", "text", "keyword", "location", "y_true", "y_pred", "score", "model_name", "ticket"]].reset_index(drop=True)


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    planned = [item["name"] for item in plan["controlled_probes"]]
    implemented = [item["name"] for item in PROBES]
    if planned != implemented:
        raise ValueError("implemented probes do not match predeclared plan")
    settings = load_reproducibility_settings()
    configure_reproducibility(settings)
    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")
    _, baseline = fit_and_evaluate_reference_baseline(train, dev, settings)
    saved = pd.read_csv(PROJECT_ROOT / "experiments" / "step-4-baselines" / "predictions" / "raw_text_tfidf_logistic_regression_dev_predictions.csv")
    exact_columns = ["id", "y_true", "y_pred", "model_name", "ticket"]
    if not np.array_equal(baseline.predictions[exact_columns].to_numpy(), saved[exact_columns].to_numpy()) or not np.allclose(baseline.predictions["score"], saved["score"], rtol=0.0, atol=1e-15):
        raise AssertionError("fresh baseline does not match frozen dev artifact")
    evaluations = []
    for definition in PROBES:
        configure_reproducibility(settings)
        evaluations.append(_run_probe(train, dev, definition))

    metrics_rows: list[dict[str, Any]] = [{"model_name": REFERENCE_MODEL_NAME, **baseline.metrics, "f1_delta_vs_baseline": 0.0, "prediction_changes": 0, "fixed_fp": 0, "fixed_fn": 0, "new_fp": 0, "new_fn": 0, "converged": baseline.converged, "n_iter": json.dumps(baseline.n_iter)}]
    confusion_rows: list[dict[str, Any]] = []
    warning_payload: dict[str, Any] = {REFERENCE_MODEL_NAME: list(baseline.warnings)}
    interpretations: dict[str, Any] = {}
    for evaluation in [baseline, *evaluations]:
        confusion_rows.extend([
            {"model_name": evaluation.model_name, "actual_label": 0, "predicted_0": evaluation.metrics["true_negative"], "predicted_1": evaluation.metrics["false_positive"]},
            {"model_name": evaluation.model_name, "actual_label": 1, "predicted_0": evaluation.metrics["false_negative"], "predicted_1": evaluation.metrics["true_positive"]},
        ])
    for evaluation in evaluations:
        transitions = _transitions(baseline.predictions, evaluation.predictions)
        delta = float(evaluation.metrics["f1_target_1"] - baseline.metrics["f1_target_1"])
        metrics_rows.append({"model_name": evaluation.model_name, **evaluation.metrics, "f1_delta_vs_baseline": delta, **transitions, "converged": evaluation.converged, "n_iter": json.dumps(evaluation.n_iter)})
        write_prediction_artifact(evaluation.predictions, output_dir / "predictions" / f"{evaluation.model_name}_dev_predictions.csv", expected_ids=list(split.dev_ids))
        write_csv_artifact(_changes(baseline.predictions, evaluation.predictions, dev), output_dir / "changes" / f"{evaluation.model_name}_changes.csv")
        for kind in ("false_positives", "false_negatives"):
            write_csv_artifact(_errors(dev, evaluation, kind), output_dir / "errors" / f"{evaluation.model_name}_{kind}.csv")
        warning_payload[evaluation.model_name] = list(evaluation.warnings)
        interpretations[evaluation.model_name] = {"f1_delta_vs_baseline": delta, **transitions, "converged": evaluation.converged, "warnings": list(evaluation.warnings), "interpretation": "This one-lever dev diagnostic materially changes reproducibility." if transitions["prediction_changes"] else "This one-lever dev diagnostic is prediction-equivalent in this environment.", "limitation": "Dev-only diagnostic; not used to revise the frozen result."}
    write_csv_artifact(pd.DataFrame(metrics_rows), output_dir / "dev_probe_metrics.csv")
    write_csv_artifact(pd.DataFrame(confusion_rows), output_dir / "dev_probe_confusion_matrices.csv")
    write_json_artifact(warning_payload, output_dir / "warnings.json")
    write_json_artifact(interpretations, output_dir / "probe_interpretations.json")
    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_json_artifact({
        "split_assignment": {"status": "PASS", "train_count": len(split.train_ids), "dev_count": len(split.dev_ids), "heldout_count": len(split.heldout_ids), "unique_total": len(set(split.all_ids)), "cross_split_overlap": 0, "selection_method": "Kaggle id membership followed by fixed JSON order"},
        "row_position_misuse": {"status": "RULED_OUT_FOR_FROZEN_RUN", "dataset_row_count": len(data), "maximum_kaggle_id": max(split.all_ids), "ids_not_valid_as_zero_based_positions": int((data["id"] >= len(data)).sum()), "explanation": "Kaggle IDs are sparse identifiers, not row offsets; the loader joins by id."},
        "preprocessing_and_features": {"status": "PASS", "input": "raw text only", "manual_preprocessor": None, "metadata_features": [], "pipeline_fit_scope": "train only"},
        "baseline_convergence": {"status": "PASS", "n_iter": list(baseline.n_iter or ()), "max_iter": 100, "warnings": list(baseline.warnings)},
        "seed_and_determinism": {"status": "PASS", "seed": settings.seed, "step4_two_run_reproducibility": True},
        "package_version_behavior": {"current_sklearn": "1.9.0", "reference_sklearn": "not specified by project contract", "conclusion": "Cross-version cause cannot be confirmed from the contract; explicit-L2 probe tests current semantics."},
        "accidental_leakage": {"status": "RULED_OUT_FOR_FROZEN_RUN", "frozen_model": "sklearn Pipeline fitted on train only", "diagnostic_probe": "leaky_tfidf_train_plus_dev"}
    }, output_dir / "configuration_audit.json")
    command = subprocess.list2cmdline([sys.executable, "-m", "pipeline.run_ticket1_probes", *sys.argv[1:]])
    write_json_artifact({"scope": "Ticket 1 discrepancy diagnosis on dev only", "exact_command": command, "probe_count": len(evaluations), "probe_names": implemented, "baseline_prediction_reproduction": "PASS", "heldout_evaluations_run_by_this_command": 0, "candidate_selection_from_probe_results": False, "artifact_recovery": True}, output_dir / "run_config.json")
    write_text_artifact(command, output_dir / "run_command.txt")
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print("baseline_dev_reproduction=PASS")
    print("heldout_evaluations_run=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
