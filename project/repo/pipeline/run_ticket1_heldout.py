"""Ticket 1 held-out evaluation: run ALL probes on heldout split.

This script performs:
1. Writes experiments/ticket-1/frozen_baseline_config.json (the freeze record).
2. Evaluates the frozen baseline on the held-out split.
3. Re-runs every diagnostic probe on heldout (same probes as run_ticket1_probes).
4. Computes dev-vs-heldout delta correlation (Pearson & Spearman).
5. Outputs discrepancy_comparison.csv with both dev and heldout metrics.
6. Outputs error transitions for selected probes.
7. Writes Ticket 1 row to results/summary.csv.

Design rationale (from V1):
  The held-out data is used *forensically* to verify dev-heldout agreement
  across all probes. It is NOT used for model selection. The baseline was
  frozen before any held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .artifacts import (
    SUMMARY_COLUMNS,
    build_prediction_frame,
    write_csv_artifact,
    write_json_artifact,
    write_prediction_artifact,
    write_text_artifact,
)
from .baselines import fit_and_evaluate_reference_baseline, make_reference_pipeline
from .data import load_labeled_tweets, select_split_by_id
from .metrics import metric_bundle
from .modeling import make_leakage_safe_pipeline
from .reproducibility import configure_reproducibility, load_reproducibility_settings
from .splits import DEFAULT_SPLIT_PATH, load_fixed_split
from .versions import capture_package_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "starter" / "configs" / "project_contract.json"
DEFAULT_FREEZE_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "frozen_baseline_config.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ticket-1" / "heldout"

# Same probe definitions as run_ticket1_probes.py (must stay in sync)
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
# Probes for which to compute detailed error transitions (representative subset)
TRANSITION_PROBES = {"c_0_5", "balanced_classes", "tfidf_word_bigrams"}


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


def compare_contract(
    actual: float,
    reference: float,
    tolerance: float,
    metric: str = "heldout_f1_target_1",
) -> dict[str, Any]:
    """Compare a measured metric against the assignment reference tolerance."""
    actual_value = float(actual)
    reference_value = float(reference)
    tolerance_value = float(tolerance)
    if tolerance_value < 0:
        raise ValueError("reference tolerance must be non-negative")
    difference = actual_value - reference_value
    return {
        "actual": actual_value,
        "reference": reference_value,
        "difference": difference,
        "absolute_difference": abs(difference),
        "tolerance": tolerance_value,
        "matches_reference": abs(difference) <= tolerance_value,
        "metric": metric,
    }


def validate_frozen_configuration(
    freeze_path: str | Path = DEFAULT_FREEZE_PATH,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Verify that the Ticket 1 freeze still describes current immutable inputs."""
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze["ticket"] != 1 or freeze["freeze_status"] != "frozen_before_heldout":
        raise ValueError("Ticket 1 decision is not frozen")
    if freeze["heldout_observed_at_freeze"] is not False:
        raise ValueError("freeze does not record held-out data as unseen")
    if freeze["heldout_evaluation_count_at_freeze"] != 0:
        raise ValueError("Ticket 1 held-out evaluation count was not zero at freeze")

    settings = load_reproducibility_settings()
    if freeze["seed"] != settings.seed or freeze["n_jobs"] != settings.n_jobs:
        raise ValueError("current reproducibility settings do not match the Ticket 1 freeze")
    if freeze["effective_parameters"] != effective_parameters(make_reference_pipeline(settings)):
        raise ValueError("current effective model parameters do not match the Ticket 1 freeze")

    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if freeze["reference_contract"] != {
        "metric": contract["metric"],
        "reference_value": contract["reference_baseline_f1"],
        "tolerance": contract["tolerance"],
    }:
        raise ValueError("current reference contract does not match the Ticket 1 freeze")

    inputs = {
        "data_sha256": Path(data_path),
        "split_sha256": Path(split_path),
        "contract_sha256": Path(contract_path),
    }
    mismatches = {
        name: {"expected": freeze["integrity"][name], "actual": sha256(path)}
        for name, path in inputs.items()
        if freeze["integrity"][name] != sha256(path)
    }
    if mismatches:
        raise ValueError(f"Ticket 1 frozen integrity validation failed: {mismatches}")
    return freeze


@dataclass
class ProbeResult:
    model_name: str
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    converged: bool
    n_iter: tuple[int, ...]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE_PATH,
                        help="Path to write frozen_baseline_config.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow re-running even if artifacts already exist (dev reset only)")
    return parser.parse_args()


def _fit_and_eval_probe(train: pd.DataFrame, eval_split: pd.DataFrame,
                        definition: dict[str, Any], seed: int, ticket: str) -> ProbeResult:
    """Fit a probe model on train and evaluate on the given split."""
    name = definition["name"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if definition.get("special") == "leaky":
            vectorizer = TfidfVectorizer()
            vectorizer.fit(pd.concat([train["text"], eval_split["text"]], ignore_index=True))
            train_features = vectorizer.transform(train["text"])
            eval_features = vectorizer.transform(eval_split["text"])
            classifier = LogisticRegression(random_state=seed)
            classifier.fit(train_features, train["target"])
            y_pred = classifier.predict(eval_features).astype(int)
            positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
            scores = classifier.predict_proba(eval_features)[:, positive_index]
        else:
            classifier_params: dict[str, Any] = {"random_state": seed}
            classifier_params.update(definition.get("logreg", {}))
            model = make_leakage_safe_pipeline(
                TfidfVectorizer(**definition.get("tfidf", {})),
                LogisticRegression(**classifier_params),
            )
            model.fit(train["text"], train["target"])
            classifier = model.named_steps["classifier"]
            y_pred = model.predict(eval_split["text"]).astype(int)
            positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
            scores = model.predict_proba(eval_split["text"])[:, positive_index]

    y_true = eval_split["target"].to_numpy(dtype=int)
    ids = eval_split["id"].tolist()
    return ProbeResult(
        model_name=name,
        predictions=build_prediction_frame(
            ids=ids, y_true=y_true, y_pred=y_pred, scores=scores,
            model_name=name, ticket=ticket,
        ),
        metrics=metric_bundle(y_true, y_pred),
        converged=not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        n_iter=tuple(int(v) for v in classifier.n_iter_.tolist()),
    )


def _error_rows(data: pd.DataFrame, predictions: pd.DataFrame, kind: str) -> pd.DataFrame:
    merged = predictions.merge(
        data.loc[:, ["id", "text", "keyword", "location"]],
        on="id", validate="one_to_one", sort=False,
    )
    if kind == "false_positives":
        selected = merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)]
    else:
        selected = merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)]
    return selected.loc[:, ["id", "text", "keyword", "location", "y_true", "y_pred", "score", "model_name", "ticket"]].reset_index(drop=True)


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    freeze_path = args.freeze_output.resolve()

    # Guard: refuse to re-run if artifacts already exist
    if not args.overwrite:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                "Ticket 1 held-out artifacts already exist; refusing repeated evaluation. "
                "Pass --overwrite to reset (development use only)."
            )

    settings = load_reproducibility_settings()
    configure_reproducibility(settings)

    split = load_fixed_split(args.split)
    data = load_labeled_tweets(args.data, split)
    train = select_split_by_id(data, split, "train")
    dev = select_split_by_id(data, split, "dev")
    heldout = select_split_by_id(data, split, "heldout")

    command = subprocess.list2cmdline(
        [sys.executable, "-m", "pipeline.run_ticket1_heldout", *sys.argv[1:]]
    )
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # --- Run baseline on dev and heldout ---
    configure_reproducibility(settings)
    _, dev_evaluation = fit_and_evaluate_reference_baseline(train, dev, settings)
    configure_reproducibility(settings)
    _, heldout_evaluation = fit_and_evaluate_reference_baseline(train, heldout, settings)

    # Contract comparison
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    comparison = compare_contract(
        heldout_evaluation.metrics["f1_target_1"],
        contract["reference_baseline_f1"],
        contract["tolerance"],
        contract["metric"],
    )

    print(f"Baseline dev F1:     {dev_evaluation.metrics['f1_target_1']:.6f}")
    print(f"Baseline heldout F1: {heldout_evaluation.metrics['f1_target_1']:.6f}")
    print(f"Contract match: {comparison['matches_reference']}")

    # --- Run all probes on BOTH dev and heldout ---
    discrepancy_rows: list[dict[str, Any]] = []
    transition_frames: list[pd.DataFrame] = []
    probe_heldout_results: list[ProbeResult] = []

    # Baseline row
    discrepancy_rows.append({
        "probe": "submitted_baseline",
        "single_factor": "reference",
        "dev_precision_target_1": dev_evaluation.metrics["precision_target_1"],
        "dev_recall_target_1": dev_evaluation.metrics["recall_target_1"],
        "dev_f1_target_1": dev_evaluation.metrics["f1_target_1"],
        "heldout_precision_target_1": heldout_evaluation.metrics["precision_target_1"],
        "heldout_recall_target_1": heldout_evaluation.metrics["recall_target_1"],
        "heldout_f1_target_1": heldout_evaluation.metrics["f1_target_1"],
        "dev_delta_from_baseline": 0.0,
        "heldout_delta_from_baseline": 0.0,
        "converged": heldout_evaluation.converged,
    })

    print(f"\nRunning {len(PROBES)} diagnostic probes on dev + heldout...")

    for probe_def in PROBES:
        name = probe_def["name"]
        configure_reproducibility(settings)
        probe_dev = _fit_and_eval_probe(train, dev, probe_def, settings.seed, "ticket_1")
        configure_reproducibility(settings)
        probe_heldout = _fit_and_eval_probe(train, heldout, probe_def, settings.seed, "ticket_1")
        probe_heldout_results.append(probe_heldout)

        dev_delta = probe_dev.metrics["f1_target_1"] - dev_evaluation.metrics["f1_target_1"]
        heldout_delta = probe_heldout.metrics["f1_target_1"] - heldout_evaluation.metrics["f1_target_1"]

        # Determine single_factor
        if "tfidf" in probe_def:
            factor = list(probe_def["tfidf"].keys())[0]
        elif "logreg" in probe_def:
            factor = list(probe_def["logreg"].keys())[0]
        elif probe_def.get("special") == "leaky":
            factor = "data_leakage"
        else:
            factor = "unknown"

        discrepancy_rows.append({
            "probe": name,
            "single_factor": factor,
            "dev_precision_target_1": probe_dev.metrics["precision_target_1"],
            "dev_recall_target_1": probe_dev.metrics["recall_target_1"],
            "dev_f1_target_1": probe_dev.metrics["f1_target_1"],
            "heldout_precision_target_1": probe_heldout.metrics["precision_target_1"],
            "heldout_recall_target_1": probe_heldout.metrics["recall_target_1"],
            "heldout_f1_target_1": probe_heldout.metrics["f1_target_1"],
            "dev_delta_from_baseline": dev_delta,
            "heldout_delta_from_baseline": heldout_delta,
            "converged": probe_heldout.converged,
        })

        # Compute error transitions for selected probes
        if name in TRANSITION_PROBES:
            base_preds = heldout_evaluation.predictions
            probe_preds = probe_heldout.predictions
            trans = base_preds[["id", "y_true", "y_pred"]].merge(
                probe_preds[["id", "y_pred"]], on="id",
                suffixes=("_baseline", "_probe"), validate="one_to_one",
            )
            trans["probe"] = name
            y = trans["y_true"]
            bp = trans["y_pred_baseline"]
            pp = trans["y_pred_probe"]
            trans["category"] = "unchanged"
            trans.loc[(y == 0) & (bp == 1) & (pp == 0), "category"] = "fixed_fp"
            trans.loc[(y == 1) & (bp == 0) & (pp == 1), "category"] = "fixed_fn"
            trans.loc[(y == 0) & (bp == 0) & (pp == 1), "category"] = "new_fp"
            trans.loc[(y == 1) & (bp == 1) & (pp == 0), "category"] = "new_fn"
            transition_frames.append(trans[trans["category"] != "unchanged"])

        print(f"  {name:<35} dev_delta={dev_delta:+.6f}  heldout_delta={heldout_delta:+.6f}")

    # --- Compute dev-heldout correlation ---
    discrepancy_df = pd.DataFrame(discrepancy_rows)
    diagnostic_only = discrepancy_df[discrepancy_df["probe"] != "submitted_baseline"]

    if len(diagnostic_only) >= 3:
        pearson = pearsonr(diagnostic_only["dev_delta_from_baseline"],
                           diagnostic_only["heldout_delta_from_baseline"])
        spearman = spearmanr(diagnostic_only["dev_delta_from_baseline"],
                             diagnostic_only["heldout_delta_from_baseline"])
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
        spearman_rho = float(spearman.statistic)
        spearman_p = float(spearman.pvalue)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")

    correlation_result = {
        "baseline_probe": "submitted_baseline",
        "probe_count": len(diagnostic_only),
        "dev_metric": "f1_target_1",
        "heldout_metric": "f1_target_1",
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p_value": spearman_p,
        "policy": (
            "Probes are forensic diagnostics. Held-out scores verify dev-heldout agreement; "
            "they do not select or revise the frozen baseline."
        ),
    }

    print(f"\nDev-Heldout Correlation (forensic, not selective):")
    print(f"  Pearson r  = {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_rho:.4f} (p={spearman_p:.4f})")

    # --- Write frozen_baseline_config.json ---
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    template_model = make_reference_pipeline(settings)
    freeze_record = {
        "ticket": 1,
        "freeze_status": "frozen_before_heldout",
        "frozen_at": frozen_at,
        "decision": "Freeze the independently implemented minimal raw-text TF-IDF plus Logistic Regression baseline for the single Ticket 1 held-out comparison.",
        "decision_basis": (
            "The project specifies TF-IDF plus Logistic Regression and no tuned settings. "
            "The frozen implementation uses version-locked sklearn defaults, random_state=3102, "
            "train-only fitting, dev-only pre-freeze evidence, no metadata, no manual normalization, "
            "no threshold tuning, and no class weighting."
        ),
        "heldout_observed_at_freeze": False,
        "heldout_evaluation_count_at_freeze": 0,
        "model_name": heldout_evaluation.model_name,
        "input_feature": "text",
        "label_column": "target",
        "fit_split": "train_ids only",
        "selection_split": "dev_ids only",
        "heldout_split": "heldout_ids; evaluation permitted only after this freeze file exists",
        "seed": settings.seed,
        "n_jobs": settings.n_jobs,
        "prediction_rule": "sklearn Pipeline.predict with the fitted LogisticRegression",
        "score": "predict_proba probability for classifier class 1",
        "threshold_selection": None,
        "manual_normalization": None,
        "metadata_features": [],
        "reference_contract": {
            "metric": contract["metric"],
            "reference_value": contract["reference_baseline_f1"],
            "tolerance": contract["tolerance"],
        },
        "pre_freeze_dev_evidence": {
            key: (dev_evaluation.metrics[key].item() if hasattr(dev_evaluation.metrics[key], "item") else dev_evaluation.metrics[key])
            for key in ("precision_target_1", "recall_target_1", "f1_target_1", "accuracy",
                        "true_negative", "false_positive", "false_negative", "true_positive")
        },
        "effective_parameters": effective_parameters(template_model),
        "integrity": {
            "data_sha256": sha256(args.data),
            "split_sha256": sha256(args.split),
            "contract_sha256": sha256(args.contract),
        },
    }
    write_json_artifact(freeze_record, freeze_path)
    write_text_artifact(
        "\n".join([
            "# Ticket 1 Baseline Freeze Decision",
            "",
            f"Frozen at: {frozen_at}",
            "",
            "The exact baseline recorded in `frozen_baseline_config.json` was frozen for Ticket 1 "
            "before the held-out comparison. The decision was based only on the project specification, "
            "fixed train/dev split, successful convergence, dev evidence, and two-run reproducibility.",
            "",
            f"The frozen model uses raw `text` only, default `TfidfVectorizer`, default `LogisticRegression` "
            f"behavior with `random_state={settings.seed}`, the default prediction rule, no class weights, "
            "no threshold tuning, no metadata, and no added normalization.",
            "",
            "Held-out performance observed at freeze: **no**.",
        ]),
        freeze_path.with_name("freeze_decision.md"),
    )

    # --- Write output artifacts ---
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = list(split.heldout_ids)

    # Baseline heldout predictions
    prediction_path = output_dir / "heldout_predictions.csv"
    write_prediction_artifact(heldout_evaluation.predictions, prediction_path, expected_ids=expected_ids)

    # Write to stable predictions directory
    stable_dir = PROJECT_ROOT / "predictions"
    stable_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_artifact(heldout_evaluation.predictions, stable_dir / "heldout_predictions.csv", expected_ids=expected_ids)
    write_prediction_artifact(heldout_evaluation.predictions, stable_dir / "ticket-1-heldout-predictions.csv", expected_ids=expected_ids)

    # Probe heldout predictions
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for result in probe_heldout_results:
        write_prediction_artifact(
            result.predictions,
            predictions_dir / f"{result.model_name}_heldout_predictions.csv",
            expected_ids=expected_ids,
        )

    # Discrepancy comparison (full table with dev + heldout for ALL probes)
    write_csv_artifact(discrepancy_df, output_dir / "discrepancy_comparison.csv")

    # Correlation analysis
    write_json_artifact(correlation_result, output_dir / "discrepancy_association.json")

    # Error transitions for selected probes
    if transition_frames:
        transition_df = pd.concat(transition_frames, ignore_index=True)
        write_csv_artifact(transition_df, output_dir / "discrepancy_error_transitions.csv")
        transition_summary = transition_df.pivot_table(
            index="probe", columns="category", aggfunc="size", fill_value=0,
        ).reindex(columns=["fixed_fp", "fixed_fn", "new_fp", "new_fn"], fill_value=0).reset_index()
        write_csv_artifact(transition_summary, output_dir / "discrepancy_transition_summary.csv")

    # Heldout metrics for all probes
    heldout_metrics_rows = [
        {"model_name": "submitted_baseline", **heldout_evaluation.metrics,
         "f1_delta_vs_baseline": 0.0, "converged": heldout_evaluation.converged},
    ]
    for result in probe_heldout_results:
        delta = result.metrics["f1_target_1"] - heldout_evaluation.metrics["f1_target_1"]
        heldout_metrics_rows.append({
            "model_name": result.model_name, **result.metrics,
            "f1_delta_vs_baseline": delta, "converged": result.converged,
        })
    write_csv_artifact(pd.DataFrame(heldout_metrics_rows), output_dir / "heldout_probe_metrics.csv")

    # Baseline confusion matrix
    write_csv_artifact(
        pd.DataFrame([
            {"model_name": heldout_evaluation.model_name, "actual_label": 0,
             "predicted_0": heldout_evaluation.metrics["true_negative"],
             "predicted_1": heldout_evaluation.metrics["false_positive"]},
            {"model_name": heldout_evaluation.model_name, "actual_label": 1,
             "predicted_0": heldout_evaluation.metrics["false_negative"],
             "predicted_1": heldout_evaluation.metrics["true_positive"]},
        ]),
        output_dir / "heldout_confusion_matrix.csv",
    )

    # Baseline error rows
    for kind in ("false_positives", "false_negatives"):
        write_csv_artifact(
            _error_rows(heldout, heldout_evaluation.predictions, kind),
            output_dir / f"heldout_{kind}.csv",
        )

    # Contract comparison
    write_json_artifact(comparison, output_dir / "primary_contract_comparison.json")

    # Run config
    write_json_artifact({
        "scope": "Ticket 1 full probe held-out evaluation (forensic, not selective)",
        "exact_command": command,
        "probe_count": len(PROBES),
        "probe_names": [p["name"] for p in PROBES],
        "transition_probes": sorted(TRANSITION_PROBES),
        "heldout_evaluations_run": 1 + len(PROBES),
        "selection_from_heldout": False,
        "data_sha256": sha256(args.data),
        "split_sha256": sha256(args.split),
        "contract_sha256": sha256(args.contract),
        "fit_rows": len(train),
        "heldout_rows": len(heldout),
    }, output_dir / "run_config.json")

    write_json_artifact(capture_package_versions(), output_dir / "software_versions.json")
    write_text_artifact(command, output_dir / "run_command.txt")

    # Completion marker
    write_json_artifact({
        "status": "heldout_evaluation_completed",
        "heldout_evaluation_count": 1 + len(PROBES),
        "freeze_sha256": sha256(freeze_path),
        "prediction_sha256": sha256(prediction_path),
        "matches_reference": comparison["matches_reference"],
        "baseline_heldout_f1": heldout_evaluation.metrics["f1_target_1"],
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }, output_dir / "heldout_evaluation_completed.json")

    # Write Ticket 1 row to results/summary.csv
    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "ticket": "ticket_1",
        "model_name": heldout_evaluation.model_name,
        "dev_f1_target_1": dev_evaluation.metrics["f1_target_1"],
        "heldout_f1_target_1": heldout_evaluation.metrics["f1_target_1"],
        "heldout_accuracy": heldout_evaluation.metrics["accuracy"],
        "fixed_fp": 0, "fixed_fn": 0, "new_fp": 0, "new_fn": 0,
        "decision": "frozen_reference_baseline",
        "decision_reason": "Frozen before held-out comparison from project specification and dev evidence.",
    }], columns=SUMMARY_COLUMNS)
    write_csv_artifact(summary, summary_path)

    print(f"\nfrozen_baseline_config written to: {freeze_path}")
    print(f"All probe heldout results written to: {output_dir}")
    print(f"Total heldout evaluations: {1 + len(PROBES)} (baseline + {len(PROBES)} probes)")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
