"""Shared Ticket 2 model, evaluation, and artifact helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .artifacts import build_prediction_frame
from .baselines import BaselineEvaluation, REFERENCE_MODEL_NAME
from .metrics import metric_bundle
from .normalization import make_normalizer
from .reproducibility import ReproducibilitySettings

TICKET_NAME = "ticket_2_normalization"


def model_name_for_variant(variant: str) -> str:
    return REFERENCE_MODEL_NAME if variant == "raw_text_control" else variant


def make_ticket2_pipeline(
    variant: str,
    settings: ReproducibilitySettings,
) -> Pipeline:
    """Add exactly one stateless normalizer before the frozen baseline steps."""

    return Pipeline(
        steps=[
            ("normalizer", make_normalizer(variant)),
            ("features", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=settings.seed)),
        ]
    )


def fit_and_evaluate_variant(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    variant: str,
    settings: ReproducibilitySettings,
) -> tuple[Pipeline, BaselineEvaluation]:
    model = make_ticket2_pipeline(variant, settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_frame["text"], train_frame["target"])
    classifier = model.named_steps["classifier"]
    positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    y_true = evaluation_frame["target"].to_numpy(dtype=int)
    y_pred = model.predict(evaluation_frame["text"]).astype(int)
    scores = model.predict_proba(evaluation_frame["text"])[:, positive_class_index]
    warning_records = tuple(
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    )
    evaluation = BaselineEvaluation(
        model_name=model_name_for_variant(variant),
        predictions=build_prediction_frame(
            ids=evaluation_frame["id"].tolist(),
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            model_name=model_name_for_variant(variant),
            ticket=TICKET_NAME,
        ),
        metrics=metric_bundle(y_true, y_pred),
        warnings=warning_records,
        converged=not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        n_iter=tuple(int(value) for value in classifier.n_iter_.tolist()),
    )
    return model, evaluation


def transition_counts(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, int]:
    merged = baseline.loc[:, ["id", "y_true", "y_pred"]].merge(
        candidate.loc[:, ["id", "y_pred"]],
        on="id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
        sort=False,
    )
    y_true = merged["y_true"]
    baseline_pred = merged["y_pred_baseline"]
    candidate_pred = merged["y_pred_candidate"]
    return {
        "prediction_changes": int((baseline_pred != candidate_pred).sum()),
        "fixed_fp": int(
            ((y_true == 0) & (baseline_pred == 1) & (candidate_pred == 0)).sum()
        ),
        "fixed_fn": int(
            ((y_true == 1) & (baseline_pred == 0) & (candidate_pred == 1)).sum()
        ),
        "new_fp": int(
            ((y_true == 0) & (baseline_pred == 0) & (candidate_pred == 1)).sum()
        ),
        "new_fn": int(
            ((y_true == 1) & (baseline_pred == 1) & (candidate_pred == 0)).sum()
        ),
    }


def prediction_change_rows(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    left = baseline.loc[:, ["id", "y_true", "y_pred", "score"]].rename(
        columns={"y_pred": "baseline_y_pred", "score": "baseline_score"}
    )
    right = candidate.loc[:, ["id", "y_pred", "score"]].rename(
        columns={"y_pred": "candidate_y_pred", "score": "candidate_score"}
    )
    merged = (
        left.merge(right, on="id", validate="one_to_one", sort=False)
        .merge(
            context.loc[:, ["id", "text", "keyword", "location"]],
            on="id",
            validate="one_to_one",
            sort=False,
        )
    )
    merged = merged[merged["baseline_y_pred"] != merged["candidate_y_pred"]].copy()
    merged["baseline_correct"] = merged["baseline_y_pred"] == merged["y_true"]
    merged["candidate_correct"] = merged["candidate_y_pred"] == merged["y_true"]
    merged["outcome"] = np.where(
        merged["candidate_correct"], "fixed_error", "new_error"
    )
    merged["transition"] = (
        merged["baseline_y_pred"].astype(str)
        + "->"
        + merged["candidate_y_pred"].astype(str)
    )
    return merged.loc[
        :,
        [
            "id",
            "text",
            "keyword",
            "location",
            "y_true",
            "baseline_y_pred",
            "candidate_y_pred",
            "baseline_score",
            "candidate_score",
            "transition",
            "baseline_correct",
            "candidate_correct",
            "outcome",
        ],
    ].reset_index(drop=True)


def error_rows(
    context: pd.DataFrame,
    evaluation: BaselineEvaluation,
    kind: str,
) -> pd.DataFrame:
    merged = evaluation.predictions.merge(
        context.loc[:, ["id", "text", "keyword", "location"]],
        on="id",
        validate="one_to_one",
        sort=False,
    )
    if kind == "false_positives":
        selected = merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)]
    elif kind == "false_negatives":
        selected = merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)]
    else:
        raise ValueError(f"unknown error kind {kind!r}")
    return selected.loc[
        :,
        [
            "id",
            "text",
            "keyword",
            "location",
            "y_true",
            "y_pred",
            "score",
            "model_name",
            "ticket",
        ],
    ].reset_index(drop=True)


def robustness_comparison(
    *,
    model: Pipeline,
    original_text: pd.Series,
    perturbed_text: pd.Series,
) -> dict[str, int | float | bool]:
    changed_text = original_text.to_numpy() != perturbed_text.to_numpy()
    affected = int(changed_text.sum())
    if affected == 0:
        return {
            "affected_rows": 0,
            "changed_predictions": 0,
            "prediction_invariance_rate": 1.0,
            "mean_absolute_score_shift": 0.0,
            "maximum_absolute_score_shift": 0.0,
            "scores_invariant_at_1e_12": True,
        }
    original_pred = model.predict(original_text).astype(int)[changed_text]
    perturbed_pred = model.predict(perturbed_text).astype(int)[changed_text]
    classifier = model.named_steps["classifier"]
    positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    original_scores = model.predict_proba(original_text)[:, positive_class_index][changed_text]
    perturbed_scores = model.predict_proba(perturbed_text)[:, positive_class_index][changed_text]
    absolute_shift = np.abs(original_scores - perturbed_scores)
    prediction_changes = int((original_pred != perturbed_pred).sum())
    return {
        "affected_rows": affected,
        "changed_predictions": prediction_changes,
        "prediction_invariance_rate": float(1.0 - prediction_changes / affected),
        "mean_absolute_score_shift": float(absolute_shift.mean()),
        "maximum_absolute_score_shift": float(absolute_shift.max()),
        "scores_invariant_at_1e_12": bool(np.all(absolute_shift <= 1e-12)),
    }
