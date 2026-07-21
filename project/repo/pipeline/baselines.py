"""Instructor floor model and minimal raw-text reference baseline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .artifacts import build_prediction_frame
from .metrics import metric_bundle
from .modeling import make_leakage_safe_pipeline
from .reproducibility import ReproducibilitySettings

FLOOR_MODEL_NAME = "train_majority_floor"
REFERENCE_MODEL_NAME = "raw_text_tfidf_logistic_regression"
TICKET_NAME = "ticket_1_baseline"


@dataclass(frozen=True)
class BaselineEvaluation:
    model_name: str
    predictions: pd.DataFrame
    metrics: dict[str, float | int]
    warnings: tuple[dict[str, Any], ...]
    converged: bool
    n_iter: tuple[int, ...] | None


def make_reference_pipeline(settings: ReproducibilitySettings) -> Pipeline:
    """Create the raw-text baseline without tuning or extra preprocessing."""

    return make_leakage_safe_pipeline(
        TfidfVectorizer(),
        LogisticRegression(random_state=settings.seed),
    )


def evaluate_floor_model(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
) -> BaselineEvaluation:
    """Fit the instructor floor model by finding the train-only majority label."""

    label_counts = train_frame["target"].value_counts()
    if label_counts.empty:
        raise ValueError("training labels must not be empty")
    maximum_count = int(label_counts.max())
    tied_labels = sorted(
        int(label) for label, count in label_counts.items() if int(count) == maximum_count
    )
    majority_label = tied_labels[0]
    y_true = dev_frame["target"].to_numpy(dtype=int)
    y_pred = np.full(y_true.shape, majority_label, dtype=int)
    scores = np.full(y_true.shape, float(majority_label), dtype=float)
    predictions = build_prediction_frame(
        ids=dev_frame["id"].tolist(),
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        model_name=FLOOR_MODEL_NAME,
        ticket=TICKET_NAME,
    )
    return BaselineEvaluation(
        model_name=FLOOR_MODEL_NAME,
        predictions=predictions,
        metrics=metric_bundle(y_true, y_pred),
        warnings=(),
        converged=True,
        n_iter=None,
    )


def fit_and_evaluate_reference_baseline(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    settings: ReproducibilitySettings,
) -> tuple[Pipeline, BaselineEvaluation]:
    """Fit on raw train text and evaluate on dev at the default decision rule."""

    model = make_reference_pipeline(settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_frame["text"], train_frame["target"])

    classifier = model.named_steps["classifier"]
    positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    y_true = dev_frame["target"].to_numpy(dtype=int)
    y_pred = model.predict(dev_frame["text"]).astype(int)
    scores = model.predict_proba(dev_frame["text"])[:, positive_class_index]
    warning_records = tuple(
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught
    )
    converged = not any(
        issubclass(warning.category, ConvergenceWarning) for warning in caught
    )
    n_iter = tuple(int(value) for value in classifier.n_iter_.tolist())
    predictions = build_prediction_frame(
        ids=dev_frame["id"].tolist(),
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        model_name=REFERENCE_MODEL_NAME,
        ticket=TICKET_NAME,
    )
    evaluation = BaselineEvaluation(
        model_name=REFERENCE_MODEL_NAME,
        predictions=predictions,
        metrics=metric_bundle(y_true, y_pred),
        warnings=warning_records,
        converged=converged,
        n_iter=n_iter,
    )
    return model, evaluation


def assert_identical_evaluations(
    first: BaselineEvaluation,
    second: BaselineEvaluation,
) -> None:
    """Fail if repeated predictions, scores, metrics, or fit diagnostics differ."""

    if first.model_name != second.model_name:
        raise AssertionError("repeated evaluations used different model names")
    if first.metrics != second.metrics:
        raise AssertionError("repeated evaluations produced different metrics")
    if first.converged != second.converged or first.n_iter != second.n_iter:
        raise AssertionError("repeated evaluations produced different fit diagnostics")
    first_values = first.predictions.to_numpy()
    second_values = second.predictions.to_numpy()
    if not np.array_equal(first_values, second_values):
        raise AssertionError("repeated evaluations produced different prediction rows")
