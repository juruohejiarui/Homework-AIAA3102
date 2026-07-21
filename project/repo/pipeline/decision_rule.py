"""Bounded Ticket 4 decision-rule and linear-model experiment helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .artifacts import build_prediction_frame
from .baselines import BaselineEvaluation
from .metrics import metric_bundle
from .reproducibility import ReproducibilitySettings

TICKET_NAME = "ticket_4_decision_rule"
BASELINE_VARIANT = "lr_c1_unweighted_default"
THRESHOLD_VARIANT = "lr_c1_unweighted_tuned_threshold"
THRESHOLDS = tuple(
    float(Decimal("0.20") + Decimal(index) * Decimal("0.01"))
    for index in range(61)
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    classifier: str
    c: float
    class_weight: str | None
    native_threshold: float
    intended_lever: str


MODEL_SPECS = (
    ModelSpec(BASELINE_VARIANT, "logistic_regression", 1.0, None, 0.5, "control"),
    ModelSpec("lr_c025_unweighted_default", "logistic_regression", 0.25, None, 0.5, "regularization"),
    ModelSpec("lr_c05_unweighted_default", "logistic_regression", 0.5, None, 0.5, "regularization"),
    ModelSpec("lr_c2_unweighted_default", "logistic_regression", 2.0, None, 0.5, "regularization"),
    ModelSpec("lr_c4_unweighted_default", "logistic_regression", 4.0, None, 0.5, "regularization"),
    # Extended from v1: higher C values to explore less-regularized regime
    ModelSpec("lr_c5_unweighted_default", "logistic_regression", 5.0, None, 0.5, "regularization"),
    ModelSpec("lr_c10_unweighted_default", "logistic_regression", 10.0, None, 0.5, "regularization"),
    ModelSpec("lr_c1_balanced_default", "logistic_regression", 1.0, "balanced", 0.5, "class_weight"),
    # Extended from v1: balanced + higher C
    ModelSpec("lr_c5_balanced_default", "logistic_regression", 5.0, "balanced", 0.5, "class_weight_and_regularization"),
    ModelSpec("lr_c10_balanced_default", "logistic_regression", 10.0, "balanced", 0.5, "class_weight_and_regularization"),
    ModelSpec("linear_svc_c1_default", "linear_svc", 1.0, None, 0.0, "second_classifier"),
)


def predictions_at_threshold(scores: Any, threshold: float) -> np.ndarray:
    """Apply the documented inclusive probability threshold."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability threshold must be between 0 and 1")
    return (values >= threshold).astype(int)


def make_ticket4_pipeline(
    spec: ModelSpec,
    settings: ReproducibilitySettings,
) -> Pipeline:
    """Hold raw-text TF-IDF fixed while changing one declared classifier lever."""

    if spec.classifier == "logistic_regression":
        classifier = LogisticRegression(
            C=spec.c,
            class_weight=spec.class_weight,
            random_state=settings.seed,
        )
    elif spec.classifier == "linear_svc":
        classifier = LinearSVC(
            C=spec.c,
            class_weight=spec.class_weight,
            random_state=settings.seed,
            dual="auto",
        )
    else:
        raise ValueError(f"unknown classifier {spec.classifier!r}")
    return Pipeline(
        steps=[
            ("features", TfidfVectorizer()),
            ("classifier", classifier),
        ]
    )


def fit_and_evaluate_spec(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    spec: ModelSpec,
    settings: ReproducibilitySettings,
) -> tuple[Pipeline, BaselineEvaluation]:
    """Fit one predeclared model on train and evaluate at its native rule."""

    model = make_ticket4_pipeline(spec, settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_frame["text"], train_frame["target"])
    classifier = model.named_steps["classifier"]
    y_true = evaluation_frame["target"].to_numpy(dtype=int)
    y_pred = model.predict(evaluation_frame["text"]).astype(int)
    if spec.classifier == "logistic_regression":
        positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        scores = model.predict_proba(evaluation_frame["text"])[:, positive_index]
    else:
        scores = model.decision_function(evaluation_frame["text"]).astype(float)
    warning_records = tuple(
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    )
    raw_n_iter = classifier.n_iter_
    n_iter_values = np.atleast_1d(raw_n_iter)
    evaluation = BaselineEvaluation(
        model_name=spec.name,
        predictions=build_prediction_frame(
            ids=evaluation_frame["id"].tolist(),
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            model_name=spec.name,
            ticket=TICKET_NAME,
        ),
        metrics=metric_bundle(y_true, y_pred),
        warnings=warning_records,
        converged=not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        n_iter=tuple(int(value) for value in n_iter_values.tolist()),
    )
    return model, evaluation


def evaluation_from_threshold(
    baseline_evaluation: BaselineEvaluation,
    threshold: float,
) -> BaselineEvaluation:
    """Create a thresholded evaluation without refitting the baseline model."""

    source = baseline_evaluation.predictions
    y_true = source["y_true"].to_numpy(dtype=int)
    y_pred = predictions_at_threshold(source["score"], threshold)
    predictions = build_prediction_frame(
        ids=source["id"].tolist(),
        y_true=y_true,
        y_pred=y_pred,
        scores=source["score"].to_numpy(dtype=float),
        model_name=THRESHOLD_VARIANT,
        ticket=TICKET_NAME,
    )
    return BaselineEvaluation(
        model_name=THRESHOLD_VARIANT,
        predictions=predictions,
        metrics=metric_bundle(y_true, y_pred),
        warnings=baseline_evaluation.warnings,
        converged=baseline_evaluation.converged,
        n_iter=baseline_evaluation.n_iter,
    )


def threshold_sweep_rows(
    baseline_evaluation: BaselineEvaluation,
) -> list[dict[str, float | int | str]]:
    """Return the complete predeclared baseline threshold sweep."""

    source = baseline_evaluation.predictions
    y_true = source["y_true"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        y_pred = predictions_at_threshold(source["score"], threshold)
        metrics = metric_bundle(y_true, y_pred)
        rows.append(
            {
                "ticket": "ticket_4",
                "threshold": threshold,
                **metrics,
            }
        )
    return rows


def select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared F1, distance-to-default, then threshold tie-break."""

    if not rows:
        raise ValueError("threshold rows must not be empty")
    best_f1 = max(float(row["f1_target_1"]) for row in rows)
    tied = [row for row in rows if abs(float(row["f1_target_1"]) - best_f1) <= 1e-12]
    return min(tied, key=lambda row: (abs(float(row["threshold"]) - 0.5), float(row["threshold"])))
