"""Leakage-safe feature families for the Ticket 3 shortcut audit."""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifacts import build_prediction_frame
from .baselines import BaselineEvaluation
from .metrics import metric_bundle
from .normalization import MENTION_PATTERN, URL_PATTERN
from .reproducibility import ReproducibilitySettings

TICKET_NAME = "ticket_3_shortcuts"
MISSING_KEYWORD = "__MISSING_KEYWORD__"
MISSING_LOCATION = "__MISSING_LOCATION__"

VARIANT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "keyword_only": ("keyword",),
    "length_only": ("length",),
    "keyword_plus_length": ("keyword", "length"),
    "location_only": ("location",),
    "keyword_plus_location": ("keyword", "location"),
    "selected_shallow_only": ("shallow",),
    "text_plus_keyword": ("text", "keyword"),
    "text_plus_selected_shallow_features": ("text", "keyword", "shallow"),
}

VARIANT_NAMES = (
    "train_majority_floor",
    "raw_text_tfidf_logistic_regression",
    *VARIANT_COMPONENTS.keys(),
)


def _frame(values: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(columns)]
    return pd.DataFrame(values, columns=columns)


class LengthFeatureTransformer(TransformerMixin, BaseEstimator):
    """Stateless text-length representation."""

    feature_names = np.asarray(
        ["character_count", "whitespace_token_count", "mean_token_length"],
        dtype=object,
    )

    def fit(self, values: Any, y: object = None) -> "LengthFeatureTransformer":
        del values, y
        return self

    def transform(self, values: Any) -> np.ndarray:
        frame = _frame(values, ("text",))
        rows = []
        for text in frame["text"].astype(str):
            tokens = text.split()
            rows.append(
                [
                    len(text),
                    len(tokens),
                    (sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0,
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names.copy()


class ShallowFeatureTransformer(TransformerMixin, BaseEstimator):
    """Stateless length, surface-count, and missingness features."""

    feature_names = np.asarray(
        [
            "character_count",
            "whitespace_token_count",
            "mean_token_length",
            "url_count",
            "mention_count",
            "hashtag_count",
            "digit_count",
            "punctuation_count",
            "uppercase_letter_ratio",
            "keyword_missing",
            "location_missing",
        ],
        dtype=object,
    )

    def fit(self, values: Any, y: object = None) -> "ShallowFeatureTransformer":
        del values, y
        return self

    def transform(self, values: Any) -> np.ndarray:
        frame = _frame(values, ("text", "keyword", "location"))
        rows = []
        for row in frame.itertuples(index=False):
            text = str(row.text)
            tokens = text.split()
            letters = [character for character in text if character.isalpha()]
            rows.append(
                [
                    len(text),
                    len(tokens),
                    (sum(len(token) for token in tokens) / len(tokens)) if tokens else 0.0,
                    len(URL_PATTERN.findall(text)),
                    len(MENTION_PATTERN.findall(text)),
                    text.count("#"),
                    sum(character.isdigit() for character in text),
                    sum(unicodedata.category(character).startswith("P") for character in text),
                    (sum(character.isupper() for character in letters) / len(letters)) if letters else 0.0,
                    int(pd.isna(row.keyword)),
                    int(pd.isna(row.location)),
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names.copy()


def _categorical(column: str, missing_token: str) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=missing_token)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float64),
            ),
        ]
    )


def make_shortcut_pipeline(
    variant: str,
    settings: ReproducibilitySettings,
) -> Pipeline:
    try:
        components = VARIANT_COMPONENTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown shortcut variant {variant!r}") from error
    transformers: list[tuple[str, Any, Any]] = []
    if "text" in components:
        transformers.append(("text", TfidfVectorizer(), "text"))
    if "keyword" in components:
        transformers.append(("keyword", _categorical("keyword", MISSING_KEYWORD), ["keyword"]))
    if "location" in components:
        transformers.append(("location", _categorical("location", MISSING_LOCATION), ["location"]))
    if "length" in components:
        transformers.append(
            (
                "length",
                Pipeline([("extract", LengthFeatureTransformer()), ("scale", StandardScaler())]),
                ["text"],
            )
        )
    if "shallow" in components:
        transformers.append(
            (
                "shallow",
                Pipeline([("extract", ShallowFeatureTransformer()), ("scale", StandardScaler())]),
                ["text", "keyword", "location"],
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, sparse_threshold=0.3)),
            ("classifier", LogisticRegression(random_state=settings.seed)),
        ]
    )


def fit_and_evaluate_shortcut_variant(
    train: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    variant: str,
    settings: ReproducibilitySettings,
) -> tuple[Pipeline, BaselineEvaluation]:
    model = make_shortcut_pipeline(variant, settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train, train["target"])
    classifier = model.named_steps["classifier"]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    y_true = evaluation_frame["target"].to_numpy(dtype=int)
    y_pred = model.predict(evaluation_frame).astype(int)
    scores = model.predict_proba(evaluation_frame)[:, positive_index]
    warning_records = tuple(
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    )
    evaluation = BaselineEvaluation(
        model_name=variant,
        predictions=build_prediction_frame(
            ids=evaluation_frame["id"].tolist(),
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            model_name=variant,
            ticket=TICKET_NAME,
        ),
        metrics=metric_bundle(y_true, y_pred),
        warnings=warning_records,
        converged=not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        n_iter=tuple(int(value) for value in classifier.n_iter_.tolist()),
    )
    return model, evaluation


def coefficient_rows(model: Pipeline, variant: str, top_per_direction: int = 25) -> list[dict[str, Any]]:
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    rows: list[dict[str, Any]] = []
    groups = np.asarray([name.split("__", 1)[0] for name in names])
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        ordered_positive = indices[np.argsort(coefficients[indices])[::-1]][:top_per_direction]
        ordered_negative = indices[np.argsort(coefficients[indices])][:top_per_direction]
        for direction, ordered in (("positive", ordered_positive), ("negative", ordered_negative)):
            for rank, index in enumerate(ordered, start=1):
                rows.append(
                    {
                        "variant": variant,
                        "feature_group": group,
                        "direction": direction,
                        "rank": rank,
                        "feature": str(names[index]),
                        "coefficient": float(coefficients[index]),
                        "absolute_coefficient": float(abs(coefficients[index])),
                    }
                )
    return rows


def mask_keyword(frame: pd.DataFrame) -> pd.DataFrame:
    perturbed = frame.copy()
    perturbed["keyword"] = np.nan
    return perturbed


def mask_location(frame: pd.DataFrame) -> pd.DataFrame:
    perturbed = frame.copy()
    perturbed["location"] = np.nan
    return perturbed


def neutralize_superficial_text(frame: pd.DataFrame) -> pd.DataFrame:
    perturbed = frame.copy()

    def neutralize(text: str) -> str:
        text = URL_PATTERN.sub(" URLTOKEN ", text)
        text = MENTION_PATTERN.sub(" MENTIONTOKEN ", text)
        text = re.sub(r"#(?=\w)", "", text)
        text = "".join(" " if unicodedata.category(character).startswith("P") else character for character in text)
        return text.casefold()

    perturbed["text"] = perturbed["text"].map(neutralize)
    return perturbed


@dataclass(frozen=True)
class PerturbationResult:
    affected_rows: int
    changed_predictions: int
    precision_target_1: float
    recall_target_1: float
    f1_target_1: float
    accuracy: float
    mean_absolute_score_shift: float
    maximum_absolute_score_shift: float


def evaluate_perturbation(
    model: Pipeline,
    original: pd.DataFrame,
    perturbed: pd.DataFrame,
) -> PerturbationResult:
    affected = (original[["text", "keyword", "location"]].fillna("<NA>") != perturbed[["text", "keyword", "location"]].fillna("<NA>")).any(axis=1).to_numpy()
    original_pred = model.predict(original).astype(int)
    perturbed_pred = model.predict(perturbed).astype(int)
    classifier = model.named_steps["classifier"]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    original_scores = model.predict_proba(original)[:, positive_index]
    perturbed_scores = model.predict_proba(perturbed)[:, positive_index]
    metrics = metric_bundle(original["target"], perturbed_pred)
    shifts = np.abs(original_scores - perturbed_scores)
    return PerturbationResult(
        affected_rows=int(affected.sum()),
        changed_predictions=int((original_pred != perturbed_pred).sum()),
        precision_target_1=float(metrics["precision_target_1"]),
        recall_target_1=float(metrics["recall_target_1"]),
        f1_target_1=float(metrics["f1_target_1"]),
        accuracy=float(metrics["accuracy"]),
        mean_absolute_score_shift=float(shifts.mean()),
        maximum_absolute_score_shift=float(shifts.max()),
    )
