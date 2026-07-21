"""Generic leakage-safe composition helpers; no ticket model is defined here."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline


def make_leakage_safe_pipeline(
    feature_transformer: TransformerMixin,
    estimator: BaseEstimator,
) -> Pipeline:
    """Compose unfitted feature and model clones into one sklearn Pipeline.

    Fitting this object on the training partition ensures the transformer's
    vocabulary/statistics are learned from training data in the same fit call.
    Dev and held-out data must only be passed to prediction methods.
    """

    return Pipeline(
        steps=[
            ("features", clone(feature_transformer)),
            ("classifier", clone(estimator)),
        ]
    )
