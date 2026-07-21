"""Reusable binary metrics with target class 1 as the positive class."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ConfusionMatrixCounts:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _validated_targets(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    if true_array.ndim != 1 or pred_array.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true_array.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true and y_pred must have the same length")
    observed = set(np.unique(np.concatenate([true_array, pred_array])).tolist())
    if not observed <= {0, 1}:
        raise ValueError(f"binary labels 0/1 required; observed {sorted(observed)}")
    return true_array, pred_array


def precision_target_1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_array, pred_array = _validated_targets(y_true, y_pred)
    return float(precision_score(true_array, pred_array, pos_label=1, zero_division=0))


def recall_target_1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_array, pred_array = _validated_targets(y_true, y_pred)
    return float(recall_score(true_array, pred_array, pos_label=1, zero_division=0))


def f1_target_1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_array, pred_array = _validated_targets(y_true, y_pred)
    return float(f1_score(true_array, pred_array, pos_label=1, zero_division=0))


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_array, pred_array = _validated_targets(y_true, y_pred)
    return float(accuracy_score(true_array, pred_array))


def confusion_matrix_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> ConfusionMatrixCounts:
    true_array, pred_array = _validated_targets(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(
        true_array, pred_array, labels=[0, 1]
    ).ravel()
    return ConfusionMatrixCounts(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def metric_bundle(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    """Return the shared scalar metrics and confusion counts for one prediction set."""

    counts = confusion_matrix_counts(y_true, y_pred)
    return {
        "precision_target_1": precision_target_1(y_true, y_pred),
        "recall_target_1": recall_target_1(y_true, y_pred),
        "f1_target_1": f1_target_1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        **counts.as_dict(),
    }
