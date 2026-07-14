"""Validation metrics shared by evaluation and experiment logging."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    if not targets:
        raise ValueError("metrics require at least one example")
    recalls = recall_score(targets, predictions, labels=range(4), average=None, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=range(4), average="macro")),
    }
    result.update({f"recall_class_{index}": float(value) for index, value in enumerate(recalls)})
    return result
