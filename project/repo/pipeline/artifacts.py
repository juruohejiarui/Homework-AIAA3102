"""Validated artifact builders and atomic writers with stable tweet IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PREDICTION_COLUMNS = ["id", "y_true", "y_pred", "score", "model_name", "ticket"]
SUMMARY_COLUMNS = [
    "ticket",
    "model_name",
    "dev_f1_target_1",
    "heldout_f1_target_1",
    "heldout_accuracy",
    "fixed_fp",
    "fixed_fn",
    "new_fp",
    "new_fn",
    "decision",
    "decision_reason",
]
THRESHOLD_SWEEP_COLUMNS = [
    "ticket",
    "threshold",
    "precision_target_1",
    "recall_target_1",
    "f1_target_1",
]
DATA_QUALITY_AUDIT_COLUMNS = [
    "id",
    "issue_type",
    "evidence",
    "disposition",
    "confidence",
]
DATA_QUALITY_DISPOSITIONS = {
    "fix",
    "keep_but_flag",
    "ambiguous",
    "reject_false_positive",
}


class ArtifactValidationError(ValueError):
    """Raised when an artifact would violate its machine-checkable contract."""


def build_prediction_frame(
    *,
    ids: Sequence[int],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float],
    model_name: str,
    ticket: str,
) -> pd.DataFrame:
    """Build a prediction table without deriving IDs from dataframe positions."""

    lengths = {len(ids), len(y_true), len(y_pred), len(scores)}
    if len(lengths) != 1:
        raise ArtifactValidationError("prediction fields must have equal lengths")
    frame = pd.DataFrame(
        {
            "id": list(ids),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "score": list(scores),
            "model_name": model_name,
            "ticket": ticket,
        },
        columns=PREDICTION_COLUMNS,
    )
    validate_prediction_frame(frame, expected_ids=ids)
    return frame


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_ids: Sequence[int] | None = None,
) -> None:
    """Enforce the exact schema, complete rows, unique IDs, and expected ID order."""

    if list(frame.columns) != PREDICTION_COLUMNS:
        raise ArtifactValidationError(
            f"prediction columns must be exactly {PREDICTION_COLUMNS}"
        )
    if frame.empty:
        raise ArtifactValidationError("prediction artifact must not be empty")
    missing_columns = frame.columns[frame.isna().any()].tolist()
    if missing_columns:
        raise ArtifactValidationError(
            f"prediction artifact has missing values in {missing_columns}"
        )
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].tolist()[:5]
        raise ArtifactValidationError(f"prediction artifact has duplicate IDs: {duplicates}")

    if expected_ids is not None:
        actual = tuple(frame["id"].tolist())
        expected = tuple(expected_ids)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))[:5]
            unexpected = sorted(set(actual) - set(expected))[:5]
            raise ArtifactValidationError(
                "prediction IDs do not match expected coverage/order; "
                f"missing={missing}, unexpected={unexpected}"
            )


def write_prediction_artifact(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    expected_ids: Sequence[int],
) -> Path:
    """Validate then atomically write a prediction CSV."""

    validate_prediction_frame(frame, expected_ids=expected_ids)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_csv_artifact(frame: pd.DataFrame, path: str | Path) -> Path:
    """Atomically write a general tabular artifact without changing row order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_json_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a deterministic UTF-8 JSON configuration artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_text_artifact(text: str, path: str | Path) -> Path:
    """Atomically write a UTF-8 text artifact with one final newline."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(text.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
