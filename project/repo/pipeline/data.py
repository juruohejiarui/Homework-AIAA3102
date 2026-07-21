"""Validated loading for the labeled Disaster Tweets source CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .splits import FixedSplit, SplitIntegrityError

REQUIRED_COLUMNS = ("id", "keyword", "location", "text", "target")


def load_labeled_tweets(path: str | Path, split: FixedSplit) -> pd.DataFrame:
    """Load the public CSV and require exact coverage of the fixed split IDs."""

    frame = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"dataset is missing columns: {sorted(missing_columns)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise SplitIntegrityError("dataset IDs must be complete and unique")
    if frame["text"].isna().any():
        raise ValueError("raw text contains missing values")
    if frame["target"].isna().any() or set(frame["target"].unique()) != {0, 1}:
        raise ValueError("target must be complete and contain binary labels 0 and 1")

    observed_ids = set(frame["id"].tolist())
    expected_ids = set(split.all_ids)
    missing_ids = expected_ids - observed_ids
    unexpected_ids = observed_ids - expected_ids
    if missing_ids or unexpected_ids:
        raise SplitIntegrityError(
            "dataset IDs do not exactly cover the fixed split; "
            f"missing={sorted(missing_ids)[:5]}, "
            f"unexpected={sorted(unexpected_ids)[:5]}"
        )
    return frame


def select_split_by_id(
    frame: pd.DataFrame,
    split: FixedSplit,
    split_name: str,
) -> pd.DataFrame:
    """Select one named partition by ID in the instructor-provided order."""

    try:
        expected_ids = split.ids_by_split[split_name]
    except KeyError as error:
        raise ValueError(f"unknown split name {split_name!r}") from error
    indexed = frame.set_index("id", drop=False)
    selected = indexed.loc[list(expected_ids)].reset_index(drop=True).copy()
    if tuple(selected["id"].tolist()) != expected_ids:
        raise SplitIntegrityError(f"{split_name} IDs are not in fixed split order")
    return selected
