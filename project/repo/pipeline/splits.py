"""Load, validate, and apply the instructor-provided fixed ID split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "starter" / "data" / "split_indices.json"
SPLIT_NAMES = ("train", "dev", "heldout")


class SplitIntegrityError(ValueError):
    """Raised when split metadata or dataset IDs violate the fixed contract."""


@dataclass(frozen=True)
class FixedSplit:
    """Immutable fixed split whose membership is keyed by Kaggle tweet ID."""

    seed: int
    split_policy: str
    train_ids: tuple[int, ...]
    dev_ids: tuple[int, ...]
    heldout_ids: tuple[int, ...]

    @property
    def ids_by_split(self) -> Mapping[str, tuple[int, ...]]:
        return {
            "train": self.train_ids,
            "dev": self.dev_ids,
            "heldout": self.heldout_ids,
        }

    @property
    def all_ids(self) -> tuple[int, ...]:
        return self.train_ids + self.dev_ids + self.heldout_ids

    def validate(self) -> None:
        """Reject empty, malformed, duplicate, or overlapping split IDs."""

        id_sets: dict[str, set[int]] = {}
        for name, ids in self.ids_by_split.items():
            if not ids:
                raise SplitIntegrityError(f"{name}_ids is empty")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
                raise SplitIntegrityError(f"{name}_ids must contain integers only")
            if len(set(ids)) != len(ids):
                raise SplitIntegrityError(f"{name}_ids contains duplicate IDs")
            id_sets[name] = set(ids)

        for left_index, left_name in enumerate(SPLIT_NAMES):
            for right_name in SPLIT_NAMES[left_index + 1 :]:
                overlap = id_sets[left_name] & id_sets[right_name]
                if overlap:
                    preview = sorted(overlap)[:5]
                    raise SplitIntegrityError(
                        f"{left_name} and {right_name} overlap at IDs {preview}"
                    )


def load_fixed_split(path: str | Path = DEFAULT_SPLIT_PATH) -> FixedSplit:
    """Parse the fixed split file without regenerating or reordering any IDs."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required_keys = {
        "seed",
        "split_policy",
        "train_ids",
        "dev_ids",
        "heldout_ids",
    }
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        raise SplitIntegrityError(f"split file is missing keys: {sorted(missing_keys)}")

    split = FixedSplit(
        seed=int(payload["seed"]),
        split_policy=str(payload["split_policy"]),
        train_ids=tuple(payload["train_ids"]),
        dev_ids=tuple(payload["dev_ids"]),
        heldout_ids=tuple(payload["heldout_ids"]),
    )
    split.validate()
    return split


def partition_frame_by_id(
    frame: pd.DataFrame,
    split: FixedSplit,
    *,
    id_column: str = "id",
    reject_unexpected_ids: bool = True,
) -> dict[str, pd.DataFrame]:
    """Partition a dataframe by stable ID and preserve the stored split order."""

    if id_column not in frame.columns:
        raise SplitIntegrityError(f"dataset is missing required ID column {id_column!r}")
    if frame[id_column].isna().any():
        raise SplitIntegrityError("dataset contains missing IDs")
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].tolist()[:5]
        raise SplitIntegrityError(f"dataset contains duplicate IDs: {duplicates}")

    observed_ids = set(frame[id_column].tolist())
    expected_ids = set(split.all_ids)
    missing_ids = expected_ids - observed_ids
    if missing_ids:
        raise SplitIntegrityError(
            f"dataset is missing {len(missing_ids)} split IDs; "
            f"first IDs: {sorted(missing_ids)[:5]}"
        )
    unexpected_ids = observed_ids - expected_ids
    if reject_unexpected_ids and unexpected_ids:
        raise SplitIntegrityError(
            f"dataset contains {len(unexpected_ids)} unexpected IDs; "
            f"first IDs: {sorted(unexpected_ids)[:5]}"
        )

    indexed = frame.set_index(id_column, drop=False)
    partitions: dict[str, pd.DataFrame] = {}
    for name, ids in split.ids_by_split.items():
        partitions[name] = indexed.loc[list(ids)].reset_index(drop=True).copy()
    return partitions


def assert_id_sequence(actual_ids: Sequence[int], expected_ids: Sequence[int]) -> None:
    """Require exact ID coverage and order for deterministic data flow."""

    actual = tuple(actual_ids)
    expected = tuple(expected_ids)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:5]
        unexpected = sorted(set(actual) - set(expected))[:5]
        raise SplitIntegrityError(
            "ID sequence does not match the fixed split order; "
            f"missing={missing}, unexpected={unexpected}"
        )
