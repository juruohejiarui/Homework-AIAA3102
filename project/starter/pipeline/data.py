"""Dataset loading, immutable split assignment, and integrity validation."""
import json
from pathlib import Path
import pandas as pd
from .config import DATA, EXPECTED_COUNTS, EXPECTED_POSITIVES


class DataValidationError(ValueError):
    pass


def load_data(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Follow README data-download instructions.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Malformed CSV {path}: {exc}") from exc
    required = {"id", "keyword", "location", "text", "target"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"CSV missing columns: {sorted(missing)}")
    return df


def load_split_ids(path: Path | None = None) -> dict[str, list[int]]:
    path = path or DATA / "split_indices.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataValidationError(f"Invalid split file {path}: {exc}") from exc
    out = {k: raw.get(f"{k}_ids") for k in ("train", "dev", "heldout")}
    if any(not isinstance(v, list) for v in out.values()):
        raise DataValidationError("Split file must contain train_ids, dev_ids, heldout_ids lists")
    return out


def validate_and_split(df: pd.DataFrame, ids: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    if df["id"].duplicated().any(): raise DataValidationError("Dataset IDs must be unique")
    if not set(df["target"].dropna().unique()).issubset({0, 1}) or df["target"].isna().any():
        raise DataValidationError("target must contain only 0 and 1")
    sets = {k: set(v) for k, v in ids.items()}
    for k, values in ids.items():
        if len(values) != len(sets[k]): raise DataValidationError(f"Duplicate split IDs in {k}")
    if sets["train"] & sets["dev"] or sets["train"] & sets["heldout"] or sets["dev"] & sets["heldout"]:
        raise DataValidationError("Splits must be pairwise disjoint")
    all_ids = set(df["id"].astype(int))
    assigned = set().union(*sets.values())
    if assigned - all_ids: raise DataValidationError(f"Missing split IDs: {sorted(assigned-all_ids)[:5]}")
    if all_ids != assigned: raise DataValidationError(f"Split union does not cover dataset; unassigned={len(all_ids-assigned)}")
    split = {k: df[df.id.isin(v)].sort_values("id").reset_index(drop=True) for k, v in ids.items()}
    for k, part in split.items():
        if len(part) != EXPECTED_COUNTS[k]: raise DataValidationError(f"{k} has {len(part)} rows, expected {EXPECTED_COUNTS[k]}")
        if int(part.target.sum()) != EXPECTED_POSITIVES[k]: raise DataValidationError(f"{k} positive count mismatch")
    return split


def get_splits(path: Path | None = None) -> dict[str, pd.DataFrame]:
    return validate_and_split(load_data(path), load_split_ids())

