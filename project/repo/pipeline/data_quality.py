"""Deterministic duplicate discovery and data-quality artifact validation."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .artifacts import (
    DATA_QUALITY_AUDIT_COLUMNS,
    DATA_QUALITY_DISPOSITIONS,
    ArtifactValidationError,
)

URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
NEAR_DUPLICATE_THRESHOLD = 0.88
NEAR_NEIGHBORS = 8


def canonicalize_text(text: str) -> str:
    """Canonicalize superficial encoding/URL variation without removing words."""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    normalized = URL_PATTERN.sub("<url>", normalized)
    return " ".join(normalized.split())


def _group_id(kind: str, value: str) -> str:
    return f"{kind}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def duplicate_members(
    frame: pd.DataFrame,
    *,
    kind: str,
) -> pd.DataFrame:
    """Return stable member rows for raw-exact or canonical duplicate groups."""

    if kind not in {"exact", "canonical"}:
        raise ValueError("duplicate kind must be 'exact' or 'canonical'")
    required = {"id", "text", "target", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"duplicate frame missing columns: {sorted(missing)}")
    working = frame.loc[:, ["id", "split", "target", "text"]].copy()
    working["canonical_text"] = working["text"].map(canonicalize_text)
    key_column = "text" if kind == "exact" else "canonical_text"
    group_sizes = working.groupby(key_column, sort=False)["id"].transform("size")
    working = working.loc[group_sizes > 1].copy()
    rows: list[dict[str, object]] = []
    for key, group in working.groupby(key_column, sort=False):
        group = group.sort_values("id", kind="stable")
        labels = sorted(int(value) for value in group["target"].unique())
        splits = sorted(str(value) for value in group["split"].unique())
        exact_variants = int(group["text"].nunique())
        identifiers = ";".join(str(int(value)) for value in group["id"])
        for record in group.itertuples(index=False):
            rows.append(
                {
                    "group_type": kind,
                    "group_id": _group_id(kind, str(key)),
                    "group_size": len(group),
                    "label_conflict": len(labels) > 1,
                    "labels_present": ";".join(str(value) for value in labels),
                    "cross_split": len(splits) > 1,
                    "splits_present": ";".join(splits),
                    "exact_text_variants": exact_variants,
                    "member_ids": identifiers,
                    "id": int(record.id),
                    "split": str(record.split),
                    "target": int(record.target),
                    "text": str(record.text),
                    "canonical_text": str(record.canonical_text),
                }
            )
    columns = [
        "group_type",
        "group_id",
        "group_size",
        "label_conflict",
        "labels_present",
        "cross_split",
        "splits_present",
        "exact_text_variants",
        "member_ids",
        "id",
        "split",
        "target",
        "text",
        "canonical_text",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["label_conflict", "group_size", "group_id", "id"],
        ascending=[False, False, True, True],
        kind="stable",
        ignore_index=True,
    )


def near_duplicate_pairs(
    frame: pd.DataFrame,
    *,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    neighbors: int = NEAR_NEIGHBORS,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Return high-similarity non-canonical-equal pairs from a bounded k-NN audit."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("near-duplicate threshold must be between zero and one")
    if neighbors < 2:
        raise ValueError("neighbors must include self plus at least one neighbor")
    working = frame.loc[:, ["id", "split", "target", "text"]].reset_index(drop=True).copy()
    working["canonical_text"] = working["text"].map(canonicalize_text)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(working["canonical_text"])
    neighbor_count = min(neighbors, len(working))
    search = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    ).fit(matrix)
    distances, indices = search.kneighbors(matrix, return_distance=True)
    pairs: dict[tuple[int, int], dict[str, object]] = {}
    for left_index in range(len(working)):
        for distance, right_index in zip(distances[left_index], indices[left_index], strict=True):
            if left_index == int(right_index):
                continue
            similarity = float(1.0 - distance)
            if similarity + 1e-12 < threshold:
                continue
            left = working.iloc[left_index]
            right = working.iloc[int(right_index)]
            if left["canonical_text"] == right["canonical_text"]:
                continue
            left_id, right_id = sorted((int(left["id"]), int(right["id"])))
            key = (left_id, right_id)
            if key in pairs and float(pairs[key]["similarity"]) >= similarity:
                continue
            if int(left["id"]) != left_id:
                left, right = right, left
            pairs[key] = {
                "id_a": int(left["id"]),
                "split_a": str(left["split"]),
                "target_a": int(left["target"]),
                "id_b": int(right["id"]),
                "split_b": str(right["split"]),
                "target_b": int(right["target"]),
                "similarity": similarity,
                "label_conflict": int(left["target"]) != int(right["target"]),
                "cross_split": str(left["split"]) != str(right["split"]),
                "text_a": str(left["text"]),
                "text_b": str(right["text"]),
            }
    columns = [
        "id_a",
        "split_a",
        "target_a",
        "id_b",
        "split_b",
        "target_b",
        "similarity",
        "label_conflict",
        "cross_split",
        "text_a",
        "text_b",
    ]
    return pd.DataFrame(list(pairs.values()), columns=columns).sort_values(
        ["label_conflict", "similarity", "id_a", "id_b"],
        ascending=[False, False, True, True],
        kind="stable",
        ignore_index=True,
    )


def duplicate_summary(
    exact: pd.DataFrame,
    canonical: pd.DataFrame,
    near: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize group/member/conflict/cross-split counts without double claims."""

    rows: list[dict[str, object]] = []
    for name, members in (("exact", exact), ("canonical", canonical)):
        groups = members.drop_duplicates("group_id")
        rows.append(
            {
                "relationship_type": name,
                "groups_or_pairs": len(groups),
                "member_rows": len(members),
                "conflicting_groups_or_pairs": int(groups["label_conflict"].sum()),
                "cross_split_groups_or_pairs": int(groups["cross_split"].sum()),
            }
        )
    rows.append(
        {
            "relationship_type": "near",
            "groups_or_pairs": len(near),
            "member_rows": int(near[["id_a", "id_b"]].stack().nunique()) if not near.empty else 0,
            "conflicting_groups_or_pairs": int(near["label_conflict"].sum()) if not near.empty else 0,
            "cross_split_groups_or_pairs": int(near["cross_split"].sum()) if not near.empty else 0,
        }
    )
    return pd.DataFrame(rows)


def validate_data_quality_audit(
    frame: pd.DataFrame,
    *,
    valid_ids: set[int] | None = None,
) -> None:
    """Validate the exact audit schema, stable IDs, dispositions, and confidence scale."""

    if list(frame.columns) != DATA_QUALITY_AUDIT_COLUMNS:
        raise ArtifactValidationError(
            f"data-quality columns must be exactly {DATA_QUALITY_AUDIT_COLUMNS}"
        )
    if frame.empty:
        raise ArtifactValidationError("data-quality audit must not be empty")
    if frame.isna().any().any():
        raise ArtifactValidationError("data-quality audit must not contain missing values")
    if not frame["disposition"].isin(DATA_QUALITY_DISPOSITIONS).all():
        observed = sorted(set(frame["disposition"]) - DATA_QUALITY_DISPOSITIONS)
        raise ArtifactValidationError(f"invalid data-quality dispositions: {observed}")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0, inclusive="both").all():
        raise ArtifactValidationError("confidence must be numeric and in [0, 1]")
    if (frame["issue_type"].astype(str).str.strip() == "").any() or (frame["evidence"].astype(str).str.strip() == "").any():
        raise ArtifactValidationError("issue_type and evidence must be nonempty")
    if valid_ids is not None:
        unexpected = sorted(set(frame["id"].astype(int)) - valid_ids)
        if unexpected:
            raise ArtifactValidationError(f"audit contains unknown IDs: {unexpected[:5]}")


def attach_split(frame: pd.DataFrame, split_by_id: Mapping[int, str]) -> pd.DataFrame:
    """Attach explicit split membership by stable ID."""

    result = frame.copy()
    result["split"] = result["id"].map(split_by_id)
    if result["split"].isna().any():
        raise ValueError("split mapping does not cover every row ID")
    return result
