import pandas as pd
import pytest

from pipeline.artifacts import ArtifactValidationError, DATA_QUALITY_AUDIT_COLUMNS
from pipeline.data_quality import (
    canonicalize_text,
    duplicate_members,
    validate_data_quality_audit,
)


def test_canonicalization_normalizes_urls_entities_case_and_whitespace() -> None:
    left = "  FIRE &amp; Smoke HTTPS://Example.com/a  "
    right = "fire & smoke https://different.example/b"
    assert canonicalize_text(left) == canonicalize_text(right) == "fire & smoke <url>"


def test_duplicate_members_marks_conflicts_and_cross_split() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "split": ["train", "dev", "train"],
            "target": [0, 1, 0],
            "text": ["Same", "Same", "Other"],
        }
    )
    members = duplicate_members(frame, kind="exact")
    assert members["id"].tolist() == [1, 2]
    assert members["label_conflict"].all()
    assert members["cross_split"].all()


def test_data_quality_audit_contract() -> None:
    valid = pd.DataFrame(
        [[1, "hard_negative", "figurative use", "reject_false_positive", 0.9]],
        columns=DATA_QUALITY_AUDIT_COLUMNS,
    )
    validate_data_quality_audit(valid, valid_ids={1})
    invalid = valid.copy()
    invalid.loc[0, "disposition"] = "delete"
    with pytest.raises(ArtifactValidationError):
        validate_data_quality_audit(invalid, valid_ids={1})
