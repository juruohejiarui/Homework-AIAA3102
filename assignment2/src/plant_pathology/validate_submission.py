"""Validate a private-test prediction file before submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plant_pathology import CLASSES


def validate_submission(test_csv: Path, submission_csv: Path) -> None:
    test = pd.read_csv(test_csv)
    submission = pd.read_csv(submission_csv)
    expected_columns = ["image_id", *CLASSES]
    if list(test.columns) != ["image_id"]:
        raise ValueError("test.csv must contain only image_id")
    if list(submission.columns) != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}")
    if submission["image_id"].duplicated().any():
        raise ValueError("submission contains duplicate image IDs")
    if set(submission["image_id"]) != set(test["image_id"]):
        raise ValueError("submission image IDs do not exactly match test.csv")
    probabilities = submission.loc[:, CLASSES].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("submission probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("submission probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each row of probabilities must sum to 1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--submission", type=Path, default=Path("predictions/submission.csv")
    )
    args = parser.parse_args()
    validate_submission(args.test_csv, args.submission)
    print("Submission format is valid.")


if __name__ == "__main__":
    main()
