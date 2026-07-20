"""Majority voting ensemble: combine multiple submission CSVs into one final prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plant_pathology import CLASSES
from plant_pathology.validate_submission import validate_submission


def major_voting(inputs: list[Path], output: Path, test_csv: Path) -> None:
    """Hard majority vote across N submission CSVs, save vote proportions as output."""
    submissions = [pd.read_csv(p) for p in inputs]
    if not submissions:
        raise ValueError("at least one input submission is required")

    image_ids = submissions[0]["image_id"].tolist()
    for s in submissions[1:]:
        if s["image_id"].tolist() != image_ids:
            raise ValueError("all submissions must share the same image_id order")

    n_models = len(submissions)
    prob_arrays = np.stack([s[list(CLASSES)].to_numpy(dtype=float) for s in submissions], axis=0)

    # hard votes: each model votes for its argmax class (probabilities ignored beyond argmax)
    votes = np.eye(len(CLASSES))[prob_arrays.argmax(axis=2)]  # (n_models, n_samples, n_classes)
    vote_counts = votes.sum(axis=0)  # (n_samples, n_classes)
    winners = vote_counts.argmax(axis=1)  # ties broken by lower class index

    # output probabilities = vote proportions (sum to 1)
    result_probs = vote_counts / n_models

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_probs, columns=list(CLASSES)).assign(image_id=image_ids).loc[
        :, ["image_id", *CLASSES]
    ].to_csv(output, index=False)

    validate_submission(test_csv, output)
    print(f"Ensemble of {n_models} models saved to {output}")

    # report class distribution
    class_ids, counts = np.unique(winners, return_counts=True)
    for idx, count in zip(class_ids, counts, strict=True):
        print(f"  {CLASSES[idx]}: {count}")
    ties = (vote_counts == vote_counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
    print(f"  ties (broken by class order): {ties.sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Majority voting ensemble")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Input submission CSV files")
    parser.add_argument(
        "--output", type=Path, default=Path("predictions/submission_major_voting.csv")
    )
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()
    major_voting(args.inputs, args.output, args.test_csv)


if __name__ == "__main__":
    main()
