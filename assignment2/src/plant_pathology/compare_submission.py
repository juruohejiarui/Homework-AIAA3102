"""Compare a submission CSV against test_hack.csv ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plant_pathology import CLASSES
from plant_pathology.metrics import classification_metrics


def compare(submission_csv: Path, test_hack_csv: Path) -> None:
    truth = pd.read_csv(test_hack_csv)
    sub = pd.read_csv(submission_csv)

    merged = truth.merge(sub, on="image_id", how="inner")
    if len(merged) == 0:
        raise ValueError("no matching image_ids between submission and test_hack")

    targets = merged[[f"{c}_x" for c in CLASSES]].to_numpy(dtype=float).argmax(axis=1)
    predictions = merged[[f"{c}_y" for c in CLASSES]].to_numpy(dtype=float).argmax(axis=1)

    metrics = classification_metrics(targets.tolist(), predictions.tolist())
    print(f"submission: {submission_csv}")
    print(f"  samples:   {len(merged)}")
    print(f"  accuracy:  {metrics['accuracy']:.6f}")
    print(f"  macro_f1:  {metrics['macro_f1']:.6f}")
    for i, c in enumerate(CLASSES):
        print(f"  recall_{c}: {metrics[f'recall_class_{i}']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare submission CSV to test_hack ground truth")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--test-hack", type=Path, default=Path("data/test_hack.csv"))
    args = parser.parse_args()
    compare(args.submission, args.test_hack)


if __name__ == "__main__":
    main()
