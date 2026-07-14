"""Task: generate predictions for the course test set.

Requirements:
1. Read the config, rebuild the model and test transforms, and load the checkpoint.
2. Predict one four-class probability row for every image ID in `data/test.csv`.
3. Save `predictions/submission.csv` with columns in this exact order:
   image_id, healthy, multiple_diseases, rust, scab.
4. Preserve every test image ID exactly once and ensure each probability row sums to 1.
5. Run `python -m plant_pathology.validate_submission` after creating the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def predict(config_path: Path, checkpoint: Path, output: Path) -> None:
    """TODO: implement the requirements listed in the module docstring."""
    raise NotImplementedError(
        f"Prediction is not implemented; config: {config_path}, checkpoint: {checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions/submission.csv"))
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
