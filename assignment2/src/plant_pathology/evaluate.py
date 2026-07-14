"""Task: evaluate the baseline and final models on the validation set.

Requirements:
1. Read the config, rebuild the model and transforms, and load the checkpoint.
2. Report accuracy, macro F1, and recall for each class.
3. Save labeled confusion matrices as `results/confusion_matrix_baseline.png` and
   `results/confusion_matrix_final.png`.
4. Inspect at least eight incorrect predictions and record them in
   `results/error_analysis.csv`.
5. Group the selected errors into at least two common error types for discussion in `REPORT.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(config_path: Path, checkpoint: Path) -> None:
    """TODO: implement the requirements listed in the module docstring."""
    raise NotImplementedError(
        f"Evaluation is not implemented; config: {config_path}, checkpoint: {checkpoint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
