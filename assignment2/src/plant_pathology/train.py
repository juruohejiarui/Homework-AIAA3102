"""Task: implement model training and experiment recording.

Requirements:
1. Read the JSON config and set the random seed.
2. Build train and validation data loaders, model, loss, and optimizer.
3. Train for the configured number of epochs and evaluate on the fixed validation set after each
   epoch. The required baseline config must train manual ResNet18 from scratch for exactly 50
   configured epochs.
4. Save the checkpoint with the best validation macro F1.
5. Save the baseline training curves as `results/training_curves_baseline.png`.
6. Append the run settings and final validation results to `results/experiments.csv`.
7. Train both the baseline and transfer-learning configurations.
8. Run ablation experiments that change one factor at a time. These may include, but are not limited
   to, batch size and learning rate. Keep image_size=128 for required comparisons. Explore as many
   meaningful choices as the compute budget permits. Use no more than 50 epochs for each ablation;
   you may stop earlier when appropriate. Explain the experiments in `REPORT.md`.
9. For optional bonus work, save one comparison figure containing train and validation accuracy
   curves for both the original method and the algorithmically improved method.
10. You may use the model and configuration you consider best to generate the final prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(config_path: Path) -> None:
    """Train one configured experiment and append its result record.

    TODO: implement the requirements listed in this module's top-level docstring.
    """
    raise NotImplementedError(f"Training is not implemented; requested config: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
