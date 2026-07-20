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
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from plant_pathology import CLASSES
from plant_pathology.data import build_transforms, load_test_csv
from plant_pathology.models import build_model
from plant_pathology.train import loader_options
from plant_pathology.validate_submission import validate_submission


class TestImageDataset(Dataset[tuple[torch.Tensor, str]]):
    """Load unlabeled images for the private test manifest."""

    def __init__(self, frame: pd.DataFrame, image_dir: Path, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = build_transforms(image_size, training=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = str(self.frame.iloc[index]["image_id"])
        with Image.open(self.image_dir / f"{image_id}.jpg") as image:
            return self.transform(image.convert("RGB")), image_id


def predict(config_path: Path, checkpoint: Path, output: Path) -> None:
    """Generate one class-probability row for every image in the test manifest."""
    with config_path.open(encoding="utf-8") as file:
        config: dict[str, object] = json.load(file)
    test_csv = Path(str(config["test_csv"]))
    dataset = TestImageDataset(
        load_test_csv(test_csv),
        Path(str(config["image_dir"])),
        int(config["image_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_options(config, device, int(config["seed"])),
    )
    model = build_model(
        str(config["model"]),
        pretrained=False,
        freeze_backbone=bool(config["freeze_backbone"]),
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    image_ids: list[str] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, identifiers in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                batch_probabilities = torch.softmax(model(images), dim=1).cpu().tolist()
            image_ids.extend(identifiers)
            probabilities.extend(batch_probabilities)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probabilities, columns=CLASSES).assign(image_id=image_ids).loc[
        :, ["image_id", *CLASSES]
    ].to_csv(output, index=False)
    validate_submission(test_csv, output)
    print(f"Saved and validated submission: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions/submission.csv"))
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
