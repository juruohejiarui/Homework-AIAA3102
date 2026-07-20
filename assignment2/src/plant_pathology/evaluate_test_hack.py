"""Evaluate a checkpoint against data/test_hack.csv (ground-truth test labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from plant_pathology import CLASSES
from plant_pathology.data import LeafDataset, build_transforms, load_labeled_csv
from plant_pathology.metrics import classification_metrics
from plant_pathology.models import build_model
from plant_pathology.train import loader_options


def evaluate_test_hack(config_path: Path, checkpoint: Path) -> None:
    with config_path.open(encoding="utf-8") as file:
        config: dict[str, object] = json.load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeafDataset(
        load_labeled_csv(Path("data/test_hack.csv")),
        Path(str(config["image_dir"])),
        build_transforms(int(config["image_size"]), training=False),
    )
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
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    metrics = classification_metrics(targets, predictions)
    run_id = config["run_id"]
    print(
        f"[test_hack] {run_id}: accuracy={metrics['accuracy']:.6f}  "
        f"macro_f1={metrics['macro_f1']:.6f}"
    )
    for index, class_name in enumerate(CLASSES):
        print(f"  recall_{class_name}={metrics[f'recall_class_{index}']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    evaluate_test_hack(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
