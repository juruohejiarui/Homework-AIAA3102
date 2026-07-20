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
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from plant_pathology import CLASSES
from plant_pathology.data import LeafDataset, build_transforms, load_labeled_csv
from plant_pathology.metrics import classification_metrics
from plant_pathology.models import build_model
from plant_pathology.train import loader_options


def error_group(true_label: str, predicted_label: str) -> str:
    return f"{true_label}_predicted_as_{predicted_label}"


def save_confusion_matrix(targets: list[int], predictions: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASSES)))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    figure, axis = plt.subplots(figsize=(7, 6))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_error_analysis(errors: list[dict[str, object]], output_path: Path) -> None:
    fields = [
        "image_id",
        "true_label",
        "predicted_label",
        "confidence",
        "failure_group",
        "observation",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(errors[:8])


def evaluate(config_path: Path, checkpoint: Path) -> None:
    """Evaluate a checkpoint on the fixed validation split and save required artifacts."""
    with config_path.open(encoding="utf-8") as file:
        config: dict[str, object] = json.load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeafDataset(
        load_labeled_csv(Path(str(config["validation_csv"]))),
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
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    errors: list[dict[str, object]] = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = torch.softmax(model(images), dim=1).cpu()
            batch_predictions = probabilities.argmax(dim=1)
            for index, (label, prediction, probability) in enumerate(
                zip(labels.tolist(), batch_predictions.tolist(), probabilities, strict=True)
            ):
                if label != prediction:
                    image_id = str(dataset.frame.iloc[offset + index]["image_id"])
                    errors.append(
                        {
                            "image_id": image_id,
                            "true_label": CLASSES[label],
                            "predicted_label": CLASSES[prediction],
                            "confidence": f"{probability[prediction].item():.6f}",
                            "failure_group": error_group(CLASSES[label], CLASSES[prediction]),
                            "observation": (
                                "Inspect leaf appearance and lesion distribution manually."
                            ),
                        }
                    )
            offset += labels.size(0)
            targets.extend(labels.tolist())
            predictions.extend(batch_predictions.tolist())
    metrics = classification_metrics(targets, predictions)

    matrix_name = f"confusion_matrix_{config['run_id']}.png"
    error_name = f"error_analysis_{config['run_id']}.csv"
        
    save_confusion_matrix(targets, predictions, Path("results") / matrix_name)
    errors.sort(key=lambda row: float(str(row["confidence"])), reverse=True)
    save_error_analysis(errors, Path("results") / error_name)
    
    # 按照作业要求，也保存一份默认的 error_analysis.csv
    save_error_analysis(errors, Path("results/error_analysis.csv"))
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    for index, class_name in enumerate(CLASSES):
        print(f"recall_{class_name}={metrics[f'recall_class_{index}']:.6f}")
    print(f"Saved confusion matrix to results/{matrix_name}")
    print(f"Recorded {min(8, len(errors))} validation errors in results/error_analysis.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
