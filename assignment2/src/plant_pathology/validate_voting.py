"""Generate validation-set predictions and compare single models vs majority voting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from plant_pathology import CLASSES
from plant_pathology.data import LeafDataset, build_transforms, load_labeled_csv
from plant_pathology.metrics import classification_metrics
from plant_pathology.models import build_model, get_model_name
from plant_pathology.train import loader_options


def predict_validation(config_path: Path, checkpoint_path: Path, output_path: Path) -> Path:
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
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    image_ids: list[str] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                batch_probabilities = torch.softmax(model(images), dim=1).cpu().tolist()
            probabilities.extend(batch_probabilities)

    image_ids = dataset.frame["image_id"].tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probabilities, columns=list(CLASSES)).assign(image_id=image_ids).loc[
        :, ["image_id", *CLASSES]
    ].to_csv(output_path, index=False)
    return output_path


def evaluate_prediction(prediction_csv: Path, validation_csv: Path) -> dict[str, float]:
    truth = load_labeled_csv(validation_csv)
    pred = pd.read_csv(prediction_csv)
    merged = truth.merge(pred, on="image_id", how="inner")
    if len(merged) != len(truth):
        raise ValueError("prediction rows do not match validation rows")
    targets = (
        merged.loc[:, [f"{name}_x" for name in CLASSES]].to_numpy(dtype=float).argmax(axis=1)
    )
    predictions = (
        merged.loc[:, [f"{name}_y" for name in CLASSES]].to_numpy(dtype=float).argmax(axis=1)
    )
    return classification_metrics(targets.tolist(), predictions.tolist())


def vote_predictions(inputs: list[Path], output: Path) -> Path:
    frames = [pd.read_csv(path) for path in inputs]
    if not frames:
        raise ValueError("at least one validation prediction file is required")
    image_ids = frames[0]["image_id"].tolist()
    for frame in frames[1:]:
        if frame["image_id"].tolist() != image_ids:
            raise ValueError("all inputs must use the same image_id order")
    probabilities = np.stack(
        [frame.loc[:, list(CLASSES)].to_numpy(dtype=float) for frame in frames],
        axis=0,
    )
    votes = np.eye(len(CLASSES))[probabilities.argmax(axis=2)]
    vote_counts = votes.sum(axis=0)
    vote_probs = vote_counts / len(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(vote_probs, columns=list(CLASSES)).assign(image_id=image_ids).loc[
        :, ["image_id", *CLASSES]
    ].to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate major voting against single models")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run_id list, e.g. transfer transfer_seed_2",
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--validation-csv", type=Path, default=Path("data/validation.csv"))
    parser.add_argument(
        "--vote-output",
        type=Path,
        default=Path("predictions/validation_vote.csv"),
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("results/validation_voting_comparison.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    member_prediction_paths: list[Path] = []
    for run_id in args.runs:
        config_path = args.configs_dir / f"{run_id}.json"
        checkpoint_path = args.checkpoints_dir / get_model_name(run_id)
        prediction_path = args.predictions_dir / f"validation_{run_id}.csv"
        predict_validation(config_path, checkpoint_path, prediction_path)
        member_prediction_paths.append(prediction_path)
        metrics = evaluate_prediction(prediction_path, args.validation_csv)
        row: dict[str, object] = {
            "name": run_id,
            "kind": "single",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        for index, class_name in enumerate(CLASSES):
            row[f"recall_{class_name}"] = metrics[f"recall_class_{index}"]
        rows.append(row)

    vote_path = vote_predictions(member_prediction_paths, args.vote_output)
    vote_metrics = evaluate_prediction(vote_path, args.validation_csv)
    vote_row: dict[str, object] = {
        "name": vote_path.stem,
        "kind": "major_voting",
        "accuracy": vote_metrics["accuracy"],
        "macro_f1": vote_metrics["macro_f1"],
    }
    for index, class_name in enumerate(CLASSES):
        vote_row[f"recall_{class_name}"] = vote_metrics[f"recall_class_{index}"]
    rows.append(vote_row)

    report = pd.DataFrame(rows).sort_values(by="macro_f1", ascending=False)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_csv, index=False)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved comparison to {args.report_csv}")


if __name__ == "__main__":
    main()