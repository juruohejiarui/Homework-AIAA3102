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
import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from plant_pathology.data import LeafDataset, build_transforms, load_labeled_csv
from plant_pathology.metrics import classification_metrics
from plant_pathology.models import build_model

EXPERIMENT_FIELDS = [
    "run_id",
    "model",
    "main_change",
    "image_size",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "seed",
    "val_accuracy",
    "val_macro_f1",
    "notes",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def loader_options(config: dict[str, object], device: torch.device) -> dict[str, object]:
    num_workers = int(config.get("num_workers", min(8, os.cpu_count() or 1)))
    options: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = int(config.get("prefetch_factor", 4))
    return options


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
    return classification_metrics(targets, predictions)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = range(1, len(history) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="train loss")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Baseline training loss")
    axes[0].legend()
    axes[1].plot(epochs, [entry["train_accuracy"] for entry in history], label="train accuracy")
    axes[1].plot(epochs, [entry["val_accuracy"] for entry in history], label="validation accuracy")
    axes[1].plot(epochs, [entry["val_macro_f1"] for entry in history], label="validation macro F1")
    axes[1].set(xlabel="epoch", ylabel="score", title="Baseline accuracy and macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def record_experiment(config: dict[str, object], metrics: dict[str, float]) -> None:
    results_path = Path(str(config["results_csv"]))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))
    row = {
        "run_id": str(config["run_id"]),
        "model": str(config["model"]),
        "main_change": "initial model" if config["run_id"] == "baseline" else "configured run",
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "seed": config["seed"],
        "val_accuracy": f"{metrics['accuracy']:.6f}",
        "val_macro_f1": f"{metrics['macro_f1']:.6f}",
        "notes": "best checkpoint selected by validation macro F1",
    }
    existing_rows = [entry for entry in existing_rows if entry.get("run_id") != row["run_id"]]
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def build_optimizer(model: nn.Module, config: dict[str, object]) -> Optimizer:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

def compute_class_weights(dataset: LeafDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse class frequencies for weighted loss."""
    class_counts = dataset.frame.loc[:, list(dataset.frame.columns)[1:]].sum().to_numpy()
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(config_path: Path) -> None:
    """Train one experiment, save its best checkpoint, and record validation metrics."""
    with config_path.open(encoding="utf-8") as file:
        config: dict[str, object] = json.load(file)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    image_dir = Path(str(config["image_dir"]))
    train_dataset = LeafDataset(
        load_labeled_csv(Path(str(config["train_csv"]))),
        image_dir,
        build_transforms(int(config["image_size"]), training=True),
    )
    validation_dataset = LeafDataset(
        load_labeled_csv(Path(str(config["validation_csv"]))),
        image_dir,
        build_transforms(int(config["image_size"]), training=False),
    )
    options = loader_options(config, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **options,
    )
    model = build_model(
        str(config["model"]),
        pretrained=bool(config["pretrained"]),
        freeze_backbone=bool(config["freeze_backbone"]),
    ).to(device)
    # 使用类别权重解决 multiple_diseases 样本过少的问题
    if config["label_weights"] :
        class_weights = compute_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"weights: {class_weights}")
    else :
        criterion = nn.CrossEntropyLoss()
        print("No label weights")
    optimizer = build_optimizer(model, config)
    
    # 添加学习率调度器
    if config["scheduler"] :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        print("Use scheduler")
    else :
        scheduler = None
        print("No scheduler")
    
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    checkpoint_path = Path(str(config["checkpoint_dir"])) / f"{config['run_id']}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in trange(1, int(config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        metrics = evaluate_loader(model, validation_loader, device)
        history.append(
            {
                "train_loss": running_loss / total,
                "train_accuracy": correct / total,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        tqdm.write(
            f"epoch {epoch}/{config['epochs']}: loss={history[-1]['train_loss']:.4f}, "
            f"train_acc={history[-1]['train_accuracy']:.4f}, val_f1={metrics['macro_f1']:.4f}"
        )
        
        # 更新学习率调度器
        if scheduler is not None : 
            scheduler.step(metrics["macro_f1"])
        
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            torch.save(
                {"model_state": model.state_dict(), "config": config, "epoch": epoch},
                checkpoint_path,
            )

    if best_metrics is None:
        raise RuntimeError("No training epochs were completed")
    if config["run_id"] == "baseline":
        save_training_curves(history, Path("results/training_curves_baseline.png"))
    record_experiment(config, best_metrics)
    print(f"Saved best checkpoint to {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
