"""Task: complete the course image dataset.

Requirements:
1. Implement `LeafDataset.__len__` and `LeafDataset.__getitem__`.
2. Read the image ID and one-hot class columns from the provided dataframe.
3. Open `<image_dir>/<image_id>.jpg`, convert it to RGB, and apply the provided transform.
4. Return `(image_tensor, class_index)`, where the class index is an integer from 0 to 3.
5. Use the provided `build_transforms` function for the required baseline and transfer runs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from plant_pathology import CLASSES


class LeafDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load labeled course images from a dataframe."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of labeled examples."""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one RGB image and return its transformed tensor and class index."""
        row = self.frame.iloc[index]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        target = int(row.loc[list(CLASSES)].to_numpy(dtype=int).argmax())
        return tensor, target


def load_labeled_csv(path: Path) -> pd.DataFrame:
    """Load and validate the instructor-provided train or validation manifest."""
    frame = pd.read_csv(path)
    required = {"image_id", *CLASSES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not (frame.loc[:, CLASSES].sum(axis=1) == 1).all():
        raise ValueError("Every labeled row must have exactly one active class")
    return frame.reset_index(drop=True)


def load_test_csv(path: Path) -> pd.DataFrame:
    """Load the private-test manifest, which must contain image IDs but no labels."""
    frame = pd.read_csv(path)
    if list(frame.columns) != ["image_id"]:
        raise ValueError("test.csv must contain only the image_id column")
    return frame.reset_index(drop=True)


def build_transforms(image_size: int, training: bool) -> Callable[[Image.Image], torch.Tensor]:
    """Return the default preprocessing used by the required experiments."""
    if training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
