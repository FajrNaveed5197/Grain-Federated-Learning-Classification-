from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GrainDataset(Dataset):
    """
    Reads grain-image records from a CSV manifest with columns:
    path, source, label
    """

    def __init__(
        self,
        manifest_path: str | Path,
        class_mapping_path: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(class_mapping_path, "r", encoding="utf-8") as handle:
            self.class_to_idx = json.load(handle)

        self.df = pd.read_csv(self.manifest_path)

        required_columns = {"path", "source", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}"
            )

        unknown_labels = set(self.df["label"].unique()) - set(self.class_to_idx)
        if unknown_labels:
            raise ValueError(f"Unknown labels in manifest: {sorted(unknown_labels)}")

        self.df["target"] = self.df["label"].map(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing at row {index}: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["target"])
