from pathlib import Path

import h5py
import numpy as np
import polars as pl
import torch

from ..label_schema import LabelSchema
from .base_dataset import BaseDataset, PerSampleDataset
from .common_types import (
    ACDCMaskColor,
    ACDCMaskColorMyocard,
    ACDCMaskLabel,
    ACDCMaskLabelMyocard,
)
from .types import Sample, TemplateAssets


class ACDCSliceDataset(BaseDataset):
    def __init__(self, acdc_base_folder: Path | str, samples_df: pl.DataFrame) -> None:
        super().__init__()
        if "path" not in samples_df.columns:
            raise ValueError("DataFrame must contain a 'path' column.")

        self._root = Path(acdc_base_folder)

        # Ensure 'stem' column exists
        if "stem" not in samples_df.columns:
            samples_df = samples_df.with_columns(
                stem=pl.col("path").str.extract(r"[/\\]?([^/\\]+)\.[^.]+$", 1)
            )

        # Precompute full path strings and stems into flat lists to eliminate
        # DataFrame queries and Path concatenation overhead in __getitem__
        raw_paths = samples_df["path"].to_list()
        self._full_paths: list[str] = [str(self._root / p) for p in raw_paths]
        self._stems: list[str] = samples_df["stem"].to_list()
        self._length = len(self._full_paths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Sample:
        im, label = self.open_single_path(self._full_paths[index])

        # from_numpy is zero-copy; .float()/.long() ensures correct tensor dtypes
        return {
            "image": torch.from_numpy(im).float(),
            "target_labels": torch.from_numpy(label).long(),
            "sample_id": self._stems[index],
        }

    @staticmethod
    def open_single_path(full_path: str) -> tuple[np.ndarray, np.ndarray]:
        # Context manager prevents file descriptor leaks across DataLoader workers
        with h5py.File(full_path, "r") as file:
            # Slicing with [:] reads contiguous arrays without extra np.array() copies
            image = file["image"][:]
            label = file["label"][:]
        return image, label

    @property
    def label_schema(self) -> LabelSchema:
        return LabelSchema.from_lists(ACDCMaskLabel, ACDCMaskColor)

    @property
    def template_assets(self) -> TemplateAssets:
        raise NotImplementedError("Template Assets not ready")


class ACDCSliceMyocardiumOnlyDataset(ACDCSliceDataset):
    def __init__(self, acdc_base_folder: Path | str, samples_df: pl.DataFrame) -> None:
        super().__init__(acdc_base_folder, samples_df)

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        # Convert multiclass ACDC labels (1: RV, 2: MYO, 3: LV) to binary myocardium mask (MYO == 2)
        sample["target_labels"] = (sample["target_labels"] == 2).long()
        return sample

    @property
    def label_schema(self) -> LabelSchema:
        return LabelSchema.from_lists(ACDCMaskLabelMyocard, ACDCMaskColorMyocard)
