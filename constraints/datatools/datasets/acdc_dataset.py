from pathlib import Path

import h5py
import numpy as np
import polars as pl
import torch
import torch.nn.functional

from ..label_schema import LabelSchema
from .base_dataset import PerSampleDataset
from .common_types import (
    ACDCMaskColor,
    ACDCMaskColorMyocard,
    ACDCMaskLabel,
    ACDCMaskLabelMyocard,
)
from .types import Sample


class ACDCSliceDataset(PerSampleDataset):
    def __init__(
        self,
        acdc_base_folder: Path | str,
        samples_df: pl.DataFrame,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if "path" not in samples_df.columns:
            raise ValueError("DataFrame must contain a 'path' column.")
        if image_size is not None and any(size <= 0 for size in image_size):
            raise ValueError("image_size dimensions must be positive.")

        self._root = Path(acdc_base_folder)
        self._image_size = image_size

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

        image = torch.from_numpy(im).float().unsqueeze(0)
        target_labels = torch.from_numpy(label).long()
        if self._image_size is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self._image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            target_labels = (
                torch.nn.functional.interpolate(
                    target_labels[None, None].float(),
                    size=self._image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        return {
            "image": image,
            "target_labels": target_labels,
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


class ACDCSliceMyocardiumOnlyDataset(ACDCSliceDataset):
    def __init__(
        self,
        acdc_base_folder: Path | str,
        samples_df: pl.DataFrame,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(acdc_base_folder, samples_df, image_size=image_size)

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        # Convert multiclass labels to a binary myocardium mask (MYO == 2).
        sample["target_labels"] = (sample["target_labels"] == 2).long()
        return sample

    @property
    def label_schema(self) -> LabelSchema:
        return LabelSchema.from_lists(ACDCMaskLabelMyocard, ACDCMaskColorMyocard)
