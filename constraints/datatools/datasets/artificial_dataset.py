import csv
from pathlib import Path
from typing import get_args

import numpy as np
import torch

from constraints.datatools.datasets.types import TemplateAssets

from ...utils import signed_distance_kornia, signed_distance_scipy
from ..label_schema import LabelSchema
from .base_dataset import PerSampleDataset
from .types import Sample, SDFMode

BAD_INDICES_FILENAME = "bad_indices.csv"
_ArtificialMaskLabel = ["background", "boundary", "lumen", "plaque"]
_ArtificialMaskColor = [
    (0.0, 0.0, 0.0),  # background
    (0.90, 0.10, 0.10),  # red
    (0.10, 0.70, 0.10),  # green
    (0.10, 0.35, 0.95),
]


def _load_valid_indices(
    folder: Path, num_samples: int, bad_indices_fname: str
) -> np.ndarray:
    bad_indices_path = folder / bad_indices_fname
    if not bad_indices_path.exists():
        return np.arange(num_samples)

    with bad_indices_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "index" not in reader.fieldnames:
            raise ValueError(f"{bad_indices_path} must contain an 'index' column.")
        bad_indices = {
            int(row["index"])
            for row in reader
            if row.get("index") is not None and row["index"].strip()
        }

    invalid_indices = sorted(
        index for index in bad_indices if index < 0 or index >= num_samples
    )
    if invalid_indices:
        raise ValueError(
            f"{bad_indices_path} contains indices outside [0, {num_samples}): "
            f"{invalid_indices}"
        )

    valid_mask = np.ones(num_samples, dtype=bool)
    valid_mask[list(bad_indices)] = False
    return np.flatnonzero(valid_mask)


class CachedArtificialDataset(PerSampleDataset):
    """Base class for artificial datasets.

    This class provides a common interface for artificial datasets, which are
    typically used for testing and validation purposes. Subclasses should
    implement the `__len__` and `__getitem__` methods to provide access to
    the dataset samples.
    """

    def __init__(
        self,
        folder: Path,
        sdf_mode: SDFMode = "scipy",
        return_transform: bool = False,
        return_template_sdf: bool = False,
        bad_indices_fname: str | None = BAD_INDICES_FILENAME,
    ):
        self._images = np.load(f"{folder}/img.npy", mmap_mode="r")
        self._masks = np.load(f"{folder}/mask.npy", mmap_mode="r")
        self._sdf_kornia = np.load(f"{folder}/sdf_kornia.npy", mmap_mode="r")
        self._sdf_scipy = np.load(f"{folder}/sdf_scipy.npy", mmap_mode="r")
        self._template = np.load(f"{folder}/template.npy")
        self._transform = np.load(f"{folder}/transform.npy", mmap_mode="r")
        self._valid_indices = (
            _load_valid_indices(folder, len(self._images), bad_indices_fname)
            if bad_indices_fname
            else np.arange(len(self._images))
        )
        if sdf_mode not in get_args(SDFMode):
            raise ValueError(f"Unknown sdf_mode: {sdf_mode}")
        self._sdf_mode = sdf_mode
        self._label_schema = self._create_label_schema()
        self._return_transform = return_transform
        self._return_template_sdf = return_template_sdf
        if return_template_sdf:
            template_labels = self._mask_to_label_map(torch.from_numpy(self._template))
            template_foreground = self.label_schema.label_map_to_foreground_one_hot(
                template_labels
            ).float()
            if sdf_mode == "kornia":
                self._template_sdf = signed_distance_kornia(template_foreground)
            elif sdf_mode == "scipy":
                self._template_sdf = signed_distance_scipy(template_foreground)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int) -> Sample:
        idx = int(self._valid_indices[index])
        if self._sdf_mode == "kornia":
            sdf = torch.from_numpy(np.array(self._sdf_kornia[idx]))
        elif self._sdf_mode == "scipy":
            sdf = torch.from_numpy(np.array(self._sdf_scipy[idx]))
        else:
            raise ValueError(f"Unknown sdf_mode: {self._sdf_mode}")
        mask = torch.from_numpy(np.array(self._masks[idx]))

        template = torch.from_numpy(self._template)
        sample = Sample(
            image=torch.from_numpy(np.array(self._images[idx])),
            target_labels=self._mask_to_label_map(mask),
            sample_id=str(idx) + "_real_" + str(index) + "_filtered",
            sdf=sdf,
            template=self._mask_to_label_map(template),
        )
        if self._return_transform:
            sample["transform"] = torch.from_numpy(np.array(self._transform[idx]))
        if self._return_template_sdf:
            sample["template_sdf"] = self._template_sdf.clone()
        return sample

    def _create_label_schema(self) -> LabelSchema:
        return LabelSchema.from_lists(
            names=_ArtificialMaskLabel, colors=_ArtificialMaskColor
        )

    def _mask_to_label_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert either full-class or foreground-only channel masks to IDs."""
        if mask.ndim != 3:
            raise ValueError(f"Expected mask shape [C, H, W], got {tuple(mask.shape)}")
        if mask.shape[0] == self.label_schema.num_classes:
            return self.label_schema.one_hot_to_label_map(mask)
        if mask.shape[0] == len(self.label_schema.foreground_ids):
            return self.label_schema.foreground_one_hot_to_label_map(mask)
        raise ValueError(
            "Expected a full-class or foreground-only mask with "
            f"{self.label_schema.num_classes} or "
            f"{len(self.label_schema.foreground_ids)} channels, got {mask.shape[0]}"
        )

    @property
    def label_schema(self) -> LabelSchema:
        return self._label_schema
