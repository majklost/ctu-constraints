from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..datatools.datasets import ARTIFICIAL_MASK_NUM_CLASSES
from ..voxelmorph.utils import random_disp, spatial_transform
from .utils import get_standard_mask


@dataclass
class AffineSampleBound:
    dx_min: float = -0.5
    dx_max: float = 0.5
    dy_min: float = -0.5
    dy_max: float = 0.5
    angle_min: float = -np.pi
    angle_max: float = np.pi


ROT_ONLY = AffineSampleBound(
    dx_min=0, dx_max=0, dy_min=0, dy_max=0, angle_min=-np.pi, angle_max=np.pi
)

SMALL = AffineSampleBound(
    dx_min=-0.08, dx_max=0.08, dy_min=-0.08, dy_max=0.08, angle_min=-np.pi / 18, angle_max=np.pi / 18
)
NO_AFFINE = AffineSampleBound(
    dx_min=0, dx_max=0, dy_min=0, dy_max=0, angle_min=0, angle_max=0
)


def _sample_affine_matrix(
    sample_specs: AffineSampleBound,
    rng: np.random.Generator,
) -> torch.Tensor:
    random_theta = rng.uniform(sample_specs.angle_min, sample_specs.angle_max)
    random_dx = rng.uniform(sample_specs.dx_min, sample_specs.dx_max)
    random_dy = rng.uniform(sample_specs.dy_min, sample_specs.dy_max)
    return torch.tensor(
        [
            [np.cos(random_theta), -np.sin(random_theta), random_dx],
            [np.sin(random_theta), np.cos(random_theta), random_dy],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def _apply_affine(template: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
    batched_template = template.unsqueeze(0)
    grid = F.affine_grid(
        affine_matrix,
        list(batched_template.size()),
        align_corners=False,
    )
    return F.grid_sample(
        batched_template,
        grid,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] == ARTIFICIAL_MASK_NUM_CLASSES:
        mask = mask[1:]
    return (mask[0] * 0.2) + (mask[1] * 0.4) + (mask[2] * 0.7)


def _fill_missing_background(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[0] != ARTIFICIAL_MASK_NUM_CLASSES:
        return mask

    mask = mask.clamp(0.0, 1.0)
    missing_mass = (1.0 - mask.sum(dim=0, keepdim=True)).clamp_min(0.0)
    return torch.cat([mask[0:1] + missing_mass, mask[1:]], dim=0).clamp(0.0, 1.0)


class ArteryGeneratorAffine(Dataset):
    def __init__(
        self,
        fixed_seed: int,
        num_samples=1000,
        img_size=(256, 256),
        sample_specs: AffineSampleBound | None = None,
        template: torch.Tensor | None = None,
        speckle: float | None = None,
    ):
        """
        num_samples: number of samples to generate
        img_size: size of the generated images (height, width)
        sample_specs: affine transformation bounds (dx, dy, angle).
        template: canonical template, or the default standard mask if omitted.
        """
        if fixed_seed < 0:
            raise ValueError("fixed_seed must be non-negative.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        self.num_samples = num_samples
        self.img_size = tuple(img_size)
        self.sample_specs = sample_specs or AffineSampleBound()
        self.template = template if template is not None else get_standard_mask()
        if self.template.ndim != 3:
            raise ValueError("template must have shape [C, H, W].")
        if tuple(self.template.shape[-2:]) != self.img_size:
            self.template = F.interpolate(
                self.template.unsqueeze(0),
                size=self.img_size,
                mode="nearest",
            ).squeeze(0)
        self.speckle = speckle
        self.fixed_seed = fixed_seed

    def _sample_seed(self, idx: int, stream: int = 0) -> int:
        """Derive a deterministic, stream-specific seed for this sample."""
        return int(
            np.random.SeedSequence([11, self.fixed_seed, idx, stream]).generate_state(
                1, dtype=np.uint64
            )[0]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of range for the dataset.")
        rng = np.random.default_rng(self._sample_seed(idx, stream=0))
        affine_matrix = _sample_affine_matrix(self.sample_specs, rng)
        target_mask = _fill_missing_background(_apply_affine(self.template, affine_matrix))
        img = _mask_to_image(target_mask)

        # Apply deterministic ultrasound speckle noise in-memory.
        if self.speckle is not None:
            torch_seed = self._sample_seed(idx, stream=1)
            torch_rng = torch.Generator(device=img.device).manual_seed(torch_seed)
            speckle = torch.randn(
                img.shape,
                dtype=img.dtype,
                device=img.device,
                generator=torch_rng,
            )
            img = torch.clamp(img + speckle * self.speckle, 0.0, 1.0)
        img = img.unsqueeze(0)  # Shape: [1, H, W]

        # Forward the raw tensors seamlessly into your UNet forward pipeline
        return {
            "img": img,  # Shape: [1, H, W]
            "mask": target_mask,  # Shape: [C, H, W]
            "template": self.template,  # Shape: [C, H, W]
            "affine": affine_matrix.squeeze(0),  # Shape: [2, 3
        }


class ArteryGeneratorDeformed(Dataset):
    """Generate template deformed by fractal noise"""

    def __init__(
        self,
        fixed_seed: int,
        num_samples=1000,
        img_size=(256, 256),
        template: torch.Tensor | None = None,
        sample_specs: AffineSampleBound = NO_AFFINE,
        scales: float | int | list[float] = 10,
        magnitude: float = 3.0,
        integrations: int = 0,
        fractal_mode: Literal["blur", "upsample"] = "upsample",
        speckle: float | None = None,
    ):
        if fixed_seed < 0:
            raise ValueError("fixed_seed must be non-negative.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_samples = num_samples
        self.img_size = tuple(img_size)
        self.template = template if template is not None else get_standard_mask()
        if self.template.ndim != 3:
            raise ValueError("template must have shape [C, H, W].")
        if tuple(self.template.shape[-2:]) != self.img_size:
            self.template = F.interpolate(
                self.template.unsqueeze(0),
                size=self.img_size,
                mode="nearest",
            ).squeeze(0)
        self.template = self.template.contiguous()
        self._batched_template = self.template.unsqueeze(0)
        self.scales: float | int | list[float] = scales
        self.magnitude = magnitude
        self.integrations = integrations
        self.sample_specs: AffineSampleBound = sample_specs
        self.fixed_seed = fixed_seed
        self.speckle = speckle
        self.fractal_mode: Literal["blur", "upsample"] = fractal_mode

    def _sample_seed(self, idx: int, stream: int = 0) -> int:
        """Derive a deterministic, stream-specific seed for this sample."""
        return int(
            np.random.SeedSequence([23, self.fixed_seed, idx, stream]).generate_state(
                1, dtype=np.uint64
            )[0]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of range for the dataset.")

        rng = np.random.default_rng(self._sample_seed(idx, stream=2))
        affine_matrix = _sample_affine_matrix(self.sample_specs, rng)
        affine_template = _apply_affine(self.template, affine_matrix).contiguous()
        batched_template = affine_template.unsqueeze(0)  # Shape: [1, C, H, W]
        trf_seed = self._sample_seed(idx, stream=0)
        # Keep per-sample randomness deterministic without mutating global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(trf_seed)
            field = random_disp(
                batched_template.shape,
                scales=self.scales,
                magnitude=self.magnitude,
                integrations=self.integrations,
                fractal_mode=self.fractal_mode,
            )
        target_mask = _fill_missing_background(spatial_transform(batched_template, field, isdisp=True).squeeze(0))
        img = _mask_to_image(target_mask)

        # Apply deterministic ultrasound speckle noise in-memory.
        if self.speckle is not None:
            torch_seed = self._sample_seed(idx, stream=1)
            torch_rng = torch.Generator(device=img.device).manual_seed(torch_seed)
            speckle = torch.randn(
                img.shape,
                dtype=img.dtype,
                device=img.device,
                generator=torch_rng,
            )
            img = torch.clamp(img + speckle * self.speckle, 0.0, 1.0)
        img = img.unsqueeze(0)  # Shape: [1, H, W]
        # Forward the raw tensors seamlessly into your UNet forward pipeline
        return {
            "img": img,  # Shape: [1, H, W]
            "mask": target_mask,  # Shape: [C, H, W]
            "template": self.template,  # Shape: [C, H, W]
            "field": field.squeeze(0),  # Shape: [2, H, W]
            "affine": affine_matrix.squeeze(0),  # Shape: [2, 3]
        }
