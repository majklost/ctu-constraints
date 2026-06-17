from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import Dataset

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
            self.template = nnf.interpolate(
                self.template.unsqueeze(0),
                size=self.img_size,
                mode="nearest",
            ).squeeze(0)
        self.speckle = speckle
        self.fixed_seed = fixed_seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of range for the dataset.")
        # Generate a distinct random spatial offset for this training instance
        # Random rotation between -pi and +pi
        rng = np.random.default_rng([self.fixed_seed, idx])
        random_theta = rng.uniform(
            self.sample_specs.angle_min, self.sample_specs.angle_max
        )
        random_dx = rng.uniform(
            self.sample_specs.dx_min, self.sample_specs.dx_max
        )
        random_dy = rng.uniform(
            self.sample_specs.dy_min, self.sample_specs.dy_max
        )
        # Build the exact affine target matrix
        affine_matrix = torch.tensor(
            [
                [np.cos(random_theta), -np.sin(random_theta), random_dx],
                [np.sin(random_theta), np.cos(random_theta), random_dy],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        # Warp the canonical template to form our unique Target Ground Truth Mask
        grid = nnf.affine_grid(
            affine_matrix, list(self.template.unsqueeze(0).size()), align_corners=False
        )
        target_mask = nnf.grid_sample(
            self.template.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Create a synthetic ultrasound scan out of the ground truth channels
        # Wall is grey, plaque is brighter/textured, background has speckle noise
        img = (target_mask[0] * 0.2) + (target_mask[1] * 0.4) + (target_mask[2] * 0.7)

        # Apply deterministic ultrasound speckle noise in-memory.
        if self.speckle is not None:
            torch_seed = int(
                np.random.SeedSequence([self.fixed_seed, idx, 1]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
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
