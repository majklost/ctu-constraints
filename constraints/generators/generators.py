from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

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
        num_samples=1000,
        img_size=(256, 256),
        sample_specs: AffineSampleBound | None = None,
        template: torch.Tensor | None = None,
        speckle: float | None = None,
    ):
        """
        num_samples: number of samples to generate
        img_size: size of the generated images (height, width)
        sample_specs: specifications for the affine transformations (dx, dy, angle) if provided, otherwise defaults are used
        template: the canonical template for the generated images if provided, otherwise a default template (get_standard_mask()) is used
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.sample_specs = sample_specs or AffineSampleBound()
        self.template = template or get_standard_mask()  # Shape: (H, W, C)
        self.speckle = speckle

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a distinct random spatial offset for this training instance
        # Random rotation between -pi and +pi
        random_theta = np.random.uniform(
            self.sample_specs.angle_min, self.sample_specs.angle_max
        )
        random_dx = np.random.uniform(
            self.sample_specs.dx_min, self.sample_specs.dx_max
        )
        random_dy = np.random.uniform(
            self.sample_specs.dy_min, self.sample_specs.dy_max
        )
        # Build the exact affine target matrix
        M = torch.tensor(
            [
                [np.cos(random_theta), -np.sin(random_theta), random_dx],
                [np.sin(random_theta), np.cos(random_theta), random_dy],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        # Warp the canonical template to form our unique Target Ground Truth Mask
        grid = F.affine_grid(
            M, list(self.template.unsqueeze(0).size()), align_corners=False
        )
        target_mask = F.grid_sample(
            self.template.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Create a synthetic ultrasound scan out of the ground truth channels
        # Wall is grey, plaque is brighter/textured, background has speckle noise
        img = (target_mask[1] * 0.4) + (target_mask[2] * 0.7)

        # Apply multiplicative ultrasound speckle noise in-memory
        if self.speckle is not None:
            speckle = torch.randn_like(img) * 0.05
            img = torch.clamp(img + speckle, 0.0, 1.0).unsqueeze(0)  # Shape: [1, H, W]

        # Forward the raw tensors seamlessly into your UNet forward pipeline
        return (
            img,
            target_mask,
            self.template,
            M,
        )  # Return the synthetic image, target mask, canonical template, and affine matrix
