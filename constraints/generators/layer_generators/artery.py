"""Base artery rasterization shared by source creation and previews."""

import numpy as np

from ..types import ArteryClass, EmptyArteryConfig
from .rasterizer import polar_grid


def create_empty_artery(config: EmptyArteryConfig) -> np.ndarray:
    radius, _ = polar_grid(config.image_size)
    labels = np.full(config.image_size, ArteryClass.BACKGROUND, dtype=np.uint8)
    labels[radius <= config.lumen_radius_px + config.wall_thickness_px] = (
        ArteryClass.BOUNDARY
    )
    labels[radius <= config.lumen_radius_px] = ArteryClass.LUMEN
    return labels
