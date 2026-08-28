from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .types import AppearanceKind

LabelMap = NDArray[np.uint8]

DEFAULT_CLASS_INTENSITIES: Mapping[AppearanceKind, float] = MappingProxyType(
    {
        AppearanceKind.BACKGROUND: 0.0,
        AppearanceKind.BOUNDARY: 0.45,
        AppearanceKind.LUMEN: 0.25,
        AppearanceKind.PLAQUE: 1.0,
    }
)


def create_grayscale_image_from_label_mask(
    label_mask: LabelMap,
    class_intensities: Mapping[AppearanceKind, float],
) -> NDArray[np.float32]:
    """Map every class in a label mask to a configured grayscale intensity."""
    if label_mask.ndim != 2:
        raise ValueError("label_mask must have shape [H, W]")

    image = np.empty(label_mask.shape, dtype=np.float32)
    for class_id in np.unique(label_mask):
        try:
            appearance = AppearanceKind(int(class_id))
        except ValueError as error:
            raise ValueError(f"unknown appearance ID: {class_id}") from error
        if appearance not in class_intensities:
            raise ValueError(f"missing grayscale intensity for {appearance.name}")
        intensity = float(class_intensities[appearance])
        if not isfinite(intensity):
            raise ValueError(f"intensity for {appearance.name} must be finite")
        image[label_mask == class_id] = intensity
    return image
