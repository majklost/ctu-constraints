from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .types import ArteryClass

LabelMap = NDArray[np.uint8]

DEFAULT_CLASS_INTENSITIES: Mapping[ArteryClass, float] = MappingProxyType(
    {
        ArteryClass.BACKGROUND: 0.0,
        ArteryClass.BOUNDARY: 0.65,
        ArteryClass.LUMEN: 0.25,
        ArteryClass.PLAQUE: 1.0,
    }
)


def create_grayscale_image_from_label_mask(
    label_mask: LabelMap,
    class_intensities: Mapping[ArteryClass, float],
) -> NDArray[np.float32]:
    """Map every class in a label mask to a configured grayscale intensity."""
    if label_mask.ndim != 2:
        raise ValueError("label_mask must have shape [H, W]")

    image = np.empty(label_mask.shape, dtype=np.float32)
    for class_id in np.unique(label_mask):
        try:
            artery_class = ArteryClass(int(class_id))
        except ValueError as error:
            raise ValueError(f"unknown artery class ID: {class_id}") from error
        if artery_class not in class_intensities:
            raise ValueError(f"missing grayscale intensity for {artery_class.name}")
        intensity = float(class_intensities[artery_class])
        if not isfinite(intensity):
            raise ValueError(f"intensity for {artery_class.name} must be finite")
        image[label_mask == class_id] = intensity
    return image
