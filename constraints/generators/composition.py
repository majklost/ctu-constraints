"""Pure ordered composition of independent label and grayscale patches."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layer_generators import TRANSPARENT_LABEL, LayerPatch
from .rendering import DEFAULT_CLASS_INTENSITIES, create_grayscale_image_from_label_mask


@dataclass(frozen=True)
class ComposedLayers:
    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]


def compose_layers(
    empty_artery: np.ndarray,
    layers: Iterable[LayerPatch],
) -> ComposedLayers:
    """Overlay label and grayscale patches independently in input order."""
    artery = np.asarray(empty_artery)
    if artery.ndim != 2:
        raise ValueError("empty_artery must have shape [H, W]")
    if not np.isin(artery, [0, 1, 2]).all():
        raise ValueError("empty_artery must contain only IDs 0, 1, and 2")

    patches = tuple(layers)
    for layer_index, patch in enumerate(patches):
        if patch.labels.shape != artery.shape:
            raise ValueError(f"layer {layer_index} does not match artery shape")

    target_labels = artery.astype(np.uint8, copy=True)
    image = create_grayscale_image_from_label_mask(
        target_labels, DEFAULT_CLASS_INTENSITIES
    )
    for patch in patches:
        label_pixels = patch.labels != TRANSPARENT_LABEL
        image_pixels = ~np.isnan(patch.image)
        target_labels[label_pixels] = patch.labels[label_pixels]
        image[image_pixels] = patch.image[image_pixels]
    return ComposedLayers(image=image, target_labels=target_labels)
