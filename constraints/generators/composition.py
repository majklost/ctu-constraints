"""Pure ordered composition of independently stored anatomy layers."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import PlaqueLayer


@dataclass(frozen=True)
class ComposedLabelMaps:
    """Supervision and visual-material maps resolved from ordered layers."""

    target_labels: NDArray[np.uint8]
    appearance_labels: NDArray[np.uint8]


def compose_label_maps(
    empty_artery: np.ndarray,
    layers: Iterable[PlaqueLayer],
) -> ComposedLabelMaps:
    """Overlay target and appearance values in the exact input order."""
    artery = np.asarray(empty_artery)
    if artery.ndim != 2:
        raise ValueError("empty_artery must have shape [H, W]")
    if not np.isin(artery, [0, 1, 2]).all():
        raise ValueError("empty_artery must contain only IDs 0, 1, and 2")

    resolved_layers = tuple(layers)
    for layer_index, layer in enumerate(resolved_layers):
        if np.asarray(layer.mask).shape != artery.shape:
            raise ValueError(f"plaque layer {layer_index} does not match artery shape")

    target_labels = artery.astype(np.uint8, copy=True)
    appearance_labels = artery.astype(np.uint8, copy=True)
    for layer in resolved_layers:
        target_labels[layer.mask] = layer.target_class
        appearance_labels[layer.mask] = layer.resolved_appearance
    return ComposedLabelMaps(
        target_labels=target_labels,
        appearance_labels=appearance_labels,
    )
