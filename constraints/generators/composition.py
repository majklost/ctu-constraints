"""Pure ordered composition of independently stored anatomy layers."""

from collections.abc import Iterable

import numpy as np

from .types import PlaqueLayer


def compose_target_labels(
    empty_artery: np.ndarray,
    layers: Iterable[PlaqueLayer],
) -> np.ndarray:
    """Overlay layers in input order and return standard class IDs 0--3."""
    artery = np.asarray(empty_artery)
    if artery.ndim != 2:
        raise ValueError("empty_artery must have shape [H, W]")
    if not np.isin(artery, [0, 1, 2]).all():
        raise ValueError("empty_artery must contain only IDs 0, 1, and 2")

    resolved_layers = tuple(layers)
    for layer_index, layer in enumerate(resolved_layers):
        if np.asarray(layer.mask).shape != artery.shape:
            raise ValueError(f"plaque layer {layer_index} does not match artery shape")

    output = artery.astype(np.uint8, copy=True)
    for layer in resolved_layers:
        output[layer.mask] = layer.target_class
    return output
