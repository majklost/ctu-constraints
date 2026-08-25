"""Pure composition of named, independently stored anatomy layers."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .types import ArteryClass


@dataclass(frozen=True)
class PlaqueLayer:
    """One named binary layer and its resolved anatomical target class.

    A real plaque uses ``PLAQUE``. A fake plaque uses ``BOUNDARY`` or ``LUMEN``.
    The name remains available to image rendering, where different fake-plaque
    collections may have different appearances despite sharing a target class.
    """

    name: str
    mask: np.ndarray
    target_class: ArteryClass

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plaque layer name must not be empty")
        if self.target_class not in {
            ArteryClass.BOUNDARY,
            ArteryClass.LUMEN,
            ArteryClass.PLAQUE,
        }:
            raise ValueError("plaque target must be boundary, lumen, or plaque")


def compose_target_labels(
    empty_artery: np.ndarray,
    layers: Iterable[PlaqueLayer],
) -> np.ndarray:
    """Overlay layers on an empty artery and return standard class IDs 0--3.

    Fake layers are applied in input order. Real-plaque layers are applied last,
    so real anatomy wins wherever real and fake masks overlap.
    """
    artery = np.asarray(empty_artery)
    if artery.ndim != 2:
        raise ValueError("empty_artery must have shape [H, W]")
    if not np.isin(artery, [0, 1, 2]).all():
        raise ValueError("empty_artery must contain only IDs 0, 1, and 2")

    resolved_layers = tuple(layers)
    for layer in resolved_layers:
        if np.asarray(layer.mask).shape != artery.shape:
            raise ValueError(f"layer {layer.name!r} does not match artery shape")

    output = artery.astype(np.uint8, copy=True)
    for layer in resolved_layers:
        if layer.target_class != ArteryClass.PLAQUE:
            output[np.asarray(layer.mask, dtype=bool)] = layer.target_class
    for layer in resolved_layers:
        if layer.target_class == ArteryClass.PLAQUE:
            output[np.asarray(layer.mask, dtype=bool)] = ArteryClass.PLAQUE
    return output
