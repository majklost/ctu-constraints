"""Cheap geometry checks used while preparing transform presets."""

import numpy as np
from scipy.ndimage import binary_dilation


def foreground_margin(labels: np.ndarray) -> int:
    """Smallest pixel distance of non-background content from an image edge."""
    foreground = np.asarray(labels) != 0
    if foreground.ndim != 2 or not foreground.any():
        return -1
    y, x = np.nonzero(foreground)
    height, width = foreground.shape
    return int(
        np.minimum.reduce([y.min(), x.min(), height - 1 - y.max(), width - 1 - x.max()])
    )


def validate_foreground_margin(
    labels: np.ndarray, minimum_margin: int
) -> tuple[bool, int]:
    if minimum_margin < 0:
        raise ValueError("minimum_margin must be non-negative")
    margin = foreground_margin(labels)
    return margin >= minimum_margin, margin


def minimum_jacobian_determinant(
    field: np.ndarray, support: np.ndarray | None = None
) -> float:
    """Return the minimum determinant of `x -> x + field(x)` in `(dy, dx)` order."""
    displacement = np.asarray(field, dtype=np.float32)
    if displacement.ndim != 3 or displacement.shape[0] != 2:
        raise ValueError("field must have shape [2,H,W] in (dy, dx) order")
    dy_dy, dy_dx = np.gradient(displacement[0])
    dx_dy, dx_dx = np.gradient(displacement[1])
    determinant = (1 + dy_dy) * (1 + dx_dx) - dy_dx * dx_dy
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != determinant.shape or not support.any():
            raise ValueError("support must be a non-empty Boolean image matching field")
        determinant = determinant[support]
    return float(determinant.min())


def deformation_support(labels: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Input locations that can affect foreground after backward sampling.

    Voxelmorph fields are zero-padded at image edges while being integrated.
    Consequently, a global Jacobian test can report a fold only in background
    edge pixels which never influence the artery.  Dilating the source
    foreground by the maximum displacement checks the region that can affect
    the output anatomy instead.
    """
    labels = np.asarray(labels)
    displacement = np.asarray(field)
    if labels.ndim != 2 or displacement.shape != (2, *labels.shape):
        raise ValueError("expected labels [H,W] and field [2,H,W]")
    radius = int(np.ceil(np.abs(displacement).max())) + 1
    return binary_dilation(labels != 0, iterations=radius)
