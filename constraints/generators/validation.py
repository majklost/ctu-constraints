"""Geometry checks used while preparing transform presets."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

from constraints.voxelmorph.utils import spatial_transform

from .types import DeformationRejectionConfig


@dataclass(frozen=True)
class DeformationValidationResult:
    accepted: bool
    minimum_jacobian: float
    foreground_margin_px: int
    preserves_wall: bool


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


def validate_deformation(
    field: np.ndarray,
    source_labels: np.ndarray,
    config: DeformationRejectionConfig,
) -> DeformationValidationResult:
    """Check topology and clipping for one backward-sampling displacement.

    The Jacobian is checked on a conservative dilation of source foreground,
    including every input location that may affect the warped artery. The
    foreground margin is measured after nearest-neighbor label warping.
    """
    field = np.asarray(field, dtype=np.float32)
    source_labels = np.asarray(source_labels)
    if source_labels.ndim != 2 or field.shape != (2, *source_labels.shape):
        raise ValueError("expected source_labels [H,W] and field [2,H,W]")

    support = deformation_support(source_labels, field)
    minimum_jacobian = minimum_jacobian_determinant(field, support)
    labels_tensor = torch.from_numpy(source_labels.astype(np.float32, copy=False))[
        None, None
    ]
    field_tensor = torch.from_numpy(field)
    warped_labels = spatial_transform(
        labels_tensor,
        field_tensor,
        method="nearest",
    )[0, 0].numpy()
    foreground_margin_px = foreground_margin(warped_labels)
    preserves_wall = _preserves_wall_after_warp(
        source_labels,
        field_tensor,
        config.preserved_wall_thickness_px,
    )
    accepted = (
        minimum_jacobian > config.minimum_jacobian
        and foreground_margin_px >= config.minimum_foreground_margin_px
        and preserves_wall
    )
    return DeformationValidationResult(
        accepted=accepted,
        minimum_jacobian=minimum_jacobian,
        foreground_margin_px=foreground_margin_px,
        preserves_wall=preserves_wall,
    )


def _preserves_wall_after_warp(
    source_labels: np.ndarray,
    field_tensor: torch.Tensor,
    wall_thickness_px: int,
) -> bool:
    """Stress-test whether a thin outer wall survives nearest resampling."""
    if wall_thickness_px == 0:
        return True
    foreground = source_labels != 0
    if not foreground.any():
        return False
    distance_inside = distance_transform_edt(foreground)
    stress_labels = np.zeros(source_labels.shape, dtype=np.float32)
    stress_labels[foreground] = 2
    stress_labels[foreground & (distance_inside <= wall_thickness_px)] = 1
    warped = spatial_transform(
        torch.from_numpy(stress_labels)[None, None],
        field_tensor,
        method="nearest",
    )[0, 0].numpy()
    background_edge = binary_dilation(warped == 0, structure=np.ones((3, 3)))
    return not bool(np.any(background_edge & (warped == 2)))
