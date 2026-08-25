from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

from ..types import ArteryClass, ArterySpec, FloatArray, PlaqueSpec

LabelMap = NDArray[np.uint8]


@dataclass(frozen=True)
class PowerPlaqueParameters:
    """Serializable parameters for an ellipse-like polar plaque.

    Depths are resolved pixel measurements. ``shape_power=0.5`` produces the
    familiar square-root profile of an ellipse in local angular/radial
    coordinates. Larger values concentrate the plaque around its central angle.
    """

    angle_rad: float
    angular_width_rad: float
    inward_depth_px: float
    wall_depth_px: float
    shape_power: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.angle_rad,
            self.angular_width_rad,
            self.inward_depth_px,
            self.wall_depth_px,
            self.shape_power,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("all power-plaque parameters must be finite")
        if not 0 < self.angular_width_rad <= 2 * pi:
            raise ValueError("angular_width_rad must be in (0, 2*pi]")
        if self.inward_depth_px <= 0:
            raise ValueError("inward_depth_px must be positive")
        if self.wall_depth_px < 0:
            raise ValueError("wall_depth_px must be non-negative")
        if self.shape_power <= 0:
            raise ValueError("shape_power must be positive")


@dataclass(frozen=True)
class _PowerRadialBoundary:
    base_radius_px: float
    signed_depth_px: float
    angular_width_rad: float
    shape_power: float

    def __call__(self, angular_offset_rad: FloatArray) -> FloatArray:
        normalized = 2 * angular_offset_rad / self.angular_width_rad
        profile = np.clip(1 - normalized**2, 0.0, None) ** self.shape_power
        return self.base_radius_px + self.signed_depth_px * profile


def create_power_plaque(
    parameters: PowerPlaqueParameters, lumen_radius_px: float
) -> PlaqueSpec:
    """Turn serializable parameters into a callable runtime plaque spec."""
    if not isfinite(lumen_radius_px) or lumen_radius_px <= 0:
        raise ValueError("lumen_radius_px must be finite and positive")

    return PlaqueSpec(
        angle_rad=parameters.angle_rad,
        angular_width_rad=parameters.angular_width_rad,
        inner_radius=_PowerRadialBoundary(
            base_radius_px=lumen_radius_px,
            signed_depth_px=-parameters.inward_depth_px,
            angular_width_rad=parameters.angular_width_rad,
            shape_power=parameters.shape_power,
        ),
        outer_radius=_PowerRadialBoundary(
            base_radius_px=lumen_radius_px,
            signed_depth_px=parameters.wall_depth_px,
            angular_width_rad=parameters.angular_width_rad,
            shape_power=parameters.shape_power,
        ),
    )


def render_artery(spec: ArterySpec) -> LabelMap:
    """Rasterize an artery specification into a ``[H, W]`` class-ID map.

    NumPy evaluates the radial interval at every pixel directly. This avoids the
    angular sampling and polygon-edge artifacts that ``cv2.fillPoly`` would add
    for shapes already defined as radial functions.
    """
    height, width = spec.image_size
    center_y, center_x = spec.center
    y, x = np.ogrid[:height, :width]
    dy = y - center_y
    dx = x - center_x
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    labels = np.full(spec.image_size, ArteryClass.BACKGROUND, dtype=np.uint8)
    labels[radius <= spec.outer_radius_px] = ArteryClass.BOUNDARY
    labels[radius <= spec.lumen_radius_px] = ArteryClass.LUMEN

    plaque_mask = np.zeros(spec.image_size, dtype=bool)
    for plaque in spec.plaques:
        plaque_mask |= _render_plaque(
            radius,
            angle,
            plaque,
            artery_outer_radius_px=spec.outer_radius_px,
            has_wall=spec.wall_thickness_px > 0,
        )
    labels[plaque_mask] = ArteryClass.PLAQUE
    return labels


def _render_plaque(
    radius: FloatArray,
    angle: FloatArray,
    plaque: PlaqueSpec,
    artery_outer_radius_px: float,
    has_wall: bool,
) -> NDArray[np.bool_]:
    delta = np.arctan2(
        np.sin(angle - plaque.angle_rad),
        np.cos(angle - plaque.angle_rad),
    )
    support = np.abs(delta) <= plaque.angular_width_rad / 2
    mask = np.zeros(radius.shape, dtype=bool)
    if not np.any(support):
        return mask

    offsets = delta[support]
    inner = _boundary_values(plaque.inner_radius, offsets, "inner_radius")
    outer = _boundary_values(plaque.outer_radius, offsets, "outer_radius")
    if np.any(inner < 0):
        raise ValueError("plaque inner_radius must be non-negative")
    if np.any(inner > outer):
        raise ValueError("plaque inner_radius must not exceed outer_radius")
    outside_artery = outer > artery_outer_radius_px
    replaces_outer_wall = has_wall & (outer >= artery_outer_radius_px)
    if np.any(outside_artery | replaces_outer_wall):
        raise ValueError("plaque outer_radius must preserve wall before the background")

    supported_radii = radius[support]
    mask[support] = (supported_radii >= inner) & (supported_radii <= outer)
    return mask


def _boundary_values(boundary, offsets: FloatArray, name: str) -> FloatArray:
    values = np.asarray(boundary(offsets), dtype=np.float64)
    try:
        values = np.broadcast_to(values, offsets.shape)
    except ValueError as error:
        raise ValueError(
            f"{name} must return a scalar or an array matching its input shape"
        ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} returned non-finite radii")
    return values
