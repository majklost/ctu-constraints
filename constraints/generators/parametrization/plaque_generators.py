from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

from ..types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatArray,
    PowerPlaqueParameters,
)

RadialBoundary = Callable[[FloatArray], FloatArray | float]


@dataclass(frozen=True)
class _PlaqueSpec:
    """A resolved plaque described by two radial boundary functions.

    Both functions receive the wrapped angular displacement from ``angle_rad``
    in radians and return absolute radii in pixels. They must support NumPy
    arrays, or return a scalar that can be broadcast to the input shape.

    ``_PlaqueSpec`` is deliberately a runtime rendering object. Persist the
    serializable parameters used by a plaque factory rather than attempting to
    serialize arbitrary callables.
    """

    angle_rad: float
    angular_width_rad: float
    inner_radius: RadialBoundary
    outer_radius: RadialBoundary

    def __post_init__(self) -> None:
        if not isfinite(self.angle_rad):
            raise ValueError("angle_rad must be finite")
        if not isfinite(self.angular_width_rad):
            raise ValueError("angular_width_rad must be finite")
        if not 0 < self.angular_width_rad <= 2 * pi:
            raise ValueError("angular_width_rad must be in (0, 2*pi]")


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


def create_empty_artery(
    config: EmptyArteryConfig,
) -> NDArray[np.uint8]:
    """Rasterize a centered artery label map without plaques."""
    outer_radius = config.lumen_radius_px + config.wall_thickness_px
    radius, _ = _polar_grid(config.image_size)
    labels = np.full(
        config.image_size,
        ArteryClass.BACKGROUND,
        dtype=np.uint8,
    )
    labels[radius <= outer_radius] = ArteryClass.BOUNDARY
    labels[radius <= config.lumen_radius_px] = ArteryClass.LUMEN
    return labels


def create_power_plaque_mask(
    parameters: tuple[PowerPlaqueParameters, ...],
    artery_config: EmptyArteryConfig,
    *,
    lumen_radius_px: float | None = None,
) -> NDArray[np.bool_]:
    """Resolve power parameters and rasterize their union as one Boolean mask."""
    if lumen_radius_px is None:
        lumen_radius_px = artery_config.lumen_radius_px
    plaque_specs = tuple(
        create_power_plaque(item, lumen_radius_px) for item in parameters
    )
    return _create_plaque_mask(plaque_specs, artery_config)


def _create_plaque_mask(
    plaque_specs: tuple[_PlaqueSpec, ...],
    artery_config: EmptyArteryConfig,
) -> NDArray[np.bool_]:
    """Rasterize the union of internal plaque specifications."""
    radius, angle = _polar_grid(artery_config.image_size)
    combined = np.zeros(artery_config.image_size, dtype=bool)
    outer_artery_radius = (
        artery_config.lumen_radius_px + artery_config.wall_thickness_px
    )
    has_wall = artery_config.wall_thickness_px > 0

    for plaque in plaque_specs:
        delta = np.arctan2(
            np.sin(angle - plaque.angle_rad),
            np.cos(angle - plaque.angle_rad),
        )
        support = np.abs(delta) <= plaque.angular_width_rad / 2
        if not np.any(support):
            continue

        offsets = delta[support]
        inner = _boundary_values(plaque.inner_radius, offsets, "inner_radius")
        outer = _boundary_values(plaque.outer_radius, offsets, "outer_radius")
        if np.any(inner < 0):
            raise ValueError("plaque inner_radius must be non-negative")
        if np.any(inner > outer):
            raise ValueError("plaque inner_radius must not exceed outer_radius")
        if np.any(outer > outer_artery_radius) or (
            has_wall and np.any(outer >= outer_artery_radius)
        ):
            raise ValueError(
                "plaque outer_radius must preserve wall before the background"
            )

        supported_radii = radius[support]
        plaque_mask = np.zeros(artery_config.image_size, dtype=bool)
        plaque_mask[support] = (supported_radii >= inner) & (supported_radii <= outer)
        combined |= plaque_mask

    return combined


def create_power_plaque(
    parameters: PowerPlaqueParameters, lumen_radius_px: float
) -> _PlaqueSpec:
    """Turn serializable parameters into a callable runtime plaque spec."""
    if not isfinite(lumen_radius_px) or lumen_radius_px <= 0:
        raise ValueError("lumen_radius_px must be finite and positive")

    return _PlaqueSpec(
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


def _polar_grid(image_size: tuple[int, int]) -> tuple[FloatArray, FloatArray]:
    height, width = image_size
    center_y, center_x = (height - 1) / 2, (width - 1) / 2
    y, x = np.ogrid[:height, :width]
    dy = y - center_y
    dx = x - center_x
    return np.hypot(dx, dy), np.arctan2(dy, dx)


def _boundary_values(
    boundary: RadialBoundary, offsets: FloatArray, name: str
) -> FloatArray:
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
