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
    SourceConfig,
)

RadialBoundary = Callable[[FloatArray], FloatArray | float]


@dataclass(frozen=True)
class PlaqueSpec:
    """A resolved plaque described by two radial boundary functions.

    Both functions receive the wrapped angular displacement from ``angle_rad``
    in radians and return absolute radii in pixels. They must support NumPy
    arrays, or return a scalar that can be broadcast to the input shape.

    ``PlaqueSpec`` is deliberately a runtime rendering object. Persist the
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
    emp: EmptyArteryConfig,
    image_size: tuple[int, int] = (256, 256),
) -> NDArray[np.uint8]:
    """Rasterize a centered artery label map without plaques."""
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")

    outer_radius = emp.lumen_radius_px + emp.wall_thickness_px
    if outer_radius > (min(image_size) - 1) / 2:
        raise ValueError("empty artery must fit completely inside the image")

    radius, _ = _polar_grid(image_size)
    labels = np.full(image_size, ArteryClass.BACKGROUND, dtype=np.uint8)
    labels[radius <= outer_radius] = ArteryClass.BOUNDARY
    labels[radius <= emp.lumen_radius_px] = ArteryClass.LUMEN
    return labels


def create_plaque_mask(
    plaque_specs: tuple[PlaqueSpec, ...], source_conf: SourceConfig
) -> NDArray[np.bool_]:
    """Rasterize the union of resolved plaques for one source sample."""
    radius, angle = _polar_grid(source_conf.image_size)
    combined = np.zeros(source_conf.image_size, dtype=bool)
    outer_artery_radius = (
        source_conf.empty_artery.lumen_radius_px
        + source_conf.empty_artery.wall_thickness_px
    )
    has_wall = source_conf.empty_artery.wall_thickness_px > 0

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
        plaque_mask = np.zeros(source_conf.image_size, dtype=bool)
        plaque_mask[support] = (supported_radii >= inner) & (
            supported_radii <= outer
        )
        combined |= plaque_mask

    return combined


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
