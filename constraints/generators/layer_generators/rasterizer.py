"""Reusable cyclic-polar rasterization for procedural layer generators."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi

import numpy as np

from ..types import EmptyArteryConfig, FloatArray

RadialBoundary = Callable[[FloatArray], FloatArray | float]


@dataclass(frozen=True)
class PlaqueSpec:
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
class CyclicRasterizer:
    artery: EmptyArteryConfig

    def __call__(self, specs: tuple[PlaqueSpec, ...]) -> np.ndarray:
        radius, angle = polar_grid(self.artery.image_size)
        result = np.zeros(self.artery.image_size, dtype=bool)
        outer_artery = self.artery.lumen_radius_px + self.artery.wall_thickness_px

        for spec in specs:
            delta = np.arctan2(
                np.sin(angle - spec.angle_rad), np.cos(angle - spec.angle_rad)
            )
            support = np.abs(delta) <= spec.angular_width_rad / 2
            if not np.any(support):
                continue
            offsets = delta[support]
            inner = _boundary_values(spec.inner_radius, offsets, "inner_radius")
            outer = _boundary_values(spec.outer_radius, offsets, "outer_radius")
            if np.any(inner < 0) or np.any(inner > outer):
                raise ValueError("plaque radial boundaries are invalid")
            if np.any(outer > outer_artery) or (
                self.artery.wall_thickness_px > 0 and np.any(outer >= outer_artery)
            ):
                raise ValueError("plaque must preserve wall before the background")
            values = radius[support]
            current = np.zeros(self.artery.image_size, dtype=bool)
            current[support] = (values >= inner) & (values <= outer)
            result |= current
        return result


def polar_grid(image_size: tuple[int, int]) -> tuple[FloatArray, FloatArray]:
    height, width = image_size
    center_y, center_x = (height - 1) / 2, (width - 1) / 2
    y, x = np.ogrid[:height, :width]
    return np.hypot(x - center_x, y - center_y), np.arctan2(y - center_y, x - center_x)


def _boundary_values(
    boundary: RadialBoundary, offsets: FloatArray, name: str
) -> FloatArray:
    values = np.asarray(boundary(offsets), dtype=np.float64)
    try:
        values = np.broadcast_to(values, offsets.shape)
    except ValueError as error:
        raise ValueError(f"{name} must return a scalar or matching array") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} returned non-finite radii")
    return values
