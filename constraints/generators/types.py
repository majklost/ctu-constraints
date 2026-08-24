from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RadialBoundary = Callable[[FloatArray], FloatArray | float]


class ArteryClass(IntEnum):
    BACKGROUND = 0
    BOUNDARY = 1
    LUMEN = 2
    PLAQUE = 3


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
class ArterySpec:
    """Fully resolved geometry passed to the deterministic rasterizer.

    Spatial measurements are in pixels and angles are in radians. Sampling
    configurations may use relative measurements, but should resolve them to
    this representation before rendering.
    """

    image_size: tuple[int, int] = (256, 256)
    center_yx_px: tuple[float, float] | None = None
    lumen_radius_px: float = 73.0
    wall_thickness_px: float = 12.0
    plaques: tuple[PlaqueSpec, ...] = ()

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size dimensions must be positive")
        if not isfinite(self.lumen_radius_px) or self.lumen_radius_px <= 0:
            raise ValueError("lumen_radius_px must be finite and positive")
        if not isfinite(self.wall_thickness_px) or self.wall_thickness_px <= 0:
            raise ValueError("wall_thickness_px must be finite and positive")
        if self.center_yx_px is not None and not all(
            isfinite(value) for value in self.center_yx_px
        ):
            raise ValueError("center_yx_px values must be finite")

    @property
    def center(self) -> tuple[float, float]:
        if self.center_yx_px is not None:
            return self.center_yx_px
        height, width = self.image_size
        return (height - 1) / 2, (width - 1) / 2

    @property
    def outer_radius_px(self) -> float:
        return self.lumen_radius_px + self.wall_thickness_px
