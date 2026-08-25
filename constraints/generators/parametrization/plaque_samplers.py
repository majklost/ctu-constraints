from dataclasses import dataclass
from math import isfinite, pi

import numpy as np

from .plaque_generators import PowerPlaqueParameters


@dataclass(frozen=True)
class FloatRange:
    """Bounds for a uniformly sampled floating-point parameter."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise ValueError("range bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("range minimum must not exceed maximum")

    def sample(self, rng: np.random.Generator) -> float:
        if self.minimum == self.maximum:
            return self.minimum
        return float(rng.uniform(self.minimum, self.maximum))


@dataclass(frozen=True)
class PowerPlaqueSamplingRanges:
    """Resolution-independent ranges for power-profile plaques.

    Angular measurements are in radians. ``inward_depth_fraction`` is relative
    to the lumen radius and ``wall_depth_fraction`` is relative to wall
    thickness. Sampling resolves both fractions to pixels.

    Angles do not need to be normalized to ``[-pi, pi]``. To cross the wrap
    point, use an unwrapped range such as 350 to 370 degrees in radians.
    """

    angle_rad: FloatRange = FloatRange(-pi, pi)
    angular_width_rad: FloatRange = FloatRange(pi / 12, pi / 3)
    inward_depth_fraction: FloatRange = FloatRange(0.05, 0.3)
    wall_depth_fraction: FloatRange = FloatRange(0.1, 0.5)
    shape_power: FloatRange = FloatRange(0.25, 2.0)

    def __post_init__(self) -> None:
        if self.angular_width_rad.minimum <= 0:
            raise ValueError("angular_width_rad must be positive")
        if self.angular_width_rad.maximum > 2 * pi:
            raise ValueError("angular_width_rad must not exceed 2*pi")
        if self.inward_depth_fraction.minimum <= 0:
            raise ValueError("inward_depth_fraction must be positive")
        if self.inward_depth_fraction.maximum >= 1:
            raise ValueError("inward_depth_fraction must be less than 1")
        if self.wall_depth_fraction.minimum < 0:
            raise ValueError("wall_depth_fraction must be non-negative")
        if self.wall_depth_fraction.maximum >= 1:
            raise ValueError(
                "wall_depth_fraction must be less than 1 to preserve outer wall"
            )
        if self.shape_power.minimum <= 0:
            raise ValueError("shape_power must be positive")


def sample_power_plaque_parameters(
    ranges: PowerPlaqueSamplingRanges,
    *,
    lumen_radius_px: float,
    wall_thickness_px: float,
    rng: np.random.Generator,
) -> PowerPlaqueParameters:
    """Sample serializable plaque parameters from uniform ranges.

    The caller owns ``rng`` so results are reproducible and independent of
    NumPy's global random state.
    """
    if not isfinite(lumen_radius_px) or lumen_radius_px <= 0:
        raise ValueError("lumen_radius_px must be finite and positive")
    if not isfinite(wall_thickness_px) or wall_thickness_px < 0:
        raise ValueError("wall_thickness_px must be finite and non-negative")

    return PowerPlaqueParameters(
        angle_rad=ranges.angle_rad.sample(rng),
        angular_width_rad=ranges.angular_width_rad.sample(rng),
        inward_depth_px=(ranges.inward_depth_fraction.sample(rng) * lumen_radius_px),
        wall_depth_px=ranges.wall_depth_fraction.sample(rng) * wall_thickness_px,
        shape_power=ranges.shape_power.sample(rng),
    )


def sample_power_plaque_parameter_batch(
    ranges: PowerPlaqueSamplingRanges,
    count: int,
    *,
    lumen_radius_px: float,
    wall_thickness_px: float,
    rng: np.random.Generator,
) -> tuple[PowerPlaqueParameters, ...]:
    """Sample ``count`` independent plaques from the same ranges."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return tuple(
        sample_power_plaque_parameters(
            ranges,
            lumen_radius_px=lumen_radius_px,
            wall_thickness_px=wall_thickness_px,
            rng=rng,
        )
        for _ in range(count)
    )
