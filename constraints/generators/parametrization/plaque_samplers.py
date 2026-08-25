from math import isfinite

import numpy as np

from ..types import (
    EmptyArteryConfig,
    PowerPlaqueParameters,
    PowerPlaqueSamplingRanges,
)


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
    ranges: PowerPlaqueSamplingRanges | tuple[PowerPlaqueSamplingRanges, ...],
    count: int,
    empty_artery_config: EmptyArteryConfig,
    rng: np.random.Generator,
) -> tuple[PowerPlaqueParameters, ...]:
    """Sample ``count`` plaques from shared or per-plaque ranges.

    A single :class:`PowerPlaqueSamplingRanges` is reused for every plaque. A
    tuple supplies one range configuration per plaque and must have length
    ``count``.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    if isinstance(ranges, tuple):
        if len(ranges) != count:
            raise ValueError(
                f"expected {count} plaque range configurations, got {len(ranges)}"
            )
        ranges_per_plaque = ranges
    else:
        ranges_per_plaque = (ranges,) * count

    return tuple(
        sample_power_plaque_parameters(
            plaque_ranges,
            lumen_radius_px=empty_artery_config.lumen_radius_px,
            wall_thickness_px=empty_artery_config.wall_thickness_px,
            rng=rng,
        )
        for plaque_ranges in ranges_per_plaque
    )
