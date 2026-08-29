"""Power-profile plaque generator and its serializable parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Self

import numpy as np

from ..recipe_backups import LayerBackup
from ..types import (
    AppearanceKind,
    ArteryClass,
    EmptyArteryConfig,
    FloatArray,
    FloatRange,
    SourceConfig,
)
from .rasterizer import CyclicRasterizer, PlaqueSpec
from .types import LayerResolverContext, MaskLayer


@dataclass(frozen=True)
class PowerPlaqueSamplingRanges:
    angle_rad: FloatRange = FloatRange(-pi, pi)
    angular_width_rad: FloatRange = FloatRange(pi / 12, pi / 3)
    inward_depth_fraction: FloatRange = FloatRange(0.05, 0.3)
    wall_depth_fraction: FloatRange = FloatRange(0.1, 0.5)
    shape_power: FloatRange = FloatRange(0.25, 2.0)
    offset_px_lumen: FloatRange = FloatRange.fixed(0.0)

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
            raise ValueError("wall_depth_fraction must be less than 1")
        if self.shape_power.minimum <= 0:
            raise ValueError("shape_power must be positive")

    def sample(
        self,
        num: int,
        *,
        lumen_radius_px: float,
        wall_thickness_px: float,
        rng: np.random.Generator,
    ) -> tuple[PowerPlaqueParameters, ...]:
        if isinstance(num, bool) or not isinstance(num, int) or num < 0:
            raise ValueError("num must be a non-negative integer")
        if not isfinite(lumen_radius_px) or lumen_radius_px <= 0:
            raise ValueError("lumen_radius_px must be finite and positive")
        if not isfinite(wall_thickness_px) or wall_thickness_px < 0:
            raise ValueError("wall_thickness_px must be finite and non-negative")
        return tuple(
            PowerPlaqueParameters(
                angle_rad=self.angle_rad.sample(rng),
                angular_width_rad=self.angular_width_rad.sample(rng),
                inward_depth_px=self.inward_depth_fraction.sample(rng)
                * lumen_radius_px,
                wall_depth_px=self.wall_depth_fraction.sample(rng) * wall_thickness_px,
                shape_power=self.shape_power.sample(rng),
                offset_px_lumen=self.offset_px_lumen.sample(rng),
            )
            for _ in range(num)
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: value.to_dict()
            for name, value in (
                ("angle_rad", self.angle_rad),
                ("angular_width_rad", self.angular_width_rad),
                ("inward_depth_fraction", self.inward_depth_fraction),
                ("wall_depth_fraction", self.wall_depth_fraction),
                ("shape_power", self.shape_power),
                ("offset_px_lumen", self.offset_px_lumen),
            )
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        expected = {
            "angle_rad",
            "angular_width_rad",
            "inward_depth_fraction",
            "wall_depth_fraction",
            "shape_power",
            "offset_px_lumen",
        }
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("invalid PowerPlaqueSamplingRanges fields")
        return cls(**{name: FloatRange.from_dict(item) for name, item in value.items()})


@dataclass(frozen=True)
class PowerPlaqueParameters:
    angle_rad: float
    angular_width_rad: float
    inward_depth_px: float
    wall_depth_px: float
    shape_power: float = 0.5
    offset_px_lumen: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.angle_rad,
            self.angular_width_rad,
            self.inward_depth_px,
            self.wall_depth_px,
            self.shape_power,
            self.offset_px_lumen,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("all power-plaque parameters must be finite")
        if not 0 < self.angular_width_rad <= 2 * pi:
            raise ValueError("angular_width_rad must be in (0, 2*pi]")
        if self.inward_depth_px <= 0 or self.wall_depth_px < 0:
            raise ValueError("invalid plaque depth")
        if self.shape_power <= 0:
            raise ValueError("shape_power must be positive")


@dataclass(frozen=True)
class PowerPlaqueSample:
    mask: np.ndarray
    parameters: tuple[PowerPlaqueParameters, ...]
    sample_seed: int


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
    if not isfinite(lumen_radius_px) or lumen_radius_px <= 0:
        raise ValueError("lumen_radius_px must be finite and positive")
    base_radius_px = lumen_radius_px + parameters.offset_px_lumen
    if base_radius_px <= 0:
        raise ValueError("lumen_radius_px + offset_px_lumen must be positive")
    common = {
        "base_radius_px": base_radius_px,
        "angular_width_rad": parameters.angular_width_rad,
        "shape_power": parameters.shape_power,
    }
    return PlaqueSpec(
        angle_rad=parameters.angle_rad,
        angular_width_rad=parameters.angular_width_rad,
        inner_radius=_PowerRadialBoundary(
            signed_depth_px=-parameters.inward_depth_px, **common
        ),
        outer_radius=_PowerRadialBoundary(
            signed_depth_px=parameters.wall_depth_px, **common
        ),
    )


def create_power_plaque_mask(
    parameters: tuple[PowerPlaqueParameters, ...],
    artery_config: EmptyArteryConfig,
    *,
    lumen_radius_px: float | None = None,
) -> np.ndarray:
    lumen_radius_px = (
        artery_config.lumen_radius_px if lumen_radius_px is None else lumen_radius_px
    )
    specs = tuple(create_power_plaque(item, lumen_radius_px) for item in parameters)
    return CyclicRasterizer(artery_config)(specs)


def sample_power_plaque_mask(
    config: SourceConfig,
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
    sample_index: int = 0,
    lumen_radius_px: float | None = None,
) -> PowerPlaqueSample:
    if seed < 0 or sample_index < 0:
        raise ValueError("seed and sample_index must be non-negative")
    ranges = PowerPlaqueSamplingRanges() if ranges is None else ranges
    ranges_per_plaque = ranges if isinstance(ranges, tuple) else (ranges,)
    if not ranges_per_plaque:
        raise ValueError("at least one plaque range is required")
    lumen_radius_px = (
        config.empty_artery.lumen_radius_px
        if lumen_radius_px is None
        else lumen_radius_px
    )
    sequence = np.random.SeedSequence([seed, sample_index])
    sample_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(sample_seed)
    parameters = tuple(
        parameter
        for item in ranges_per_plaque
        for parameter in item.sample(
            1,
            lumen_radius_px=lumen_radius_px,
            wall_thickness_px=config.empty_artery.wall_thickness_px,
            rng=rng,
        )
    )
    return PowerPlaqueSample(
        create_power_plaque_mask(
            parameters, config.empty_artery, lumen_radius_px=lumen_radius_px
        ),
        parameters,
        sample_seed,
    )


def power_layer_backup(
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
    lumen_radius_px: float | None = None,
    target_class: ArteryClass = ArteryClass.PLAQUE,
    appearance: AppearanceKind | None = None,
) -> LayerBackup:
    ranges = (PowerPlaqueSamplingRanges(),) if ranges is None else ranges
    ranges = (ranges,) if isinstance(ranges, PowerPlaqueSamplingRanges) else ranges
    if not ranges:
        raise ValueError("power layer requires at least one sampling range")
    groups: list[dict[str, Any]] = []
    for item in ranges:
        sampling = item.to_dict()
        if groups and groups[-1]["sampling"] == sampling:
            groups[-1]["count"] += 1
        else:
            groups.append({"count": 1, "sampling": sampling})
    return LayerBackup(
        "power-v2",
        {
            "seed": seed,
            "lumen_radius_px": lumen_radius_px,
            "ranges": groups,
            "target_class": ArteryClass(target_class).name.lower(),
            "appearance": (
                None if appearance is None else AppearanceKind(appearance).name.lower()
            ),
        },
    )


def _power_range_group(value: Any) -> tuple[PowerPlaqueSamplingRanges, ...]:
    if (
        not isinstance(value, dict)
        or value.keys() != {"count", "sampling"}
        or isinstance(value["count"], bool)
        or not isinstance(value["count"], int)
        or value["count"] <= 0
    ):
        raise ValueError("invalid power-v2 range group")
    return (PowerPlaqueSamplingRanges.from_dict(value["sampling"]),) * value["count"]
