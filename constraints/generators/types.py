from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite, pi
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PlaqueGenerationMethod = Literal["power"]


class ArteryClass(IntEnum):
    BACKGROUND = 0
    BOUNDARY = 1
    LUMEN = 2
    PLAQUE = 3


@dataclass(frozen=True)
class EmptyArteryConfig:
    lumen_radius_px: float = 73.0
    wall_thickness_px: float = 12.0

    def __post_init__(self) -> None:
        if not isfinite(self.lumen_radius_px) or self.lumen_radius_px <= 0:
            raise ValueError("lumen_radius_px must be finite and positive")
        if not isfinite(self.wall_thickness_px) or self.wall_thickness_px < 0:
            raise ValueError("wall_thickness_px must be finite and non-negative")


@dataclass(frozen=True)
class SourceConfig:
    """
    Minimal information to create a new dataset
    """

    num_elements: int
    image_size: tuple[int, int] = (256, 256)
    empty_artery: EmptyArteryConfig = field(default_factory=EmptyArteryConfig)
    plaque_generation_method: PlaqueGenerationMethod = "power"

    def __post_init__(self) -> None:
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")
        if self.plaque_generation_method != "power":
            raise ValueError("unsupported plaque_generation_method")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        maximum_radius = (min(self.image_size) - 1) / 2
        outer_radius = (
            self.empty_artery.lumen_radius_px
            + self.empty_artery.wall_thickness_px
        )
        if outer_radius > maximum_radius:
            raise ValueError("empty artery must fit completely inside the image")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by source datasets."""
        return {
            "num_elements": self.num_elements,
            "image_size": list(self.image_size),
            "empty_artery": {
                "lumen_radius_px": self.empty_artery.lumen_radius_px,
                "wall_thickness_px": self.empty_artery.wall_thickness_px,
            },
            "plaque_generation_method": self.plaque_generation_method,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Load and validate a source configuration from decoded JSON."""
        expected = {
            "num_elements",
            "image_size",
            "empty_artery",
            "plaque_generation_method",
        }
        unknown = value.keys() - expected
        missing = expected - value.keys()
        if unknown or missing:
            raise ValueError(
                f"invalid SourceConfig fields; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        empty_artery = value["empty_artery"]
        if not isinstance(empty_artery, dict):
            raise ValueError("empty_artery must be an object")
        try:
            return cls(
                num_elements=value["num_elements"],
                image_size=tuple(value["image_size"]),
                empty_artery=EmptyArteryConfig(**empty_artery),
                plaque_generation_method=value["plaque_generation_method"],
            )
        except (TypeError, KeyError) as error:
            raise ValueError("invalid SourceConfig value") from error


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

    @classmethod
    def fixed(cls, number: float) -> Self:
        return cls(minimum=number, maximum=number)


@dataclass(frozen=True)
class RigidBounds:
    angle: FloatRange
    dx: FloatRange
    dy: FloatRange

    def __post_init__(self) -> None:
        "TODO - do checks"


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
class NoiseConfig:
    speckle_std: float = 0.0
    black_rectangle_probability: float = 0.0
    black_rectangle_size: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class DeformationConfig:
    shape: tuple[int, int]
    scales: float | int | list[float] = 14
    magnitude: float = 7.0
    integrations: int = 2
    voxsize: float = 1.0
    fractal_mode: Literal["blur", "upsample"] = "blur"
    max_attempts: int = 1


# REWORK (now we dont need this)
# @dataclass(frozen=True)
# class ArteryParameters:
#     """Serializable subset of :class:`ArterySpec` used by source artifacts."""

#     image_size: tuple[int, int] = (256, 256)
#     center_yx_px: tuple[float, float] | None = None
#     lumen_radius_px: float = 73.0
#     wall_thickness_px: float = 12.0

#     def resolve(self) -> ArterySpec:
#         return ArterySpec(**asdict(self))
