from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

class ArteryClass(IntEnum):
    BACKGROUND = 0
    BOUNDARY = 1
    LUMEN = 2
    PLAQUE = 3


class AppearanceKind(IntEnum):
    """Kinds of visual material used to synthesize an input image.

    Anatomical values intentionally match :class:`ArteryClass`, allowing an
    artery label map to serve as its default appearance map.
    """

    BACKGROUND = ArteryClass.BACKGROUND.value
    BOUNDARY = ArteryClass.BOUNDARY.value
    LUMEN = ArteryClass.LUMEN.value
    PLAQUE = ArteryClass.PLAQUE.value
    SHADOW = 4


@dataclass(frozen=True)
class EmptyArteryConfig:
    lumen_radius_px: float = 73.0
    wall_thickness_px: float = 12.0
    image_size: tuple[int, int] = (256, 256)

    def __post_init__(self) -> None:
        if not isfinite(self.lumen_radius_px) or self.lumen_radius_px <= 0:
            raise ValueError("lumen_radius_px must be finite and positive")
        if not isfinite(self.wall_thickness_px) or self.wall_thickness_px < 0:
            raise ValueError("wall_thickness_px must be finite and non-negative")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions")
        maximum_radius = (min(self.image_size) - 1) / 2
        if self.lumen_radius_px + self.wall_thickness_px > maximum_radius:
            raise ValueError("empty artery must fit completely inside the image")


@dataclass(frozen=True)
class SourceConfig:
    """Dataset-level configuration for a collection of artery samples."""

    num_elements: int
    empty_artery: EmptyArteryConfig = field(default_factory=EmptyArteryConfig)

    def __post_init__(self) -> None:
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by source datasets."""
        return {
            "num_elements": self.num_elements,
            "empty_artery": {
                "image_size": list(self.empty_artery.image_size),
                "lumen_radius_px": self.empty_artery.lumen_radius_px,
                "wall_thickness_px": self.empty_artery.wall_thickness_px,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Load and validate a source configuration from decoded JSON."""
        expected = {"num_elements", "empty_artery"}
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
            empty_artery = dict(empty_artery)
            empty_artery["image_size"] = tuple(empty_artery["image_size"])
            return cls(
                num_elements=value["num_elements"],
                empty_artery=EmptyArteryConfig(**empty_artery),
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

    def to_dict(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {"minimum", "maximum"}:
            raise ValueError("invalid FloatRange fields")
        try:
            return cls(minimum=value["minimum"], maximum=value["maximum"])
        except TypeError as error:
            raise ValueError("invalid FloatRange value") from error


@dataclass(frozen=True)
class RigidConfig:
    angle: FloatRange = FloatRange(-np.pi, np.pi)
    dx: FloatRange = FloatRange(-0.5, 0.5)
    dy: FloatRange = FloatRange(-0.5, 0.5)

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        return self.angle.sample(rng), self.dx.sample(rng), self.dy.sample(rng)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"minimum": value.minimum, "maximum": value.maximum}
            for name, value in (
                ("angle", self.angle),
                ("dx", self.dx),
                ("dy", self.dy),
            )
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {"angle", "dx", "dy"}:
            raise ValueError("invalid RigidConfig fields")
        return cls(**{name: FloatRange.from_dict(value[name]) for name in value})


@dataclass(frozen=True)
class RigidRejectionConfig:
    minimum_foreground_margin_px: int = 1
    max_attempts: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_foreground_margin_px, bool)
            or not isinstance(self.minimum_foreground_margin_px, int)
            or self.minimum_foreground_margin_px < 0
        ):
            raise ValueError(
                "minimum_foreground_margin_px must be a non-negative integer"
            )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "minimum_foreground_margin_px": self.minimum_foreground_margin_px,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        expected = {"minimum_foreground_margin_px", "max_attempts"}
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("invalid RigidRejectionConfig fields")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("invalid RigidRejectionConfig value") from error


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for deterministic image noise.

    ``seed`` identifies the noise realization collection. Individual samples
    derive independent random streams from this seed and their source index.
    """

    speckle_std: float = 0.0
    speckle_mode: Literal["multiplicative", "additive"] = "multiplicative"
    black_rectangle_probability: float = 0.0
    black_rectangle_size: tuple[int, int] = (0, 0)
    seed: int = 0

    def __post_init__(self) -> None:
        if not isfinite(self.speckle_std) or self.speckle_std < 0:
            raise ValueError("speckle_std must be finite and non-negative")
        if self.speckle_mode not in ("multiplicative", "additive"):
            raise ValueError("speckle_mode must be 'multiplicative' or 'additive'")
        if (
            not isfinite(self.black_rectangle_probability)
            or not 0 <= self.black_rectangle_probability <= 1
        ):
            raise ValueError("black_rectangle_probability must be in [0, 1]")
        if len(self.black_rectangle_size) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.black_rectangle_size
        ):
            raise ValueError(
                "black_rectangle_size must contain two non-negative integers"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("noise seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "speckle_std": self.speckle_std,
            "speckle_mode": self.speckle_mode,
            "black_rectangle_probability": self.black_rectangle_probability,
            "black_rectangle_size": list(self.black_rectangle_size),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        legacy_fields = {
            "speckle_std",
            "black_rectangle_probability",
            "black_rectangle_size",
            "seed",
        }
        expected = legacy_fields | {"speckle_mode"}
        if not isinstance(value, dict) or value.keys() not in (
            legacy_fields,
            expected,
        ):
            raise ValueError("invalid NoiseConfig fields")
        try:
            return cls(
                speckle_std=value["speckle_std"],
                speckle_mode=value.get("speckle_mode", "multiplicative"),
                black_rectangle_probability=value["black_rectangle_probability"],
                black_rectangle_size=tuple(value["black_rectangle_size"]),
                seed=value["seed"],
            )
        except (TypeError, KeyError) as error:
            raise ValueError("invalid NoiseConfig value") from error


@dataclass(frozen=True)
class DeformationConfig:
    scales: float | int | list[float] = 14
    magnitude: float = 7.0
    integrations: int = 2
    voxsize: float = 1.0
    fractal_mode: Literal["blur", "upsample"] = "blur"

    def __post_init__(self) -> None:
        scales = self.scales if isinstance(self.scales, list) else [self.scales]
        try:
            invalid_scales = not scales or any(
                isinstance(scale, bool) or not isfinite(scale) or scale <= 0
                for scale in scales
            )
        except TypeError as error:
            raise ValueError("scales must contain positive finite numbers") from error
        if invalid_scales:
            raise ValueError("scales must contain positive finite numbers")
        if (
            isinstance(self.magnitude, bool)
            or not isfinite(self.magnitude)
            or self.magnitude < 0
        ):
            raise ValueError("magnitude must be finite and non-negative")
        if (
            isinstance(self.integrations, bool)
            or not isinstance(self.integrations, int)
            or self.integrations < 0
        ):
            raise ValueError("integrations must be a non-negative integer")
        if (
            isinstance(self.voxsize, bool)
            or not isfinite(self.voxsize)
            or self.voxsize <= 0
        ):
            raise ValueError("voxsize must be finite and positive")
        if self.fractal_mode not in {"blur", "upsample"}:
            raise ValueError("fractal_mode must be 'blur' or 'upsample'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scales": self.scales,
            "magnitude": self.magnitude,
            "integrations": self.integrations,
            "voxsize": self.voxsize,
            "fractal_mode": self.fractal_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        expected = {
            "scales",
            "magnitude",
            "integrations",
            "voxsize",
            "fractal_mode",
        }
        if value.keys() != expected:
            raise ValueError("invalid DeformationConfig fields")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("invalid DeformationConfig value") from error


@dataclass(frozen=True)
class DeformationRejectionConfig:
    """Acceptance criteria for sampled deformation fields."""

    minimum_jacobian: float = 0.0
    minimum_foreground_margin_px: int = 1
    max_attempts: int = 20

    def __post_init__(self) -> None:
        if not isfinite(self.minimum_jacobian):
            raise ValueError("minimum_jacobian must be finite")
        if (
            isinstance(self.minimum_foreground_margin_px, bool)
            or not isinstance(self.minimum_foreground_margin_px, int)
            or self.minimum_foreground_margin_px < 0
        ):
            raise ValueError(
                "minimum_foreground_margin_px must be a non-negative integer"
            )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_jacobian": self.minimum_jacobian,
            "minimum_foreground_margin_px": self.minimum_foreground_margin_px,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        expected = {
            "minimum_jacobian",
            "minimum_foreground_margin_px",
            "max_attempts",
        }
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("invalid DeformationRejectionConfig fields")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError("invalid DeformationRejectionConfig value") from error


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
