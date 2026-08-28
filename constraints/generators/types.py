from dataclasses import dataclass, field
from enum import IntEnum
from math import isfinite, pi
from pathlib import Path
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


@dataclass(frozen=True)
class PowerPlaqueSamplingRanges:
    """Resolution-independent ranges for power-profile plaques.

    Angular measurements are in radians. ``inward_depth_fraction`` is relative
    to the lumen radius and ``wall_depth_fraction`` is relative to wall
    thickness. Sampling resolves both fractions to pixels. ``offset_px_lumen``
    is a signed radial offset from the lumen boundary: negative values move a
    plaque into the lumen and positive values move it into the wall.

    Angles do not need to be normalized to ``[-pi, pi]``. To cross the wrap
    point, use an unwrapped range such as 350 to 370 degrees in radians.
    """

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
            raise ValueError(
                "wall_depth_fraction must be less than 1 to preserve outer wall"
            )
        if self.shape_power.minimum <= 0:
            raise ValueError("shape_power must be positive")

    def sample(
        self,
        num: int,
        *,
        lumen_radius_px: float,
        wall_thickness_px: float,
        rng: np.random.Generator,
    ) -> tuple["PowerPlaqueParameters", ...]:
        """Sample ``num`` resolved parameter sets from these ranges."""
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
                inward_depth_px=(
                    self.inward_depth_fraction.sample(rng) * lumen_radius_px
                ),
                wall_depth_px=(
                    self.wall_depth_fraction.sample(rng) * wall_thickness_px
                ),
                shape_power=self.shape_power.sample(rng),
                offset_px_lumen=self.offset_px_lumen.sample(rng),
            )
            for _ in range(num)
        )


@dataclass(frozen=True)
class PowerPlaqueParameters:
    """Serializable parameters for an ellipse-like polar plaque.

    Depths are resolved pixel measurements. ``shape_power=0.5`` produces the
    familiar square-root profile of an ellipse in local angular/radial
    coordinates. Larger values concentrate the plaque around its central angle.
    ``offset_px_lumen`` is added to the lumen radius before both radial
    boundaries are constructed.
    """

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
        if self.inward_depth_px <= 0:
            raise ValueError("inward_depth_px must be positive")
        if self.wall_depth_px < 0:
            raise ValueError("wall_depth_px must be non-negative")
        if self.shape_power <= 0:
            raise ValueError("shape_power must be positive")


@dataclass(frozen=True)
class SavedPlaque:
    """Reference to one stored mask collection and how it should be composed."""

    name: str
    target_class: ArteryClass = ArteryClass.PLAQUE
    appearance: AppearanceKind | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or Path(self.name).name != self.name
        ):
            raise ValueError("plaque collection name must be a filename component")
        target_class = ArteryClass(self.target_class)
        if target_class not in {
            ArteryClass.BOUNDARY,
            ArteryClass.LUMEN,
            ArteryClass.PLAQUE,
        }:
            raise ValueError("saved plaque target must be boundary, lumen, or plaque")
        object.__setattr__(self, "target_class", target_class)
        if self.appearance is not None:
            object.__setattr__(self, "appearance", AppearanceKind(self.appearance))


@dataclass(frozen=True)
class PlaqueLayer:
    """One plaque-like mask with independent target and visual meanings."""

    mask: NDArray[np.bool_]
    target_class: ArteryClass = ArteryClass.PLAQUE
    appearance: AppearanceKind | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("plaque layer mask must have shape [H, W]")
        target_class = ArteryClass(self.target_class)
        if target_class not in {
            ArteryClass.BOUNDARY,
            ArteryClass.LUMEN,
            ArteryClass.PLAQUE,
        }:
            raise ValueError("plaque layer target must be boundary, lumen, or plaque")
        object.__setattr__(self, "mask", mask.astype(bool, copy=False))
        object.__setattr__(self, "target_class", target_class)
        if self.appearance is not None:
            object.__setattr__(
                self,
                "appearance",
                AppearanceKind(self.appearance),
            )

    @property
    def resolved_appearance(self) -> AppearanceKind:
        """Use the target class as appearance when no override is supplied."""
        if self.appearance is not None:
            return self.appearance
        return AppearanceKind(self.target_class.value)


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
