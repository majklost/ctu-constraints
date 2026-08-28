"""Serializable instructions for recreating recipe artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Self

from .types import (
    DeformationConfig,
    DeformationRejectionConfig,
    PowerPlaqueSamplingRanges,
    RigidConfig,
    RigidRejectionConfig,
)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_optional_name(name: str | None, kind: str) -> None:
    if name is not None and (
        not isinstance(name, str) or not name or Path(name).name != name
    ):
        raise ValueError(f"{kind} name must be a filename component")


@dataclass(frozen=True)
class PowerPlaqueBackup:
    ranges: tuple[PowerPlaqueSamplingRanges, ...]
    seed: int
    lumen_radius_px: float | None = None

    def __post_init__(self) -> None:
        ranges = tuple(self.ranges)
        if not ranges or not all(
            isinstance(item, PowerPlaqueSamplingRanges) for item in ranges
        ):
            raise ValueError("plaque backup requires at least one sampling range")
        _validate_seed(self.seed)
        if self.lumen_radius_px is not None and (
            not isfinite(self.lumen_radius_px) or self.lumen_radius_px <= 0
        ):
            raise ValueError("lumen_radius_px must be finite and positive")
        object.__setattr__(self, "ranges", ranges)

    def to_dict(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        for ranges in self.ranges:
            sampling = ranges.to_dict()
            if groups and groups[-1]["sampling"] == sampling:
                groups[-1]["count"] += 1
            else:
                groups.append({"count": 1, "sampling": sampling})
        return {
            "type": "power",
            "seed": self.seed,
            "lumen_radius_px": self.lumen_radius_px,
            "ranges": groups,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {
            "type",
            "seed",
            "lumen_radius_px",
            "ranges",
        }:
            raise ValueError("invalid PowerPlaqueBackup fields")
        if value["type"] != "power" or not isinstance(value["ranges"], list):
            raise ValueError("invalid power plaque backup")
        ranges: list[PowerPlaqueSamplingRanges] = []
        for group in value["ranges"]:
            if (
                not isinstance(group, dict)
                or group.keys() != {"count", "sampling"}
                or isinstance(group["count"], bool)
                or not isinstance(group["count"], int)
                or group["count"] <= 0
            ):
                raise ValueError("invalid power plaque range group")
            ranges.extend(
                [PowerPlaqueSamplingRanges.from_dict(group["sampling"])]
                * group["count"]
            )
        return cls(
            ranges=tuple(ranges),
            seed=value["seed"],
            lumen_radius_px=value["lumen_radius_px"],
        )


@dataclass(frozen=True)
class DeformationBackup:
    config: DeformationConfig = field(default_factory=DeformationConfig)
    rejection: DeformationRejectionConfig = field(
        default_factory=DeformationRejectionConfig
    )
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "rejection": self.rejection.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {
            "seed",
            "config",
            "rejection",
        }:
            raise ValueError("invalid DeformationBackup fields")
        return cls(
            config=DeformationConfig.from_dict(value["config"]),
            rejection=DeformationRejectionConfig.from_dict(value["rejection"]),
            seed=value["seed"],
        )


@dataclass(frozen=True)
class RigidBackup:
    config: RigidConfig = field(default_factory=RigidConfig)
    rejection: RigidRejectionConfig = field(default_factory=RigidRejectionConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "rejection": self.rejection.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {
            "seed",
            "config",
            "rejection",
        }:
            raise ValueError("invalid RigidBackup fields")
        return cls(
            config=RigidConfig.from_dict(value["config"]),
            rejection=RigidRejectionConfig.from_dict(value["rejection"]),
            seed=value["seed"],
        )


@dataclass(frozen=True)
class SavedDeformation:
    name: str | None = None
    backup: DeformationBackup | None = None

    def __post_init__(self) -> None:
        _validate_optional_name(self.name, "deformation")
        if self.name is None and self.backup is None:
            raise ValueError("deformation requires a name or backup")


@dataclass(frozen=True)
class SavedRigid:
    name: str | None = None
    backup: RigidBackup | None = None

    def __post_init__(self) -> None:
        _validate_optional_name(self.name, "rigid")
        if self.name is None and self.backup is None:
            raise ValueError("rigid requires a name or backup")
