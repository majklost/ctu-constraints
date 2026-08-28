"""Serializable instructions for recreating recipe artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from .types import (
    DeformationConfig,
    DeformationRejectionConfig,
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
class LayerBackup:
    """Portable call to a registered layer resolver."""

    resolver: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resolver, str) or not self.resolver:
            raise ValueError("layer resolver must be a non-empty string")
        try:
            params = json.loads(json.dumps(self.params, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "layer resolver params must be finite JSON values"
            ) from error
        if not isinstance(params, dict):
            raise ValueError("layer resolver params must be an object")
        object.__setattr__(self, "params", MappingProxyType(params))

    def to_dict(self) -> dict[str, Any]:
        return {"resolver": self.resolver, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if not isinstance(value, dict) or value.keys() != {"resolver", "params"}:
            raise ValueError("invalid LayerBackup fields")
        return cls(resolver=value["resolver"], params=value["params"])


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
