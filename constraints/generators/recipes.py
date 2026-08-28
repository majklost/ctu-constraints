"""Serializable recipes for composing and materializing artificial datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from constraints.devices import DeviceSelection

from .layer_generators import SavedLayer
from .recipe_backups import (
    DeformationBackup,
    LayerBackup,
    RigidBackup,
    SavedDeformation,
    SavedRigid,
)
from .storage import write_json
from .types import NoiseConfig

RECIPE_FORMAT_NAME = "composed-artificial-recipe"
RECIPE_FORMAT_VERSION = 5

if TYPE_CHECKING:
    from .recipe_ensure import EnsureReport
    from .sdf_cache import SDFCacheConfig


@dataclass(frozen=True)
class Recipe:
    """One central definition for preview, materialization, and composition."""

    source: str | None = None
    layers: tuple[SavedLayer, ...] = ()
    deformation: SavedDeformation | str | None = None
    rigid: SavedRigid | str | None = None
    noise: NoiseConfig | None = None
    sdf_cache: SDFCacheConfig | None = None

    format_version: ClassVar[int] = RECIPE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.source is not None:
            source = Path(self.source)
            if source.is_absolute() or not source.parts or ".." in source.parts:
                raise ValueError("recipe source must be a safe relative path")
            object.__setattr__(self, "source", source.as_posix())

        layers = tuple(self.layers)
        if not all(isinstance(layer, SavedLayer) for layer in layers):
            raise TypeError("recipe layers must contain SavedLayer instances")
        names = [layer.name for layer in layers if layer.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("a recipe cannot contain a layer collection twice")
        object.__setattr__(self, "layers", layers)

        if isinstance(self.deformation, str):
            object.__setattr__(
                self, "deformation", SavedDeformation(name=self.deformation)
            )
        elif self.deformation is not None and not isinstance(
            self.deformation, SavedDeformation
        ):
            raise TypeError("recipe deformation must be a SavedDeformation or None")

        if isinstance(self.rigid, str):
            object.__setattr__(self, "rigid", SavedRigid(name=self.rigid))
        elif self.rigid is not None and not isinstance(self.rigid, SavedRigid):
            raise TypeError("recipe rigid must be a SavedRigid or None")

        if self.noise is not None and not isinstance(self.noise, NoiseConfig):
            raise TypeError("recipe noise must be a NoiseConfig instance or None")
        if self.sdf_cache is not None:
            from .sdf_cache import SDFCacheConfig

            if not isinstance(self.sdf_cache, SDFCacheConfig):
                raise TypeError("recipe sdf_cache must be an SDFCacheConfig or None")

    @property
    def deformation_name(self) -> str | None:
        return None if self.deformation is None else self.deformation.name

    @property
    def rigid_name(self) -> str | None:
        return None if self.rigid is None else self.rigid.name

    def require_resolved(self) -> None:
        """Require names for every artifact needed by a stored dataset."""
        missing = [
            f"layers[{index}]"
            for index, item in enumerate(self.layers)
            if item.name is None
        ]
        if self.deformation is not None and self.deformation.name is None:
            missing.append("deformation")
        if self.rigid is not None and self.rigid.name is None:
            missing.append("rigid")
        if missing:
            raise ValueError(f"recipe has unnamed artifacts: {', '.join(missing)}")

    def resolve_source_root(self, source_root: Path | None = None) -> Path:
        if source_root is not None:
            return Path(source_root)
        if self.source is None:
            raise ValueError("recipe has no source; pass source_root explicitly")
        from constraints.utils import get_data_folder

        data_root = get_data_folder().resolve()
        resolved = (data_root / self.source).resolve()
        if not resolved.is_relative_to(data_root):
            raise ValueError("recipe source resolves outside the data folder")
        return resolved

    def with_names(
        self,
        *,
        layers: Mapping[int, str] | None = None,
        deformation: str | None = None,
        rigid: str | None = None,
    ) -> Self:
        """Return a copy with names assigned to dynamic artifacts."""
        layer_names = {} if layers is None else dict(layers)
        resolved_layers = tuple(
            replace(item, name=layer_names.get(index, item.name))
            for index, item in enumerate(self.layers)
        )
        resolved_deformation = self.deformation
        if deformation is not None:
            if resolved_deformation is None:
                raise ValueError("cannot name a recipe without deformation")
            resolved_deformation = replace(resolved_deformation, name=deformation)
        resolved_rigid = self.rigid
        if rigid is not None:
            if resolved_rigid is None:
                raise ValueError("cannot name a recipe without rigid")
            resolved_rigid = replace(resolved_rigid, name=rigid)
        return replace(
            self,
            layers=resolved_layers,
            deformation=resolved_deformation,
            rigid=resolved_rigid,
        )

    def ensure(
        self,
        source_root: Path | None = None,
        *,
        overwrite: bool = False,
        device: DeviceSelection = "auto",
        sdf_batch_size: int = 16,
    ) -> EnsureReport:
        """Create or validate all stored artifacts after a complete preflight."""
        from .recipe_ensure import ensure_recipe

        return ensure_recipe(
            self,
            source_root,
            overwrite=overwrite,
            device=device,
            sdf_batch_size=sdf_batch_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": RECIPE_FORMAT_NAME,
            "format_version": self.format_version,
            "source": self.source,
            "layers": [_layer_to_dict(item) for item in self.layers],
            "deformation": _deformation_to_dict(self.deformation),
            "rigid": _rigid_to_dict(self.rigid),
            "noise": None if self.noise is None else self.noise.to_dict(),
            "sdf_cache": (None if self.sdf_cache is None else self.sdf_cache.to_dict()),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if (
            not isinstance(value, dict)
            or value.get("format_name") != RECIPE_FORMAT_NAME
        ):
            raise ValueError("unsupported recipe format")
        version = value.get("format_version")
        if version != RECIPE_FORMAT_VERSION:
            raise ValueError("unsupported recipe format version")
        expected = {
            "format_name",
            "format_version",
            "source",
            "layers",
            "deformation",
            "rigid",
            "noise",
            "sdf_cache",
        }
        if value.keys() != expected:
            raise ValueError("invalid Recipe fields")
        try:
            from .sdf_cache import SDFCacheConfig

            return cls(
                source=value["source"],
                layers=tuple(_layer_from_dict(item) for item in value["layers"]),
                deformation=_deformation_from_dict(value["deformation"]),
                rigid=_rigid_from_dict(value["rigid"]),
                noise=(
                    None
                    if value["noise"] is None
                    else NoiseConfig.from_dict(value["noise"])
                ),
                sdf_cache=(
                    None
                    if value["sdf_cache"] is None
                    else SDFCacheConfig.from_dict(value["sdf_cache"])
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid Recipe value") from error

    def save_json(self, path: Path) -> None:
        write_json(Path(path), self.to_dict())

    @classmethod
    def load_json(cls, path: Path) -> Self:
        try:
            value = json.loads(Path(path).read_text())
        except json.JSONDecodeError as error:
            raise ValueError("invalid Recipe JSON") from error
        return cls.from_dict(value)


def _layer_to_dict(item: SavedLayer) -> dict[str, Any]:
    return {
        "name": item.name,
        "backup": None if item.backup is None else item.backup.to_dict(),
    }


def _layer_from_dict(value: dict[str, Any]) -> SavedLayer:
    expected = {"name", "backup"}
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("invalid Recipe layer fields")
    return SavedLayer(
        name=value["name"],
        backup=(
            None if value["backup"] is None else LayerBackup.from_dict(value["backup"])
        ),
    )


def _deformation_to_dict(item: SavedDeformation | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "name": item.name,
        "backup": None if item.backup is None else item.backup.to_dict(),
    }


def _deformation_from_dict(value: dict[str, Any] | None) -> SavedDeformation | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.keys() != {"name", "backup"}:
        raise ValueError("invalid Recipe deformation fields")
    return SavedDeformation(
        name=value["name"],
        backup=(
            None
            if value["backup"] is None
            else DeformationBackup.from_dict(value["backup"])
        ),
    )


def _rigid_to_dict(item: SavedRigid | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "name": item.name,
        "backup": None if item.backup is None else item.backup.to_dict(),
    }


def _rigid_from_dict(value: dict[str, Any] | None) -> SavedRigid | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.keys() != {"name", "backup"}:
        raise ValueError("invalid Recipe rigid fields")
    return SavedRigid(
        name=value["name"],
        backup=(
            None if value["backup"] is None else RigidBackup.from_dict(value["backup"])
        ),
    )
