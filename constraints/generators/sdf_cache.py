"""Identity contracts for a future pre-rigid SDF cache."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from .recipes import Recipe
from .types import ArteryClass

SDF_CACHE_IDENTITY_VERSION = 1
SDF_CACHE_MANIFEST_FILENAME = "manifest.json"
SDF_CACHE_ARRAY_FILENAME = "sdf.npy"


@dataclass(frozen=True)
class SDFCacheConfig:
    """Settings that determine the values and channels in a cached SDF."""

    implementation: Literal["scipy_edt"] = "scipy_edt"
    foreground_classes: tuple[ArteryClass, ...] = (
        ArteryClass.BOUNDARY,
        ArteryClass.LUMEN,
        ArteryClass.PLAQUE,
    )

    def __post_init__(self) -> None:
        if self.implementation != "scipy_edt":
            raise ValueError("unsupported SDF implementation")
        classes = tuple(ArteryClass(value) for value in self.foreground_classes)
        if not classes or ArteryClass.BACKGROUND in classes:
            raise ValueError(
                "SDF foreground classes must be non-empty and non-background"
            )
        if len(classes) != len(set(classes)):
            raise ValueError("SDF foreground classes must be unique")
        object.__setattr__(self, "foreground_classes", classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "implementation_version": 1,
            "foreground_classes": [
                artery_class.name.lower() for artery_class in self.foreground_classes
            ],
            "dtype": "float32",
            "sign_convention": "negative_inside",
        }


@dataclass(frozen=True)
class SDFCacheIdentity:
    """Versioned identity of target labels used by a pre-rigid SDF cache.

    This is an explicit projection rather than a hash of the complete Recipe.
    Adding an image-only Recipe field therefore cannot invalidate SDF caches.
    """

    source_dataset_id: str
    target_plaques: tuple[tuple[str, ArteryClass], ...]
    deformation: str | None
    sdf: SDFCacheConfig = field(default_factory=SDFCacheConfig)

    identity_version: ClassVar[int] = SDF_CACHE_IDENTITY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source_dataset_id, str) or not self.source_dataset_id:
            raise ValueError("source_dataset_id must be a non-empty string")
        target_plaques = tuple(
            (name, ArteryClass(target_class))
            for name, target_class in self.target_plaques
        )
        if any(
            not isinstance(name, str) or not name or Path(name).name != name
            for name, _ in target_plaques
        ):
            raise ValueError("SDF plaque names must be filename components")
        names = [name for name, _ in target_plaques]
        if len(names) != len(set(names)):
            raise ValueError("an SDF identity cannot contain a plaque twice")
        object.__setattr__(self, "target_plaques", target_plaques)

    @classmethod
    def from_recipe(
        cls,
        source_dataset_id: str,
        recipe: Recipe,
        sdf: SDFCacheConfig | None = None,
    ) -> "SDFCacheIdentity":
        """Select only recipe fields that affect a pre-rigid target SDF."""
        return cls(
            source_dataset_id=source_dataset_id,
            target_plaques=tuple(
                (plaque.name, plaque.target_class) for plaque in recipe.plaques
            ),
            deformation=recipe.deformation,
            sdf=SDFCacheConfig() if sdf is None else sdf,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical identity payload; edit this contract deliberately."""
        return {
            "identity_version": self.identity_version,
            "source_dataset_id": self.source_dataset_id,
            "target_composition": {
                "composition_contract": "ordered_overwrite_v1",
                "plaques": [
                    {"name": name, "target_class": target_class.name.lower()}
                    for name, target_class in self.target_plaques
                ],
            },
            "deformation": {
                "name": self.deformation,
                "application_contract": "backward_nearest_v1",
            },
            "sdf": self.sdf.to_dict(),
        }

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 of the explicit canonical identity payload."""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    @property
    def directory_name(self) -> str:
        return f"sdf-v{self.identity_version}-{self.digest}"

    def cache_directory(self, source_root: Path) -> Path:
        """Return the proposed location without creating it."""
        return Path(source_root) / "derived" / self.directory_name
