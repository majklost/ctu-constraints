"""Identity contracts for a future pre-rigid SDF cache."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self

import numpy as np
import torch

from constraints.datatools.datasets.types import SDFMode
from constraints.devices import DeviceSelection
from constraints.utils import signed_distance_kornia, signed_distance_scipy

from .recipes import Recipe
from .storage import write_json
from .types import ArteryClass

SDF_CACHE_IDENTITY_VERSION = 1
SDF_CACHE_MANIFEST_FILENAME = "manifest.json"
SDF_CACHE_ARRAY_FILENAME = "sdf.npy"


@dataclass(frozen=True)
class SDFCacheConfig:
    """Settings that determine the values and channels in a cached SDF."""

    mode: SDFMode = "scipy"
    foreground_classes: tuple[ArteryClass, ...] = (
        ArteryClass.BOUNDARY,
        ArteryClass.LUMEN,
        ArteryClass.PLAQUE,
    )

    def __post_init__(self) -> None:
        if self.mode not in ("kornia", "scipy"):
            raise ValueError("unsupported SDF mode")
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
            "mode": self.mode,
            "implementation_version": 1,
            "foreground_classes": [
                artery_class.name.lower() for artery_class in self.foreground_classes
            ],
            "dtype": "float32",
            "sign_convention": "negative_inside",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        expected = {
            "mode",
            "implementation_version",
            "foreground_classes",
            "dtype",
            "sign_convention",
        }
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("invalid SDFCacheConfig fields")
        if (
            value["implementation_version"] != 1
            or value["dtype"] != "float32"
            or value["sign_convention"] != "negative_inside"
            or not isinstance(value["foreground_classes"], list)
        ):
            raise ValueError("unsupported SDFCacheConfig value")
        try:
            return cls(
                mode=value["mode"],
                foreground_classes=tuple(
                    ArteryClass[name.upper()] for name in value["foreground_classes"]
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid SDFCacheConfig value") from error


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
            deformation=recipe.deformation_name,
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


class _PreRigidTargetDataset(Protocol):
    """Minimal dataset surface needed to materialize an SDF cache."""

    root: Path
    recipe: Recipe

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    def sdf_cache_identity(
        self, config: SDFCacheConfig | None = None
    ) -> SDFCacheIdentity: ...


def create_sdf_cache(
    dataset: _PreRigidTargetDataset,
    config: SDFCacheConfig | None = None,
    *,
    batch_size: int = 16,
    device: DeviceSelection = "auto",
) -> tuple[Path, Path]:
    """Materialize one content-addressed pre-rigid SDF cache.

    The supplied dataset must not apply a rigid transform. SDF computation is
    delegated to the established utility selected by :class:`SDFMode`.
    """
    if dataset.recipe.rigid is not None:
        raise ValueError("pre-rigid SDF caching requires a recipe without rigid")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    config = SDFCacheConfig() if config is None else config
    identity = dataset.sdf_cache_identity(config)
    cache_folder = identity.cache_directory(dataset.root)
    if cache_folder.exists():
        raise FileExistsError(f"SDF cache already exists: {cache_folder}")

    num_elements = len(dataset)
    if num_elements <= 0:
        raise ValueError("cannot cache an empty dataset")
    first_labels = _target_labels_array(dataset[0])
    height, width = first_labels.shape
    num_channels = len(config.foreground_classes)
    array_path = cache_folder / SDF_CACHE_ARRAY_FILENAME
    manifest_path = cache_folder / SDF_CACHE_MANIFEST_FILENAME
    temporary_array = cache_folder / f".{SDF_CACHE_ARRAY_FILENAME}.tmp"

    cache_folder.mkdir(parents=True)
    values = np.lib.format.open_memmap(
        temporary_array,
        mode="w+",
        dtype=np.float32,
        shape=(num_elements, num_channels, height, width),
    )
    try:
        for start in range(0, num_elements, batch_size):
            stop = min(start + batch_size, num_elements)
            labels = torch.stack(
                [
                    torch.from_numpy(first_labels)
                    if index == 0
                    else torch.from_numpy(_target_labels_array(dataset[index]))
                    for index in range(start, stop)
                ]
            )
            foreground = torch.stack(
                tuple(
                    labels == int(artery_class)
                    for artery_class in config.foreground_classes
                ),
                dim=1,
            )
            if config.mode == "scipy":
                sdf = signed_distance_scipy(foreground)
            else:
                sdf = signed_distance_kornia(foreground, device=device)
            values[start:stop] = sdf.detach().cpu().numpy()

        values.flush()
        temporary_array.replace(array_path)
        write_json(
            manifest_path,
            {
                "format_name": "composed-artificial-sdf-cache",
                "format_version": 1,
                "status": "complete",
                "cache_key": identity.digest,
                "identity": identity.to_dict(),
                "array": {
                    "relative_path": SDF_CACHE_ARRAY_FILENAME,
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "layout": "NCHW",
                },
            },
        )
    except BaseException:
        temporary_array.unlink(missing_ok=True)
        array_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        cache_folder.rmdir()
        raise
    finally:
        del values
    return array_path, manifest_path


def _target_labels_array(sample: Mapping[str, Any]) -> np.ndarray:
    labels = sample["target_labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("target_labels must have shape [H, W]")
    return np.array(labels, dtype=np.uint8, copy=True)
