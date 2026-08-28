"""Small runtime contract shared by all layer generators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..types import AppearanceKind, ArteryClass, SourceConfig

if TYPE_CHECKING:
    from ..recipe_backups import LayerBackup

TRANSPARENT_LABEL = -1


@dataclass(frozen=True)
class LayerPatch:
    """Independent label and grayscale patches.

    ``labels == -1`` and ``image == NaN`` are transparent. All other pixels
    overwrite the current composition, independently for each array.
    """

    labels: NDArray[np.signedinteger]
    image: NDArray[np.floating]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        image = np.asarray(self.image)
        if labels.ndim != 2 or image.shape != labels.shape:
            raise ValueError("layer labels and image must have the same [H, W] shape")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("layer labels must have an integer dtype")
        valid_labels = labels != TRANSPARENT_LABEL
        if not np.isin(labels[valid_labels], [int(item) for item in ArteryClass]).all():
            raise ValueError("layer labels contain an unknown artery class")
        if not np.issubdtype(image.dtype, np.floating):
            raise TypeError("layer image must have a floating dtype")
        if np.isinf(image).any():
            raise ValueError("layer image may contain finite values or NaN only")
        object.__setattr__(self, "labels", labels.astype(np.int8, copy=False))
        object.__setattr__(self, "image", image.astype(np.float32, copy=False))


@dataclass(frozen=True)
class MaskLayer:
    """Convenience output for one mask with uniform label and appearance."""

    mask: NDArray[np.bool_]
    target_class: ArteryClass = ArteryClass.PLAQUE
    appearance: AppearanceKind | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("layer mask must have shape [H, W]")
        target_class = ArteryClass(self.target_class)
        if target_class is ArteryClass.BACKGROUND:
            raise ValueError("mask layer target must be a foreground class")
        object.__setattr__(self, "mask", mask.astype(bool, copy=False))
        object.__setattr__(self, "target_class", target_class)
        if self.appearance is not None:
            object.__setattr__(self, "appearance", AppearanceKind(self.appearance))


type LayerOutput = LayerPatch | MaskLayer


@dataclass(frozen=True)
class LayerResolverContext:
    source_root: Path
    source_config: SourceConfig
    sample_index: int


class LayerGenerator(Protocol):
    """Callable boundary used by notebook and persisted layer resolvers."""

    def __call__(
        self, context: LayerResolverContext, params: Mapping[str, Any]
    ) -> LayerOutput: ...


type LayerResolver = Callable[[LayerResolverContext, Mapping[str, Any]], LayerOutput]


@dataclass(frozen=True)
class LayerCollection:
    labels: np.ndarray
    image: np.ndarray


@dataclass(frozen=True)
class SavedLayer:
    """Reference to a stored collection, optionally with creation parameters."""

    name: str | None = None
    backup: LayerBackup | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (
            not isinstance(self.name, str)
            or not self.name
            or Path(self.name).name != self.name
        ):
            raise ValueError("layer collection name must be a filename component")
        if self.name is None and self.backup is None:
            raise ValueError("saved layer requires a name or backup")
        if self.backup is not None:
            from ..recipe_backups import LayerBackup

            if not isinstance(self.backup, LayerBackup):
                raise TypeError("layer backup must be a LayerBackup")
