"""Convenient orchestration used by generation scripts and datasets."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .composition import PlaqueLayer, compose_target_labels
from .rendering import (
    DEFAULT_CLASS_INTENSITIES,
    create_grayscale_image_from_label_mask,
)
from .source import generate_plaque_masks_power, load_source_config
from .types import ArteryClass, PowerPlaqueSamplingRanges, SourceConfig


@dataclass(frozen=True)
class ComposedSampleArrays:
    """NumPy representation produced before adapting to a dataset API."""

    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]


def get_source_config(source_root: Path) -> SourceConfig:
    """Return the validated configuration of an existing source dataset."""
    return load_source_config(source_root)


def create_plaque_collection(
    source_root: Path,
    name: str,
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
) -> tuple[Path, Path]:
    """Create a named plaque collection inside an existing source dataset.

    This is the script-facing API. It resolves the source configuration and
    storage location, then dispatches to the configured plaque generator.
    """
    source_root = Path(source_root)
    config = get_source_config(source_root)
    if config.plaque_generation_method != "power":
        raise ValueError(
            f"unsupported plaque generation method: "
            f"{config.plaque_generation_method}"
        )

    plaque_folder = source_root / "plaques"
    generate_plaque_masks_power(
        plaque_folder,
        name,
        config,
        ranges,
        seed=seed,
    )
    return plaque_folder / f"{name}.npy", plaque_folder / f"{name}.jsonl"


def compose_artificial_sample(
    empty_artery: np.ndarray,
    layers: Iterable[PlaqueLayer],
    class_intensities: Mapping[ArteryClass, float] = DEFAULT_CLASS_INTENSITIES,
) -> ComposedSampleArrays:
    """Compose target anatomy and its initial grayscale image.

    Fake plaque layers resolve to their configured anatomical target, but are
    rendered with plaque intensity. Later pipeline stages may add deformation,
    rigid movement, or image-only noise around this operation.
    """
    layers = tuple(layers)
    target_labels = compose_target_labels(empty_artery, layers)
    image = create_grayscale_image_from_label_mask(
        target_labels,
        class_intensities,
    )
    plaque_intensity = float(class_intensities[ArteryClass.PLAQUE])
    for layer in layers:
        image[np.asarray(layer.mask, dtype=bool)] = plaque_intensity
    return ComposedSampleArrays(image=image, target_labels=target_labels)
