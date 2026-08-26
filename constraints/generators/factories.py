"""Convenient orchestration used by generation scripts and datasets."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .composition import PlaqueLayer, compose_target_labels
from .deformation import (
    apply_deformation,
    generate_deformation_fields,
    sample_valid_deformation,
)
from .parametrization.plaque_generators import create_empty_artery
from .rendering import (
    DEFAULT_CLASS_INTENSITIES,
    create_grayscale_image_from_label_mask,
)
from .source import (
    generate_plaque_masks_power,
    load_source_config,
    sample_power_plaque_mask,
)
from .types import (
    ArteryClass,
    DeformationConfig,
    DeformationRejectionConfig,
    PowerPlaqueParameters,
    PowerPlaqueSamplingRanges,
    SourceConfig,
)
from .validation import DeformationValidationResult


@dataclass(frozen=True)
class ComposedSampleArrays:
    """NumPy representation produced before adapting to a dataset API."""

    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]


@dataclass(frozen=True)
class PreviewArtificialSample:
    """Storage-free sample plus the random parameters used to create it."""

    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]
    plaque_parameters: tuple[PowerPlaqueParameters, ...]
    deformation_field: NDArray[np.float32] | None
    deformation_validation: DeformationValidationResult | None


def preview_artificial_sample(
    source_config: SourceConfig,
    plaque_ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    deformation_config: DeformationConfig | None = None,
    deformation_rejection: DeformationRejectionConfig | None = None,
    *,
    seed: int,
    sample_index: int = 0,
    deformation_device: torch.device | str = "cpu",
    class_intensities: Mapping[ArteryClass, float] = DEFAULT_CLASS_INTENSITIES,
) -> PreviewArtificialSample:
    """Create one directly inspectable sample without reading or writing files."""
    empty_artery = create_empty_artery(
        source_config.empty_artery,
        source_config.image_size,
    )
    plaque = sample_power_plaque_mask(
        source_config,
        plaque_ranges,
        seed=seed,
        sample_index=sample_index,
    )
    deformation = None
    plaque_mask = plaque.mask
    if deformation_config is not None:
        deformation = sample_valid_deformation(
            source_config,
            empty_artery,
            deformation_config,
            deformation_rejection,
            seed=seed,
            sample_index=sample_index,
            device=deformation_device,
        )
        empty_artery = np.rint(
            apply_deformation(empty_artery, deformation.field, method="nearest")
        ).astype(np.uint8)
        plaque_mask = (
            apply_deformation(plaque_mask, deformation.field, method="nearest")
            > 0.5
        )

    arrays = compose_artificial_sample(
        empty_artery,
        (PlaqueLayer("preview", plaque_mask, ArteryClass.PLAQUE),),
        class_intensities,
    )
    return PreviewArtificialSample(
        image=arrays.image,
        target_labels=arrays.target_labels,
        plaque_parameters=plaque.parameters,
        deformation_field=None if deformation is None else deformation.field,
        deformation_validation=(
            None if deformation is None else deformation.validation
        ),
    )


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


def create_deformation_collection(
    source_root: Path,
    name: str,
    config: DeformationConfig,
    rejection: DeformationRejectionConfig | None = None,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[Path, Path]:
    """Create one named deformation collection inside a source dataset."""
    source_root = Path(source_root)
    source_config = get_source_config(source_root)
    return generate_deformation_fields(
        source_root / "deformations",
        name,
        source_config,
        np.load(source_root / "empty_artery.npy", mmap_mode="r"),
        config,
        rejection,
        seed=seed,
        device=device,
    )


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
