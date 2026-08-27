"""Convenient orchestration used by generation scripts and datasets."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from constraints.devices import DeviceSelection

from .composition import compose_label_maps
from .deformation import (
    apply_deformation,
    generate_deformation_fields,
    load_deformation_fields,
    sample_valid_deformation,
)
from .parametrization.plaque_generators import create_empty_artery
from .rendering import (
    DEFAULT_CLASS_INTENSITIES,
    create_grayscale_image_from_label_mask,
)
from .rigid import apply_rigid, generate_rigid_parameters, sample_valid_rigid
from .source import (
    generate_plaque_masks_power,
    load_source_config,
)
from .types import (
    AppearanceKind,
    DeformationConfig,
    DeformationRejectionConfig,
    EmptyArteryConfig,
    PlaqueLayer,
    PowerPlaqueSamplingRanges,
    RigidConfig,
    RigidRejectionConfig,
    SourceConfig,
)
from .validation import DeformationValidationResult


@dataclass(frozen=True)
class ComposedSampleArrays:
    """NumPy representation produced before adapting to a dataset API."""

    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]
    appearance_labels: NDArray[np.uint8]


@dataclass(frozen=True)
class PreviewArtificialSample:
    """Storage-free sample plus the random parameters used to create it."""

    image: NDArray[np.float32]
    target_labels: NDArray[np.uint8]
    appearance_labels: NDArray[np.uint8]
    deformation_field: NDArray[np.float32] | None
    deformation_validation: DeformationValidationResult | None
    rigid_parameters: NDArray[np.float32] | None = None


def preview_artificial_sample(
    artery_config: EmptyArteryConfig,
    plaque_layers: Iterable[PlaqueLayer] = (),
    deformation_config: DeformationConfig | None = None,
    deformation_rejection: DeformationRejectionConfig | None = None,
    rigid_config: RigidConfig | None = None,
    rigid_rejection: RigidRejectionConfig | None = None,
    *,
    seed: int,
    sample_index: int = 0,
    deformation_device: DeviceSelection = "auto",
    class_intensities: Mapping[AppearanceKind, float] = DEFAULT_CLASS_INTENSITIES,
) -> PreviewArtificialSample:
    """Transform and compose directly supplied Boolean plaque layers."""
    empty_artery = create_empty_artery(artery_config)
    plaque_layers = tuple(plaque_layers)

    deformation = None
    if deformation_config is not None:
        deformation = sample_valid_deformation(
            empty_artery,
            deformation_config,
            deformation_rejection,
            seed=seed + 1,
            sample_index=sample_index,
            device=deformation_device,
        )
    rigid_source = empty_artery
    if deformation is not None:
        rigid_source = np.rint(
            apply_deformation(empty_artery, deformation.field, method="nearest")
        ).astype(np.uint8)

    rigid = None
    if rigid_config is not None:
        rigid = sample_valid_rigid(
            rigid_source,
            rigid_config,
            rigid_rejection,
            seed=seed + 2,
            sample_index=sample_index,
        )
    arrays = compose_artificial_sample(
        empty_artery,
        plaque_layers,
        class_intensities,
        deformation_field=None if deformation is None else deformation.field,
        rigid_parameters=None if rigid is None else rigid.parameters,
    )
    return PreviewArtificialSample(
        image=arrays.image,
        target_labels=arrays.target_labels,
        appearance_labels=arrays.appearance_labels,
        deformation_field=None if deformation is None else deformation.field,
        deformation_validation=(
            None if deformation is None else deformation.validation
        ),
        rigid_parameters=None if rigid is None else rigid.parameters,
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
    device: DeviceSelection = "auto",
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


def create_rigid_collection(
    source_root: Path,
    name: str,
    config: RigidConfig,
    rejection: RigidRejectionConfig | None = None,
    *,
    deformation: str | None = None,
    seed: int,
) -> tuple[Path, Path]:
    """Create a source-level or deformation-dependent rigid preset."""
    source_root = Path(source_root)
    source_config = get_source_config(source_root)
    parent_folder = source_root
    fields = None
    if deformation is not None:
        parent_folder = source_root / "deformations" / deformation
        fields = load_deformation_fields(
            source_root / "deformations",
            deformation,
            source_config,
        )
    return generate_rigid_parameters(
        parent_folder,
        deformation,
        name,
        source_config,
        np.load(source_root / "empty_artery.npy", mmap_mode="r"),
        fields,
        config,
        rejection,
        seed=seed,
    )


def compose_artificial_sample(
    empty_artery: np.ndarray,
    layers: Iterable[PlaqueLayer],
    class_intensities: Mapping[AppearanceKind, float] = DEFAULT_CLASS_INTENSITIES,
    *,
    deformation_field: np.ndarray | None = None,
    rigid_parameters: np.ndarray | tuple[float, float, float] | None = None,
) -> ComposedSampleArrays:
    """Apply stored transforms and compose one artificial sample."""
    layers = tuple(layers)
    if deformation_field is not None:
        empty_artery = np.rint(
            apply_deformation(empty_artery, deformation_field, method="nearest")
        ).astype(np.uint8)
        layers = tuple(
            PlaqueLayer(
                apply_deformation(layer.mask, deformation_field, method="nearest")
                > 0.5,
                layer.target_class,
                layer.appearance,
            )
            for layer in layers
        )
    label_maps = compose_label_maps(empty_artery, layers)
    image = create_grayscale_image_from_label_mask(
        label_maps.appearance_labels,
        class_intensities,
    )
    arrays = ComposedSampleArrays(
        image=image,
        target_labels=label_maps.target_labels,
        appearance_labels=label_maps.appearance_labels,
    )
    if rigid_parameters is None:
        return arrays
    return ComposedSampleArrays(
        image=apply_rigid(arrays.image, *rigid_parameters, method="linear"),
        target_labels=np.rint(
            apply_rigid(
                arrays.target_labels,
                *rigid_parameters,
                method="nearest",
            )
        ).astype(np.uint8),
        appearance_labels=np.rint(
            apply_rigid(
                arrays.appearance_labels,
                *rigid_parameters,
                method="nearest",
            )
        ).astype(np.uint8),
    )
