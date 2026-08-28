"""Convenient orchestration used by generation scripts and datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from constraints.devices import DeviceSelection

from .artifact_metadata import write_artifact_metadata
from .composition import compose_layers
from .deformation import (
    apply_deformation,
    generate_deformation_fields,
    load_deformation_fields,
    sample_valid_deformation,
)
from .layer_generators import (
    TRANSPARENT_LABEL,
    LayerPatch,
    create_empty_artery,
    materialize_layer_collection,
)
from .noise import apply_speckle_noise
from .recipe_backups import DeformationBackup, LayerBackup, RigidBackup
from .rigid import apply_rigid, generate_rigid_parameters, sample_valid_rigid
from .source import load_source_config
from .types import (
    DeformationConfig,
    DeformationRejectionConfig,
    EmptyArteryConfig,
    NoiseConfig,
    RigidConfig,
    RigidRejectionConfig,
    SourceConfig,
)
from .validation import DeformationValidationResult

if TYPE_CHECKING:
    from .recipes import Recipe


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
    deformation_field: NDArray[np.float32] | None
    deformation_validation: DeformationValidationResult | None
    rigid_parameters: NDArray[np.float32] | None = None


def preview_artificial_sample(
    artery_config: EmptyArteryConfig | Recipe | None = None,
    layers: Iterable[LayerPatch] = (),
    deformation_config: DeformationConfig | None = None,
    deformation_rejection: DeformationRejectionConfig | None = None,
    rigid_config: RigidConfig | None = None,
    rigid_rejection: RigidRejectionConfig | None = None,
    *,
    seed: int | None = None,
    sample_index: int = 0,
    deformation_device: DeviceSelection = "auto",
    source_root: Path | None = None,
    recipe: Recipe | None = None,
    noise_config: NoiseConfig | None = None,
) -> PreviewArtificialSample:
    """Preview a low-level composition or a complete recipe."""
    from .recipes import Recipe as RecipeType

    if recipe is not None:
        if artery_config is not None:
            raise ValueError("pass either artery_config or recipe, not both")
        artery_config = recipe
    if isinstance(artery_config, RecipeType):
        from .recipe_preview import preview_recipe_sample

        return preview_recipe_sample(
            artery_config,
            source_root=source_root,
            sample_index=sample_index,
            deformation_device=deformation_device,
        )
    if not isinstance(artery_config, EmptyArteryConfig):
        raise TypeError("artery_config must be an EmptyArteryConfig or Recipe")
    if seed is None:
        raise ValueError("seed is required for low-level previews")
    empty_artery = create_empty_artery(artery_config)
    layers = tuple(layers)

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
        layers,
        deformation_field=None if deformation is None else deformation.field,
        rigid_parameters=None if rigid is None else rigid.parameters,
        noise_config=noise_config,
        sample_index=sample_index,
    )
    return PreviewArtificialSample(
        image=arrays.image,
        target_labels=arrays.target_labels,
        deformation_field=None if deformation is None else deformation.field,
        deformation_validation=(
            None if deformation is None else deformation.validation
        ),
        rigid_parameters=None if rigid is None else rigid.parameters,
    )


def get_source_config(source_root: Path) -> SourceConfig:
    """Return the validated configuration of an existing source dataset."""
    return load_source_config(source_root)


def create_layer_collection(
    source_root: Path,
    name: str,
    backup: LayerBackup,
) -> Path:
    """Materialize one named layer using its registered resolver."""
    source_root = Path(source_root)
    if not name or Path(name).name != name:
        raise ValueError("layer name must be a filename component")
    config = get_source_config(source_root)
    layer_root = source_root / "layers"
    metadata_path = layer_root / f"{name}.manifest.json"
    artifact_paths = (layer_root / name, metadata_path)
    if any(path.exists() for path in artifact_paths):
        raise FileExistsError(f"layer collection already exists: {name}")
    write_artifact_metadata(
        metadata_path,
        kind="layer-collection",
        name=name,
        definition=backup.to_dict(),
        status="preparing",
    )
    result = materialize_layer_collection(source_root, name, config, backup)
    write_artifact_metadata(
        metadata_path,
        kind="layer-collection",
        name=name,
        definition=backup.to_dict(),
        status="complete",
    )
    return result


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
    backup = DeformationBackup(
        config=config,
        rejection=(DeformationRejectionConfig() if rejection is None else rejection),
        seed=seed,
    )
    folder = source_root / "deformations"
    metadata_path = folder / f"{name}.manifest.json"
    if (folder / name).exists() or metadata_path.exists():
        raise FileExistsError(f"deformation collection already exists: {name}")
    write_artifact_metadata(
        metadata_path,
        kind="deformation-collection",
        name=name,
        definition=backup.to_dict(),
        status="preparing",
    )
    result = generate_deformation_fields(
        source_root / "deformations",
        name,
        source_config,
        np.load(source_root / "empty_artery.npy", mmap_mode="r"),
        config,
        rejection,
        seed=seed,
        device=device,
    )
    write_artifact_metadata(
        metadata_path,
        kind="deformation-collection",
        name=name,
        definition=backup.to_dict(),
        status="complete",
    )
    return result


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
    backup = RigidBackup(
        config=config,
        rejection=RigidRejectionConfig() if rejection is None else rejection,
        seed=seed,
    )
    rigid_folder = parent_folder / "rigid"
    metadata_path = rigid_folder / f"{name}.manifest.json"
    if any(
        path.exists()
        for path in (
            rigid_folder / f"{name}.npy",
            rigid_folder / f"{name}.json",
            metadata_path,
        )
    ):
        raise FileExistsError(f"rigid collection already exists: {name}")
    write_artifact_metadata(
        metadata_path,
        kind="rigid-collection",
        name=name,
        definition=backup.to_dict(),
        status="preparing",
    )
    result = generate_rigid_parameters(
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
    write_artifact_metadata(
        metadata_path,
        kind="rigid-collection",
        name=name,
        definition=backup.to_dict(),
        status="complete",
    )
    return result


def compose_artificial_sample(
    empty_artery: np.ndarray,
    layers: Iterable[LayerPatch],
    *,
    deformation_field: np.ndarray | None = None,
    rigid_parameters: np.ndarray | tuple[float, float, float] | None = None,
    noise_config: NoiseConfig | None = None,
    sample_index: int = 0,
) -> ComposedSampleArrays:
    """Apply stored transforms and compose one artificial sample."""
    layers = tuple(layers)
    if deformation_field is not None:
        empty_artery = np.rint(
            apply_deformation(empty_artery, deformation_field, method="nearest")
        ).astype(np.uint8)
        layers = tuple(_deform_layer(layer, deformation_field) for layer in layers)
    composed = compose_layers(empty_artery, layers)
    arrays = ComposedSampleArrays(
        image=composed.image,
        target_labels=composed.target_labels,
    )
    if rigid_parameters is not None:
        arrays = ComposedSampleArrays(
            image=apply_rigid(arrays.image, *rigid_parameters, method="linear"),
            target_labels=np.rint(
                apply_rigid(
                    arrays.target_labels,
                    *rigid_parameters,
                    method="nearest",
                )
            ).astype(np.uint8),
        )
    if noise_config is None:
        return arrays
    return ComposedSampleArrays(
        image=apply_speckle_noise(
            arrays.image,
            noise_config,
            sample_index=sample_index,
        ),
        target_labels=arrays.target_labels,
    )


def _deform_layer(layer: LayerPatch, field: np.ndarray) -> LayerPatch:
    label_pixels = layer.labels != TRANSPARENT_LABEL
    labels = apply_deformation(
        np.where(label_pixels, layer.labels, 0), field, method="nearest"
    ).astype(np.int8)
    warped_label_pixels = apply_deformation(label_pixels, field, method="nearest") > 0.5
    labels[~warped_label_pixels] = TRANSPARENT_LABEL

    image_pixels = ~np.isnan(layer.image)
    image = apply_deformation(
        np.where(image_pixels, layer.image, 0.0), field, method="linear"
    ).astype(np.float32)
    warped_image_pixels = apply_deformation(image_pixels, field, method="nearest") > 0.5
    image[~warped_image_pixels] = np.nan
    return LayerPatch(labels, image)
