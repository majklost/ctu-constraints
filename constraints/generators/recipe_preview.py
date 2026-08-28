"""Resolve one recipe sample from dynamic backups and stored collections."""

from pathlib import Path

import numpy as np

from constraints.devices import DeviceSelection

from .deformation import (
    apply_deformation,
    load_deformation_fields,
    sample_valid_deformation,
)
from .factories import PreviewArtificialSample, compose_artificial_sample
from .layer_generators import (
    LayerPatch,
    create_empty_artery,
    load_layer_collection,
    resolve_layer_patch,
)
from .recipes import Recipe
from .rigid import load_rigid_parameters, sample_valid_rigid
from .source import load_source_config


def preview_recipe_sample(
    recipe: Recipe,
    *,
    source_root: Path | None = None,
    sample_index: int = 0,
    deformation_device: DeviceSelection = "auto",
) -> PreviewArtificialSample:
    """Preview a recipe without materializing artifacts with backups."""
    root = recipe.resolve_source_root(source_root)
    config = load_source_config(root)
    if sample_index < 0 or sample_index >= config.num_elements:
        raise IndexError("sample_index outside source dataset")
    empty_artery = create_empty_artery(config.empty_artery)

    layers = tuple(
        _layer_patch(root, config, saved, sample_index) for saved in recipe.layers
    )

    deformation_field = None
    deformation_validation = None
    if recipe.deformation is not None:
        if recipe.deformation.backup is not None:
            backup = recipe.deformation.backup
            sample = sample_valid_deformation(
                empty_artery,
                backup.config,
                backup.rejection,
                seed=backup.seed,
                sample_index=sample_index,
                device=deformation_device,
            )
            deformation_field = sample.field
            deformation_validation = sample.validation
        else:
            if recipe.deformation.name is None:
                raise ValueError("deformation has neither a name nor backup")
            deformation_field = load_deformation_fields(
                root / "deformations", recipe.deformation.name, config
            )[sample_index]

    rigid_parameters = None
    if recipe.rigid is not None:
        if recipe.rigid.backup is not None:
            rigid_source = empty_artery
            if deformation_field is not None:
                rigid_source = np.rint(
                    apply_deformation(empty_artery, deformation_field, method="nearest")
                ).astype(np.uint8)
            backup = recipe.rigid.backup
            rigid_parameters = sample_valid_rigid(
                rigid_source,
                backup.config,
                backup.rejection,
                seed=backup.seed,
                sample_index=sample_index,
            ).parameters
        else:
            if recipe.rigid.name is None:
                raise ValueError("rigid transform has neither a name nor backup")
            parent = (
                root
                if recipe.deformation_name is None
                else root / "deformations" / recipe.deformation_name
            )
            rigid_parameters = load_rigid_parameters(parent, recipe.rigid.name, config)[
                sample_index
            ]

    arrays = compose_artificial_sample(
        empty_artery,
        layers,
        deformation_field=deformation_field,
        rigid_parameters=rigid_parameters,
        noise_config=recipe.noise,
        sample_index=sample_index,
    )
    return PreviewArtificialSample(
        image=arrays.image,
        target_labels=arrays.target_labels,
        deformation_field=deformation_field,
        deformation_validation=deformation_validation,
        rigid_parameters=rigid_parameters,
    )


def _layer_patch(root, config, layer, sample_index: int) -> LayerPatch:
    if layer.backup is not None:
        return resolve_layer_patch(
            layer.backup,
            root,
            config,
            sample_index,
        )
    if layer.name is None:
        raise ValueError("layer has neither a name nor backup")
    collection = load_layer_collection(root, layer.name, config)
    return LayerPatch(collection.labels[sample_index], collection.image[sample_index])
