import json

import numpy as np
import pytest
import torch

from constraints.datatools.datasets import ComposedArtificialDataset, Recipe
from constraints.generators.factories import (
    create_layer_collection,
    create_rigid_collection,
)
from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    SavedLayer,
    power_layer_backup,
)
from constraints.generators.source import create_source
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    NoiseConfig,
    RigidConfig,
    SourceConfig,
)


def _create_source_with_layers(tmp_path):
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=2,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )
    create_source(root, config)
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0),
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.25),
        wall_depth_fraction=FloatRange.fixed(0.2),
        shape_power=FloatRange.fixed(0.5),
    )
    create_layer_collection(root, "blob", power_layer_backup(ranges, seed=3))
    return root


def test_dataset_composes_selected_real_plaque_collection(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    dataset = ComposedArtificialDataset(root, layers=(SavedLayer("blob"),))

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (1, 65, 65)
    assert sample["image"].dtype == torch.float32
    assert sample["target_labels"].shape == (65, 65)
    assert sample["target_labels"].dtype == torch.int64
    assert torch.any(sample["target_labels"] == ArteryClass.PLAQUE)


def test_fake_plaque_changes_target_but_keeps_plaque_appearance(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    ranges = PowerPlaqueSamplingRanges(
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.25),
        wall_depth_fraction=FloatRange.fixed(0.2),
    )
    create_layer_collection(
        root,
        "fake",
        power_layer_backup(
            ranges,
            seed=3,
            target_class=ArteryClass.LUMEN,
            appearance=AppearanceKind.PLAQUE,
        ),
    )
    dataset = ComposedArtificialDataset(
        root,
        layers=(SavedLayer("fake"),),
    )
    labels = np.load(root / "layers" / "fake" / "labels.npy")

    sample = dataset[0]
    fake_pixels = torch.from_numpy(labels[0] >= 0)

    assert torch.all(sample["target_labels"][fake_pixels] == ArteryClass.LUMEN)
    assert torch.all(sample["image"][0, fake_pixels] == 1.0)


def test_dataset_uses_grayscale_values_stored_in_layer(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    recipe = Recipe(layers=(SavedLayer("blob"),))
    labels = np.load(root / "layers" / "blob" / "labels.npy")

    dataset = ComposedArtificialDataset.from_recipe(root, recipe)
    sample = dataset[0]

    assert torch.all(sample["image"][0, torch.from_numpy(labels[0] >= 0)] == 1.0)

    identity = dataset.sdf_cache_identity()
    source_id = json.loads((root / "manifest.json").read_text())["dataset_id"]
    assert identity.source_dataset_id == source_id


def test_dataset_applies_selected_deformation_before_composition(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    fields = np.zeros((2, 2, 65, 65), dtype=np.float32)
    fields[:, 1] = 2
    preset = root / "deformations" / "shift-left"
    preset.mkdir()
    np.save(preset / "fields.npy", fields)
    baseline = ComposedArtificialDataset(root, layers=(SavedLayer("blob"),))[0]
    dataset = ComposedArtificialDataset(
        root,
        layers=(SavedLayer("blob"),),
        deformation="shift-left",
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["target_labels"][:, :-2],
        baseline["target_labels"][:, 2:],
    )
    torch.testing.assert_close(sample["transform"], torch.from_numpy(fields[0]))


def test_dataset_applies_rigid_after_composition(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    create_rigid_collection(
        root,
        "shift-right",
        RigidConfig(
            angle=FloatRange.fixed(0),
            dx=FloatRange.fixed(2),
            dy=FloatRange.fixed(0),
        ),
        seed=5,
    )
    baseline = ComposedArtificialDataset(root, layers=(SavedLayer("blob"),))[0]
    recipe = Recipe(
        layers=(SavedLayer("blob"),),
        rigid="shift-right",
    )
    dataset = ComposedArtificialDataset.from_recipe(
        root,
        recipe,
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["target_labels"][:, 2:],
        baseline["target_labels"][:, :-2],
    )
    torch.testing.assert_close(sample["rigid"], torch.tensor([0.0, 2.0, 0.0]))
    assert dataset.recipe == recipe


def test_recipe_rejects_duplicate_layer_collections() -> None:
    with pytest.raises(ValueError, match="cannot contain.*twice"):
        Recipe(
            layers=(
                SavedLayer("blob"),
                SavedLayer("blob"),
            )
        )


def test_dataset_applies_deterministic_noise_from_recipe(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    recipe = Recipe(
        layers=(SavedLayer("blob"),),
        noise=NoiseConfig(speckle_std=0.2, seed=17),
    )
    first_dataset = ComposedArtificialDataset.from_recipe(root, recipe)
    second_dataset = ComposedArtificialDataset.from_recipe(root, recipe)

    first = first_dataset[0]
    repeated = first_dataset[0]
    recreated = second_dataset[0]
    other_sample = first_dataset[1]

    torch.testing.assert_close(first["image"], repeated["image"], rtol=0, atol=0)
    torch.testing.assert_close(first["image"], recreated["image"], rtol=0, atol=0)
    assert not torch.equal(first["image"], other_sample["image"])
    torch.testing.assert_close(
        first["target_labels"],
        ComposedArtificialDataset(root, layers=(SavedLayer("blob"),))[0][
            "target_labels"
        ],
    )


def test_dataset_accepts_noise_config_directly_and_uses_source_index(tmp_path) -> None:
    root = _create_source_with_layers(tmp_path)
    noise = NoiseConfig(speckle_std=0.1, seed=9)
    full = ComposedArtificialDataset(root, noise=noise)
    subset = ComposedArtificialDataset(root, noise=noise, sample_list=[1])

    torch.testing.assert_close(full[1]["image"], subset[0]["image"], rtol=0, atol=0)
