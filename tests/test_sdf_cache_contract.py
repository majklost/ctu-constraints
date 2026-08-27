import json

import numpy as np
import pytest
import torch

import constraints.generators.sdf_cache as sdf_cache_module
from constraints.generators.recipes import Recipe
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.sdf_cache import (
    SDFCacheConfig,
    SDFCacheIdentity,
    create_sdf_cache,
)
from constraints.generators.types import AppearanceKind, ArteryClass, SavedPlaque


def _identity(recipe: Recipe) -> SDFCacheIdentity:
    return SDFCacheIdentity.from_recipe("source-id", recipe)


def test_pre_rigid_sdf_identity_ignores_image_and_rigid_choices() -> None:
    target = SavedPlaque("artifact", target_class=ArteryClass.LUMEN)
    baseline = Recipe(plaques=(target,))
    image_variant = Recipe(
        plaques=(
            SavedPlaque(
                "artifact",
                target_class=ArteryClass.LUMEN,
                appearance=AppearanceKind.SHADOW,
            ),
        ),
        rigid="large",
        class_intensities={
            **DEFAULT_CLASS_INTENSITIES,
            AppearanceKind.SHADOW: 0.01,
        },
    )

    assert _identity(baseline).digest == _identity(image_variant).digest


def test_pre_rigid_sdf_identity_tracks_target_and_deformation_choices() -> None:
    baseline = _identity(Recipe(plaques=(SavedPlaque("blob"),)))
    different_target = _identity(
        Recipe(plaques=(SavedPlaque("blob", target_class=ArteryClass.LUMEN),))
    )
    different_deformation = _identity(
        Recipe(plaques=(SavedPlaque("blob"),), deformation="smooth")
    )

    assert baseline.digest != different_target.digest
    assert baseline.digest != different_deformation.digest


def test_sdf_configuration_and_directory_are_part_of_the_contract(tmp_path) -> None:
    recipe = Recipe(plaques=(SavedPlaque("blob"),))
    default = _identity(recipe)
    reordered = SDFCacheIdentity.from_recipe(
        "source-id",
        recipe,
        SDFCacheConfig(
            foreground_classes=(
                ArteryClass.PLAQUE,
                ArteryClass.LUMEN,
                ArteryClass.BOUNDARY,
            )
        ),
    )

    assert len(default.digest) == 64
    assert default.digest != reordered.digest
    assert default.cache_directory(tmp_path) == (
        tmp_path / "derived" / f"sdf-v1-{default.digest}"
    )


@pytest.mark.parametrize(
    ("mode", "selected_name"),
    (("scipy", "scipy"), ("kornia", "kornia")),
)
def test_cache_generation_dispatches_to_shared_sdf_utility(
    tmp_path,
    monkeypatch,
    mode,
    selected_name,
) -> None:
    calls = []

    def scipy(values):
        calls.append("scipy")
        return torch.ones_like(values, dtype=torch.float32)

    def kornia(values, *, device):
        calls.append("kornia")
        assert device == "cpu"
        return torch.ones_like(values, dtype=torch.float32) * 2

    monkeypatch.setattr(sdf_cache_module, "signed_distance_scipy", scipy)
    monkeypatch.setattr(sdf_cache_module, "signed_distance_kornia", kornia)
    recipe = Recipe()

    class Dataset:
        root = tmp_path
        labels = np.zeros((2, 7, 7), dtype=np.uint8)

        def __init__(self):
            self.recipe = recipe

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {"target_labels": torch.from_numpy(self.labels[index])}

        def sdf_cache_identity(self, config=None):
            return SDFCacheIdentity.from_recipe("source", self.recipe, config)

    config = SDFCacheConfig(mode=mode)
    array_path, manifest_path = create_sdf_cache(
        Dataset(),
        config,
        batch_size=2,
        device="cpu",
    )

    assert calls == [selected_name]
    expected_value = 1 if mode == "scipy" else 2
    np.testing.assert_array_equal(
        np.load(array_path),
        np.full((2, 3, 7, 7), expected_value, dtype=np.float32),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["identity"]["sdf"]["mode"] == mode
