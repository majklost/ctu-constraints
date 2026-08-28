import json
from pathlib import Path

import pytest

from constraints.generators.recipe_backups import (
    DeformationBackup,
    PowerPlaqueBackup,
    RigidBackup,
    SavedDeformation,
    SavedRigid,
)
from constraints.generators.recipes import Recipe
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.sdf_cache import SDFCacheConfig
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    NoiseConfig,
    PowerPlaqueSamplingRanges,
    RigidConfig,
    SavedPlaque,
)


def test_recipe_json_round_trip_preserves_appearance_and_intensities(tmp_path) -> None:
    path = tmp_path / "experiment-recipe.json"
    recipe = Recipe(
        source="artificial/example",
        plaques=(
            SavedPlaque(
                "shadow",
                target_class=ArteryClass.LUMEN,
                appearance=AppearanceKind.SHADOW,
                backup=PowerPlaqueBackup(ranges=(PowerPlaqueSamplingRanges(),), seed=3),
            ),
        ),
        deformation=SavedDeformation("smooth", DeformationBackup(seed=4)),
        rigid=SavedRigid("small", RigidBackup(config=RigidConfig(), seed=5)),
        class_intensities={
            **DEFAULT_CLASS_INTENSITIES,
            AppearanceKind.PLAQUE: 0.9,
            AppearanceKind.SHADOW: 0.05,
        },
        noise=NoiseConfig(speckle_std=0.15, speckle_mode="additive", seed=42),
        sdf_cache=SDFCacheConfig(mode="scipy"),
    )

    recipe.save_json(path)
    loaded = Recipe.load_json(path)

    assert loaded == recipe
    assert loaded.class_intensities[AppearanceKind.SHADOW] == 0.05
    value = json.loads(path.read_text())
    assert value["format_version"] == 3
    assert value["source"] == "artificial/example"
    assert value["plaques"][0]["backup"]["seed"] == 3
    assert value["deformation"]["backup"]["seed"] == 4
    assert value["rigid"]["backup"]["seed"] == 5
    assert value["sdf_cache"]["mode"] == "scipy"
    assert value["plaques"][0]["appearance"] == "shadow"
    assert value["class_intensities"]["plaque"] == 0.9
    assert value["noise"]["speckle_std"] == 0.15
    assert value["noise"]["speckle_mode"] == "additive"
    assert value["noise"]["seed"] == 42


def test_recipe_owns_an_immutable_copy_of_intensities() -> None:
    intensities = dict(DEFAULT_CLASS_INTENSITIES)
    recipe = Recipe(class_intensities=intensities)

    intensities[AppearanceKind.LUMEN] = 0.99

    assert recipe.class_intensities[AppearanceKind.LUMEN] == 0.25
    with pytest.raises(TypeError):
        recipe.class_intensities[AppearanceKind.LUMEN] = 0.1  # type: ignore[index]


def test_recipe_rejects_unknown_json_fields() -> None:
    value = Recipe().to_dict()
    value["future_field"] = True

    with pytest.raises(ValueError, match="Recipe fields"):
        Recipe.from_dict(value)


def test_recipe_rejects_old_format_versions() -> None:
    value = Recipe().to_dict()
    value["format_version"] = 2

    with pytest.raises(ValueError, match="format version"):
        Recipe.from_dict(value)


def test_recipe_can_assign_names_to_dynamic_artifacts() -> None:
    recipe = Recipe(
        plaques=(
            SavedPlaque(backup=PowerPlaqueBackup((PowerPlaqueSamplingRanges(),), 1)),
        ),
        deformation=SavedDeformation(backup=DeformationBackup(seed=2)),
        rigid=SavedRigid(backup=RigidBackup(seed=3)),
    )

    resolved = recipe.with_names(
        plaques={0: "blob"}, deformation="smooth", rigid="small"
    )

    resolved.require_resolved()
    assert resolved.plaques[0].name == "blob"
    assert resolved.deformation_name == "smooth"
    assert resolved.rigid_name == "small"


@pytest.mark.parametrize("filename", ["default.json", "tworeal_fake_similar.json"])
def test_checked_in_recipes_are_valid(filename) -> None:
    root = Path(__file__).parents[1]

    recipe = Recipe.load_json(root / "recipes/artificial" / filename)

    recipe.require_resolved()
