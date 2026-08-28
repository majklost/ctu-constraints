import json
from pathlib import Path

import pytest

from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    SavedLayer,
    power_layer_backup,
)
from constraints.generators.recipe_backups import (
    DeformationBackup,
    RigidBackup,
    SavedDeformation,
    SavedRigid,
)
from constraints.generators.recipes import Recipe
from constraints.generators.sdf_cache import SDFCacheConfig
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    NoiseConfig,
    RigidConfig,
)


def test_recipe_json_round_trip_preserves_layer_backup(tmp_path) -> None:
    path = tmp_path / "experiment-recipe.json"
    recipe = Recipe(
        source="artificial/example",
        layers=(
            SavedLayer(
                "shadow",
                backup=power_layer_backup(
                    PowerPlaqueSamplingRanges(),
                    seed=3,
                    target_class=ArteryClass.LUMEN,
                    appearance=AppearanceKind.SHADOW,
                ),
            ),
        ),
        deformation=SavedDeformation("smooth", DeformationBackup(seed=4)),
        rigid=SavedRigid("small", RigidBackup(config=RigidConfig(), seed=5)),
        noise=NoiseConfig(speckle_std=0.15, speckle_mode="additive", seed=42),
        sdf_cache=SDFCacheConfig(mode="scipy"),
    )

    recipe.save_json(path)
    loaded = Recipe.load_json(path)

    assert loaded == recipe
    value = json.loads(path.read_text())
    assert value["format_version"] == 5
    assert value["source"] == "artificial/example"
    assert value["layers"][0]["backup"]["params"]["seed"] == 3
    assert value["deformation"]["backup"]["seed"] == 4
    assert value["rigid"]["backup"]["seed"] == 5
    assert value["sdf_cache"]["mode"] == "scipy"
    assert value["layers"][0]["backup"]["params"]["appearance"] == "shadow"
    assert value["layers"][0]["backup"]["params"]["target_class"] == "lumen"
    assert value["noise"]["speckle_std"] == 0.15
    assert value["noise"]["speckle_mode"] == "additive"
    assert value["noise"]["seed"] == 42


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
        layers=(
            SavedLayer(backup=power_layer_backup(PowerPlaqueSamplingRanges(), seed=1)),
        ),
        deformation=SavedDeformation(backup=DeformationBackup(seed=2)),
        rigid=SavedRigid(backup=RigidBackup(seed=3)),
    )

    resolved = recipe.with_names(
        layers={0: "blob"}, deformation="smooth", rigid="small"
    )

    resolved.require_resolved()
    assert resolved.layers[0].name == "blob"
    assert resolved.deformation_name == "smooth"
    assert resolved.rigid_name == "small"


@pytest.mark.parametrize(
    "filename",
    [
        "default.json",
        "tworeal_fake_similar.json",
        "tworeal_fake_similar_offset_minus3.json",
    ],
)
def test_checked_in_recipes_are_valid(filename) -> None:
    root = Path(__file__).parents[1]

    recipe = Recipe.load_json(root / "recipes/artificial" / filename)

    recipe.require_resolved()
