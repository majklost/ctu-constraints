import json

import pytest

from constraints.generators.recipes import Recipe
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.types import AppearanceKind, ArteryClass, SavedPlaque


def test_recipe_json_round_trip_preserves_appearance_and_intensities(tmp_path) -> None:
    path = tmp_path / "experiment-recipe.json"
    recipe = Recipe(
        plaques=(
            SavedPlaque(
                "shadow",
                target_class=ArteryClass.LUMEN,
                appearance=AppearanceKind.SHADOW,
            ),
        ),
        deformation="smooth",
        rigid="small",
        class_intensities={
            **DEFAULT_CLASS_INTENSITIES,
            AppearanceKind.PLAQUE: 0.9,
            AppearanceKind.SHADOW: 0.05,
        },
    )

    recipe.save_json(path)
    loaded = Recipe.load_json(path)

    assert loaded == recipe
    assert loaded.class_intensities[AppearanceKind.SHADOW] == 0.05
    value = json.loads(path.read_text())
    assert value["format_version"] == 1
    assert value["plaques"][0]["appearance"] == "shadow"
    assert value["class_intensities"]["plaque"] == 0.9


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
