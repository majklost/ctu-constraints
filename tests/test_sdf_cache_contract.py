from constraints.generators.recipes import Recipe
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.sdf_cache import SDFCacheConfig, SDFCacheIdentity
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
