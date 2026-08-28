import json

import numpy as np
import pytest

from constraints.datatools.datasets import ComposedArtificialDataset
from constraints.generators.factories import preview_artificial_sample
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
from constraints.generators.source import create_source
from constraints.generators.types import (
    DeformationConfig,
    EmptyArteryConfig,
    FloatRange,
    RigidConfig,
    SourceConfig,
)


def _source(tmp_path):
    root = tmp_path / "source"
    create_source(
        root,
        SourceConfig(2, EmptyArteryConfig(20, 5, (65, 65))),
    )
    return root


def _layer_backup(*, inward: float = 0.2):
    return power_layer_backup(
        (
            PowerPlaqueSamplingRanges(
                angle_rad=FloatRange.fixed(0),
                angular_width_rad=FloatRange.fixed(0.5),
                inward_depth_fraction=FloatRange.fixed(inward),
                wall_depth_fraction=FloatRange.fixed(0.1),
            ),
        ),
        seed=7,
    )


def test_recipe_preview_and_ensure_use_the_same_backup(tmp_path) -> None:
    root = _source(tmp_path)
    recipe = Recipe(
        layers=(SavedLayer("blob", backup=_layer_backup()),),
        rigid=SavedRigid(
            "shift",
            RigidBackup(
                config=RigidConfig(
                    angle=FloatRange.fixed(0),
                    dx=FloatRange.fixed(1),
                    dy=FloatRange.fixed(0),
                ),
                seed=8,
            ),
        ),
    )

    preview = preview_artificial_sample(recipe=recipe, source_root=root, sample_index=1)
    report = recipe.ensure(root)
    stored = ComposedArtificialDataset.from_recipe(root, recipe)[1]

    np.testing.assert_array_equal(preview.target_labels, stored["target_labels"])
    assert report.created == ("layer 'blob'", "rigid 'shift'")
    assert (
        json.loads((root / "layers/blob.manifest.json").read_text())["status"]
        == "complete"
    )
    assert (root / "rigid/shift.npy").exists()


def test_preflight_fails_before_creating_any_artifact(tmp_path) -> None:
    root = _source(tmp_path)
    recipe = Recipe(
        layers=(
            SavedLayer("creatable", backup=_layer_backup()),
            SavedLayer("missing"),
        )
    )

    with pytest.raises(RuntimeError, match="missing.*has no backup"):
        recipe.ensure(root)

    assert not (root / "layers/creatable").exists()


def test_definition_mismatch_is_readable_and_overwritable(tmp_path) -> None:
    root = _source(tmp_path)
    original = Recipe(layers=(SavedLayer("blob", backup=_layer_backup(inward=0.2)),))
    original.ensure(root)
    changed = Recipe(layers=(SavedLayer("blob", backup=_layer_backup(inward=0.3)),))

    with pytest.raises(RuntimeError, match="inward_depth_fraction.minimum"):
        changed.ensure(root)

    report = changed.ensure(root, overwrite=True)

    assert report.replaced == ("layer 'blob'",)
    stored = json.loads((root / "layers/blob.manifest.json").read_text())
    assert stored["definition"]["params"]["ranges"][0]["sampling"][
        "inward_depth_fraction"
    ] == {
        "minimum": 0.3,
        "maximum": 0.3,
    }


def test_ensure_creates_and_reuses_requested_sdf_cache(tmp_path) -> None:
    root = _source(tmp_path)
    recipe = Recipe(
        layers=(SavedLayer("blob", backup=_layer_backup()),),
        sdf_cache=SDFCacheConfig(mode="scipy"),
    )

    first = recipe.ensure(root)
    second = recipe.ensure(root)

    assert first.sdf_cache is not None and first.sdf_cache.exists()
    assert second.sdf_cache == first.sdf_cache


def test_overwriting_deformation_recreates_its_selected_rigid(tmp_path) -> None:
    root = _source(tmp_path)
    rigid = SavedRigid("fixed", backup=RigidBackup(seed=9))
    original = Recipe(
        deformation=SavedDeformation(
            "warp", backup=DeformationBackup(DeformationConfig(magnitude=0), seed=8)
        ),
        rigid=rigid,
    )
    original.ensure(root, device="cpu")
    changed = Recipe(
        deformation=SavedDeformation(
            "warp", backup=DeformationBackup(DeformationConfig(magnitude=1), seed=8)
        ),
        rigid=rigid,
    )

    report = changed.ensure(root, overwrite=True, device="cpu")

    assert report.replaced == ("deformation 'warp'",)
    assert report.created == ("rigid 'fixed'",)
    assert (root / "deformations/warp/rigid/fixed.npy").exists()
