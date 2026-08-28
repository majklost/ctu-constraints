import json

import numpy as np
import pytest

from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    sample_power_plaque_mask,
)
from constraints.generators.source import (
    create_source,
    load_source_config,
)
from constraints.generators.types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    SourceConfig,
)


def test_create_source_initializes_only_root_artifacts(tmp_path) -> None:
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=3,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )

    create_source(root, config)

    labels = np.load(root / "empty_artery.npy")
    assert labels.shape == (65, 65)
    assert labels.dtype == np.uint8
    assert labels[32, 32] == ArteryClass.LUMEN
    assert labels[32, 54] == ArteryClass.BOUNDARY
    assert labels[0, 0] == ArteryClass.BACKGROUND
    assert (root / "layers").is_dir()
    assert (root / "deformations").is_dir()
    assert (root / "rigid").is_dir()

    manifest = json.loads((root / "manifest.json").read_text())
    assert load_source_config(root) == config
    assert "source_config" not in manifest
    assert manifest["artifacts"]["source_config"] == {
        "relative_path": "source_config.json"
    }
    assert manifest["artifacts"]["empty_artery"]["shape"] == [65, 65]


def test_create_source_refuses_to_replace_an_existing_dataset(tmp_path) -> None:
    root = tmp_path / "source"
    root.mkdir()

    with pytest.raises(FileExistsError):
        create_source(root, SourceConfig(num_elements=1))


def test_plaque_sample_seed_is_independent_of_collection_length() -> None:
    short = SourceConfig(
        num_elements=1,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )
    long = SourceConfig(
        num_elements=2,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )

    np.testing.assert_array_equal(
        sample_power_plaque_mask(short, seed=11).mask,
        sample_power_plaque_mask(long, seed=11).mask,
    )


def test_plaque_sample_can_use_an_inner_lumen_radius() -> None:
    config = SourceConfig(
        num_elements=1,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0),
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.2),
        wall_depth_fraction=FloatRange.fixed(0.1),
        shape_power=FloatRange.fixed(2),
    )

    direct = sample_power_plaque_mask(
        config,
        ranges,
        seed=19,
        lumen_radius_px=15,
    )
    assert direct.mask.any()
    assert direct.parameters[0].inward_depth_px == 3
