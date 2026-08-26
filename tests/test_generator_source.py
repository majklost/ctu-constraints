import json

import numpy as np
import pytest

from constraints.generators.source import (
    create_source,
    generate_plaque_masks_power,
    load_source_config,
    sample_power_plaque_mask,
)
from constraints.generators.types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    PowerPlaqueSamplingRanges,
    SourceConfig,
)


def test_create_source_initializes_only_root_artifacts(tmp_path) -> None:
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=3,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )

    create_source(root, config)

    labels = np.load(root / "empty_artery.npy")
    assert labels.shape == (65, 65)
    assert labels.dtype == np.uint8
    assert labels[32, 32] == ArteryClass.LUMEN
    assert labels[32, 54] == ArteryClass.BOUNDARY
    assert labels[0, 0] == ArteryClass.BACKGROUND
    assert (root / "plaques").is_dir()
    assert (root / "deformations").is_dir()

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


def test_generate_power_plaque_collection_writes_masks_and_parameters(
    tmp_path,
) -> None:
    config = SourceConfig(
        num_elements=3,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0),
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.25),
        wall_depth_fraction=FloatRange.fixed(0.2),
        shape_power=FloatRange.fixed(0.5),
    )

    generate_plaque_masks_power(tmp_path, "narrow", config, ranges, seed=7)

    masks = np.load(tmp_path / "narrow.npy")
    records = [
        json.loads(line)
        for line in (tmp_path / "narrow.jsonl").read_text().splitlines()
    ]
    assert masks.shape == (3, 65, 65)
    assert masks.dtype == np.bool_
    assert masks.any(axis=(1, 2)).all()
    assert [record["sample_index"] for record in records] == [0, 1, 2]
    assert records[0]["plaques"][0]["parameters"]["inward_depth_px"] == 5


def test_plaque_sample_seed_is_independent_of_collection_length(tmp_path) -> None:
    short = SourceConfig(
        num_elements=1,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )
    long = SourceConfig(
        num_elements=2,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )

    generate_plaque_masks_power(tmp_path / "short", "set", short, seed=11)
    generate_plaque_masks_power(tmp_path / "long", "set", long, seed=11)

    np.testing.assert_array_equal(
        np.load(tmp_path / "short" / "set.npy")[0],
        np.load(tmp_path / "long" / "set.npy")[0],
    )


def test_single_plaque_sample_matches_persisted_collection(tmp_path) -> None:
    config = SourceConfig(
        num_elements=3,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )
    direct = sample_power_plaque_mask(config, seed=13, sample_index=2)

    generate_plaque_masks_power(tmp_path, "set", config, seed=13)

    np.testing.assert_array_equal(direct.mask, np.load(tmp_path / "set.npy")[2])
    record = json.loads((tmp_path / "set.jsonl").read_text().splitlines()[2])
    assert record["sample_seed"] == direct.sample_seed
    assert record["plaques"][0]["parameters"]["angle_rad"] == (
        direct.parameters[0].angle_rad
    )
