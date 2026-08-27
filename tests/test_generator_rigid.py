import json

import numpy as np
import pytest

from constraints.generators.factories import create_rigid_collection
from constraints.generators.rigid import apply_rigid
from constraints.generators.source import create_source
from constraints.generators.types import (
    EmptyArteryConfig,
    FloatRange,
    RigidConfig,
    RigidRejectionConfig,
    SourceConfig,
)


def _source_root(tmp_path):
    root = tmp_path / "source"
    create_source(
        root,
        SourceConfig(
            num_elements=2,
            empty_artery=EmptyArteryConfig(10, 3, (33, 33)),
        ),
    )
    return root


def test_apply_rigid_uses_forward_pixel_translation() -> None:
    values = np.zeros((7, 7), dtype=np.float32)
    values[3, 3] = 1

    warped = apply_rigid(values, 0, 1, 2, method="nearest")

    assert warped[5, 4] == 1
    assert warped.sum() == 1


def test_positive_rigid_angle_rotates_content_counter_clockwise() -> None:
    values = np.zeros((7, 7), dtype=np.float32)
    values[3, 4] = 1

    warped = apply_rigid(values, np.pi / 2, 0, 0, method="nearest")

    assert warped[2, 3] == 1
    assert warped.sum() == 1


def test_rigid_collection_can_live_at_source_level(tmp_path) -> None:
    root = _source_root(tmp_path)
    config = RigidConfig(
        angle=FloatRange.fixed(0),
        dx=FloatRange.fixed(1),
        dy=FloatRange.fixed(0),
    )

    parameters_path, config_path = create_rigid_collection(
        root,
        "small",
        config,
        seed=4,
    )

    parameters = np.load(parameters_path, mmap_mode="r")
    assert isinstance(parameters, np.memmap)
    assert parameters.shape == (2, 3)
    assert parameters.dtype == np.float32
    np.testing.assert_array_equal(parameters, [[0, 1, 0], [0, 1, 0]])
    assert parameters_path == root / "rigid/small.npy"
    metadata = json.loads(config_path.read_text())
    assert metadata["parent_deformation"] is None


def test_rigid_collection_can_depend_on_a_deformation(tmp_path) -> None:
    root = _source_root(tmp_path)
    deformation = root / "deformations" / "gentle"
    (deformation / "rigid").mkdir(parents=True)
    np.save(
        deformation / "fields.npy",
        np.zeros((2, 2, 33, 33), dtype=np.float32),
    )

    parameters_path, config_path = create_rigid_collection(
        root,
        "small",
        RigidConfig(
            angle=FloatRange.fixed(0),
            dx=FloatRange.fixed(0),
            dy=FloatRange.fixed(0),
        ),
        deformation="gentle",
        seed=7,
    )

    assert parameters_path == deformation / "rigid/small.npy"
    metadata = json.loads(config_path.read_text())
    assert metadata["parent_deformation"] == "gentle"


def test_rigid_collection_rejects_clipped_foreground(tmp_path) -> None:
    root = _source_root(tmp_path)
    config = RigidConfig(
        angle=FloatRange.fixed(0),
        dx=FloatRange.fixed(40),
        dy=FloatRange.fixed(0),
    )

    with pytest.raises(RuntimeError, match="failed to sample valid rigid"):
        create_rigid_collection(
            root,
            "invalid",
            config,
            RigidRejectionConfig(max_attempts=2),
            seed=4,
        )

    rigid_folder = root / "rigid"
    assert not (rigid_folder / "invalid.npy").exists()
    assert not (rigid_folder / "invalid.json").exists()
