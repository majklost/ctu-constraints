import json

import numpy as np
import torch

import constraints.generators.deformation as deformation_module
from constraints.generators.deformation import (
    apply_deformation,
    sample_valid_deformation,
)
from constraints.generators.factories import create_deformation_collection
from constraints.generators.source import create_source
from constraints.generators.types import (
    DeformationConfig,
    EmptyArteryConfig,
    SourceConfig,
)


def test_deformation_collection_is_mmap_friendly_and_reproducible(tmp_path) -> None:
    root = tmp_path / "source"
    create_source(
        root,
        SourceConfig(
            num_elements=3,
            image_size=(33, 33),
            empty_artery=EmptyArteryConfig(10, 3),
        ),
    )
    config = DeformationConfig(
        scales=8,
        magnitude=1,
        integrations=2,
        fractal_mode="upsample",
    )

    first_path, first_config_path = create_deformation_collection(
        root, "small", config, seed=8
    )
    second_path, _ = create_deformation_collection(
        root, "small-copy", config, seed=8
    )

    first = np.load(first_path, mmap_mode="r")
    second = np.load(second_path, mmap_mode="r")
    assert isinstance(first, np.memmap)
    assert first.shape == (3, 2, 33, 33)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)

    metadata = json.loads(first_config_path.read_text())
    assert metadata["array"]["channel_order"] == ["dy", "dx"]
    assert metadata["config"] == config.to_dict()
    assert metadata["generation_device"] == "cpu"
    assert DeformationConfig.from_dict(metadata["config"]) == config
    assert first_path == root / "deformations" / "small" / "fields.npy"
    assert (root / "deformations" / "small" / "rigid").is_dir()


def test_deformation_generation_does_not_change_global_torch_rng(tmp_path) -> None:
    root = tmp_path / "source"
    create_source(
        root,
        SourceConfig(
            num_elements=1,
            image_size=(33, 33),
            empty_artery=EmptyArteryConfig(10, 3),
        ),
    )
    config = DeformationConfig(
        scales=8,
        magnitude=1,
        integrations=2,
        fractal_mode="upsample",
    )
    torch.manual_seed(19)
    expected = torch.rand(4)

    torch.manual_seed(19)
    create_deformation_collection(root, "small", config, seed=8)
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected)


def test_deformation_generation_retries_rejected_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "source"
    create_source(
        root,
        SourceConfig(
            num_elements=1,
            image_size=(33, 33),
            empty_artery=EmptyArteryConfig(10, 3),
        ),
    )
    folding = torch.zeros(1, 2, 33, 33)
    folding[0, 0] = -2 * torch.arange(33)[:, None]
    identity = torch.zeros_like(folding)
    candidates = iter((folding, identity))
    monkeypatch.setattr(
        deformation_module,
        "random_disp",
        lambda **_: next(candidates),
    )

    fields_path, config_path = create_deformation_collection(
        root,
        "retry",
        DeformationConfig(),
        seed=5,
    )

    np.testing.assert_array_equal(np.load(fields_path), identity.numpy())
    metadata = json.loads(config_path.read_text())
    assert metadata["diagnostics"]["rejected_candidate_count"] == 1
    assert metadata["diagnostics"]["attempts_per_sample"] == [2]


def test_apply_deformation_uses_stored_backward_sampling_convention() -> None:
    values = np.zeros((7, 7), dtype=np.float32)
    values[3, 3] = 1
    field = np.zeros((2, 7, 7), dtype=np.float32)
    field[1] = 1

    warped = apply_deformation(values, field, method="nearest")

    assert warped[3, 2] == 1
    assert warped.sum() == 1


def test_single_deformation_sample_matches_persisted_collection(tmp_path) -> None:
    root = tmp_path / "source"
    source_config = SourceConfig(
        num_elements=2,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )
    create_source(root, source_config)
    labels = np.load(root / "empty_artery.npy")
    config = DeformationConfig(
        scales=8,
        magnitude=1,
        integrations=2,
        fractal_mode="upsample",
    )

    direct = sample_valid_deformation(
        source_config,
        labels,
        config,
        seed=17,
        sample_index=1,
    )
    fields_path, _ = create_deformation_collection(
        root,
        "set",
        config,
        seed=17,
    )

    np.testing.assert_array_equal(direct.field, np.load(fields_path)[1])
    assert direct.attempts >= 1
    assert direct.validation.accepted
