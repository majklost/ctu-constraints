import numpy as np
import torch

from constraints.datatools.datasets import ComposedArtificialDataset
from constraints.generators.factories import (
    create_plaque_collection,
    create_rigid_collection,
)
from constraints.generators.source import create_source
from constraints.generators.types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    PowerPlaqueSamplingRanges,
    RigidBounds,
    SourceConfig,
)


def _create_source_with_plaques(tmp_path):
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=2,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )
    create_source(root, config)
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0),
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.25),
        wall_depth_fraction=FloatRange.fixed(0.2),
        shape_power=FloatRange.fixed(0.5),
    )
    create_plaque_collection(root, "blob", ranges, seed=3)
    return root


def test_dataset_composes_selected_real_plaque_collection(tmp_path) -> None:
    root = _create_source_with_plaques(tmp_path)
    dataset = ComposedArtificialDataset(root, plaques=("blob",))

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (1, 65, 65)
    assert sample["image"].dtype == torch.float32
    assert sample["target_labels"].shape == (65, 65)
    assert sample["target_labels"].dtype == torch.int64
    assert torch.any(sample["target_labels"] == ArteryClass.PLAQUE)


def test_fake_plaque_changes_target_but_keeps_plaque_appearance(tmp_path) -> None:
    root = _create_source_with_plaques(tmp_path)
    dataset = ComposedArtificialDataset(
        root,
        fake_plaques={"blob": ArteryClass.LUMEN},
    )
    masks = np.load(root / "plaques" / "blob.npy")

    sample = dataset[0]
    fake_pixels = torch.from_numpy(masks[0])

    assert torch.all(sample["target_labels"][fake_pixels] == ArteryClass.LUMEN)
    assert torch.all(sample["image"][0, fake_pixels] == 1.0)


def test_dataset_applies_selected_deformation_before_composition(tmp_path) -> None:
    root = _create_source_with_plaques(tmp_path)
    fields = np.zeros((2, 2, 65, 65), dtype=np.float32)
    fields[:, 1] = 2
    preset = root / "deformations" / "shift-left"
    preset.mkdir()
    np.save(preset / "fields.npy", fields)
    baseline = ComposedArtificialDataset(root, plaques=("blob",))[0]
    dataset = ComposedArtificialDataset(
        root,
        plaques=("blob",),
        deformation="shift-left",
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["target_labels"][:, :-2],
        baseline["target_labels"][:, 2:],
    )
    torch.testing.assert_close(sample["transform"], torch.from_numpy(fields[0]))


def test_dataset_applies_rigid_after_composition(tmp_path) -> None:
    root = _create_source_with_plaques(tmp_path)
    create_rigid_collection(
        root,
        "shift-right",
        RigidBounds(
            angle=FloatRange.fixed(0),
            dx=FloatRange.fixed(2),
            dy=FloatRange.fixed(0),
        ),
        seed=5,
    )
    baseline = ComposedArtificialDataset(root, plaques=("blob",))[0]
    dataset = ComposedArtificialDataset(
        root,
        plaques=("blob",),
        rigid="shift-right",
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["target_labels"][:, 2:],
        baseline["target_labels"][:, :-2],
    )
    torch.testing.assert_close(sample["rigid"], torch.tensor([0.0, 2.0, 0.0]))
