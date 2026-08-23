import numpy as np
import torch
from torch.utils.data import DataLoader

from constraints.datatools.datasets.artificial_dataset import CachedArtificialDataset
from constraints.datatools.datasets.types import TemplateAssets
from constraints.datatools.label_schema import LabelSchema
from constraints.datatools.template_refiners import IdentityTemplateRefiner
from constraints.datatools.template_sources import PerSampleTemplateSource

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen", "plaque"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.35, 0.95)],
)


def test_per_sample_template_source_converts_labels_to_semantic_channels():
    labels = torch.tensor([[[0, 1], [2, 3]], [[3, 2], [1, 0]]])
    source = PerSampleTemplateSource(TemplateAssets(), LABEL_SCHEMA)

    result = source({"template": labels})

    assert result.masks.shape == (2, 4, 2, 2)
    assert result.masks.dtype == torch.float32
    assert torch.equal(result.masks.argmax(dim=1), labels)
    assert IdentityTemplateRefiner()(result) is result


def test_artificial_dataset_returns_collatable_labels_and_template_sdf(tmp_path):
    sample_count, height, width = 2, 4, 5
    masks = np.zeros((sample_count, 3, height, width), dtype=np.float32)
    masks[:, 0, 1:3, 1:4] = 1.0
    template = masks[0].copy()
    images = np.zeros((sample_count, 1, height, width), dtype=np.float32)
    np.save(tmp_path / "img.npy", images)
    np.save(tmp_path / "mask.npy", masks)
    np.save(tmp_path / "sdf_kornia.npy", np.zeros_like(masks))
    np.save(tmp_path / "sdf_scipy.npy", np.zeros_like(masks))
    np.save(tmp_path / "template.npy", template)
    transforms = np.zeros((sample_count, 3, 3), dtype=np.float32)
    np.save(tmp_path / "transform.npy", transforms)

    dataset = CachedArtificialDataset(
        tmp_path, sdf_mode="scipy", return_template_sdf=True
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["target_labels"].shape == (2, height, width)
    assert batch["target_labels"].dtype == torch.long
    assert batch["template"].shape == (2, height, width)
    assert batch["template_sdf"].shape == (2, 3, height, width)


def test_artificial_dataset_filters_bad_indices_by_default(tmp_path):
    sample_count, height, width = 3, 2, 2
    masks = np.zeros((sample_count, 3, height, width), dtype=np.float32)
    images = np.arange(sample_count, dtype=np.float32).reshape(sample_count, 1, 1, 1)
    images = np.broadcast_to(images, (sample_count, 1, height, width)).copy()

    np.save(tmp_path / "img.npy", images)
    np.save(tmp_path / "mask.npy", masks)
    np.save(tmp_path / "sdf_kornia.npy", np.zeros_like(masks))
    np.save(tmp_path / "sdf_scipy.npy", np.zeros_like(masks))
    np.save(tmp_path / "template.npy", masks[0])
    np.save(tmp_path / "transform.npy", np.zeros((sample_count, 3, 3), np.float32))
    (tmp_path / "bad_indices.csv").write_text("index,violations\n1,invalid\n")

    filtered = CachedArtificialDataset(tmp_path)
    unfiltered = CachedArtificialDataset(tmp_path, bad_indices_fname=None)

    assert len(filtered) == 2
    assert [filtered[index]["sample_id"] for index in range(len(filtered))] == [
        "0_real_0_filtered",
        "2_real_1_filtered",
    ]
    assert len(unfiltered) == sample_count
