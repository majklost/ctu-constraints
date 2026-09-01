import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from constraints.datatools.datasets import ACDCSliceMyocardiumOnlyDataset


def _write_slice(path, shape: tuple[int, int]) -> None:
    image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[1:-1, 1:-1] = 2
    with h5py.File(path, "w") as file:
        file.create_dataset("image", data=image)
        file.create_dataset("label", data=label)


def test_acdc_dataset_returns_unet_ready_sample(tmp_path) -> None:
    _write_slice(tmp_path / "first.h5", (8, 10))
    dataset = ACDCSliceMyocardiumOnlyDataset(
        tmp_path,
        pl.DataFrame({"path": ["first.h5"]}),
        image_size=(16, 16),
    )

    sample = dataset[0]

    assert sample["image"].shape == (1, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["target_labels"].shape == (16, 16)
    assert sample["target_labels"].dtype == torch.int64
    assert set(sample["target_labels"].unique().tolist()) == {0, 1}
    assert sample["sample_id"] == "first"


def test_acdc_resize_allows_batching_native_variable_sizes(tmp_path) -> None:
    _write_slice(tmp_path / "first.h5", (8, 10))
    _write_slice(tmp_path / "second.h5", (12, 7))
    dataset = ACDCSliceMyocardiumOnlyDataset(
        tmp_path,
        pl.DataFrame({"path": ["first.h5", "second.h5"]}),
        image_size=(16, 16),
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["image"].shape == (2, 1, 16, 16)
    assert batch["target_labels"].shape == (2, 16, 16)

