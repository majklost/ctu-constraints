import csv

import numpy as np

from scripts.prepare_composed_artificial_dataset import (
    _fake_plaque_range,
    _real_plaque_ranges,
    _validate_split,
    _write_split,
)


def test_preparation_script_matches_explored_plaque_ranges() -> None:
    first, second = _real_plaque_ranges()
    fake = _fake_plaque_range()

    assert first.angle_rad.minimum == -np.pi / 3
    assert first.angle_rad.maximum == -np.pi / 10
    assert second.angle_rad.minimum == np.pi / 10
    assert second.angle_rad.maximum == np.pi / 3
    assert first.angular_width_rad.minimum == np.pi / 5
    assert first.inward_depth_fraction.minimum == 0.2
    assert fake.inward_depth_fraction.minimum == 0.12
    assert fake.inward_depth_fraction.maximum == 0.15
    assert fake.shape_power.minimum == 2


def test_split_csv_contract(tmp_path) -> None:
    path = tmp_path / "trn_samples.csv"
    _write_split(path, np.array([4, 1, 8]))

    _validate_split(path, expected_size=3, num_samples=10)

    with path.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"sample_index": "4"},
            {"sample_index": "1"},
            {"sample_index": "8"},
        ]
