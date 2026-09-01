import numpy as np
import torch

from constraints.datatools.label_schema import LabelSchema
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_with_wall,
    is_annular,
)

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen", "plaque"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.35, 0.95)],
)


def _valid_vessel_labels() -> torch.Tensor:
    labels = torch.zeros((32, 32), dtype=torch.long)
    labels[4:28, 4:28] = 1
    labels[12:20, 12:20] = 2
    return labels


def _violations(labels: torch.Tensor, **kwargs: object) -> tuple[bool, list[str]]:
    return does_violation_occur_with_wall(
        labels,
        LABEL_SCHEMA,
        check_wall_integrity=False,
        **kwargs,
    )


def test_diagonally_connected_background_is_not_enclosed() -> None:
    labels = _valid_vessel_labels()
    labels[6, 6] = 0
    labels[5, 5] = 0
    labels[4, 4] = 0

    occurred, details = _violations(labels)

    assert not occurred
    assert details == []


def test_small_enclosed_background_components_are_ignored() -> None:
    labels = _valid_vessel_labels()
    labels[7, 7:9] = 0

    occurred, details = _violations(
        labels,
        max_ignored_enclosed_background_area=2,
    )

    assert not occurred
    assert details == []


def test_visible_enclosed_background_component_is_a_violation() -> None:
    labels = _valid_vessel_labels()
    labels[7, 7:10] = 0

    occurred, details = _violations(
        labels,
        max_ignored_enclosed_background_area=2,
    )

    assert occurred
    assert len(details) == 1
    assert "area 3 px" in details[0]


def test_is_annular_returns_true_with_no_details_for_one_ring() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:22, 10:22] = False

    annular, details = is_annular(mask)

    assert annular
    assert details == []


def test_is_annular_explains_an_empty_mask() -> None:
    annular, details = is_annular(np.zeros((32, 32), dtype=bool))

    assert not annular
    assert details == ["Myocardium mask is empty."]


def test_is_annular_explains_a_mask_without_a_valid_hole() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True

    annular, details = is_annular(mask)

    assert not annular
    assert len(details) == 1
    assert "Open-ring/missing-hole violation" in details[0]
    assert "background has 1" in details[0]
    assert "expected 2" in details[0]


def test_is_annular_identifies_an_open_ring_from_background_topology() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:22, 10:22] = False
    mask[4:10, 16] = False

    annular, details = is_annular(mask)

    assert not annular
    assert len(details) == 1
    assert "Open-ring/missing-hole violation" in details[0]
    assert "background has 1" in details[0]


def test_is_annular_ignores_foreground_blobs_smaller_than_five_pixels() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:22, 10:22] = False
    mask[1, 1] = True

    annular, details = is_annular(mask)

    assert annular
    assert details == []


def test_is_annular_can_disable_small_foreground_blob_filtering() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:22, 10:22] = False
    mask[1, 1] = True

    annular, details = is_annular(mask, min_component_area=None)

    assert not annular
    assert len(details) == 1
    assert "found 2" in details[0]


def test_is_annular_keeps_foreground_blobs_of_exactly_five_pixels() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:22, 10:22] = False
    mask[1, 1:6] = True

    annular, details = is_annular(mask)

    assert not annular
    assert len(details) == 1
    assert "found 2" in details[0]
