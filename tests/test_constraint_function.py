import torch

from constraints.datatools.label_schema import LabelSchema
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_with_wall,
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
