import numpy as np

from constraints.generators.types import DeformationRejectionConfig
from constraints.generators.validation import validate_deformation


def _source_labels() -> np.ndarray:
    y, x = np.ogrid[:33, :33]
    return (np.hypot(y - 16, x - 16) <= 10).astype(np.uint8)


def test_identity_deformation_passes_topology_and_margin_checks() -> None:
    result = validate_deformation(
        np.zeros((2, 33, 33), dtype=np.float32),
        _source_labels(),
        DeformationRejectionConfig(minimum_foreground_margin_px=1),
    )

    assert result.accepted
    assert result.minimum_jacobian == 1
    assert result.foreground_margin_px == 6


def test_deformation_with_fold_is_rejected() -> None:
    field = np.zeros((2, 33, 33), dtype=np.float32)
    field[0] = -2 * np.arange(33)[:, None]

    result = validate_deformation(
        field,
        _source_labels(),
        DeformationRejectionConfig(),
    )

    assert not result.accepted
    assert result.minimum_jacobian < 0


def test_deformation_clipping_foreground_is_rejected() -> None:
    field = np.zeros((2, 33, 33), dtype=np.float32)
    field[1] = 10

    result = validate_deformation(
        field,
        _source_labels(),
        DeformationRejectionConfig(minimum_foreground_margin_px=1),
    )

    assert not result.accepted
    assert result.minimum_jacobian == 1
    assert result.foreground_margin_px == 0
