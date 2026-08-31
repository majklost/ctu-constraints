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


def test_deformation_rejects_disappearing_three_pixel_wall() -> None:
    y, x = np.ogrid[:33, :33]
    radius = np.hypot(y - 16, x - 16)
    labels = np.zeros((33, 33), dtype=np.uint8)
    labels[radius <= 12] = 1
    labels[radius <= 8] = 2
    field = np.zeros((2, 33, 33), dtype=np.float32)
    # Backward sampling expands radial input distances and compresses the wall
    # in the output until nearest-neighbor rasterization opens a gap.
    field[0] = 2 * (y - 16)
    field[1] = 2 * (x - 16)

    unchecked = validate_deformation(field, labels, DeformationRejectionConfig())
    checked = validate_deformation(
        field,
        labels,
        DeformationRejectionConfig(preserved_wall_thickness_px=3),
    )

    assert unchecked.accepted
    assert not checked.accepted
    assert not checked.preserves_wall


def test_wall_preservation_rejection_config_is_backward_compatible() -> None:
    legacy = {
        "minimum_jacobian": 0.0,
        "minimum_foreground_margin_px": 1,
        "max_attempts": 20,
    }
    assert DeformationRejectionConfig.from_dict(legacy).to_dict() == legacy

    enabled = DeformationRejectionConfig(preserved_wall_thickness_px=3)
    assert DeformationRejectionConfig.from_dict(enabled.to_dict()) == enabled
