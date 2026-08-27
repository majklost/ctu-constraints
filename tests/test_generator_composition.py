import numpy as np
import pytest

from constraints.generators.composition import compose_label_maps
from constraints.generators.types import AppearanceKind, ArteryClass, PlaqueLayer


def test_fake_layers_resolve_to_their_configured_anatomical_classes() -> None:
    artery = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
    boundary_fake = np.array([[False, False, True], [False, False, False]])
    lumen_fake = np.array([[False, False, False], [False, True, False]])

    label_maps = compose_label_maps(
        artery,
        [
            PlaqueLayer(boundary_fake, ArteryClass.BOUNDARY),
            PlaqueLayer(lumen_fake, ArteryClass.LUMEN),
        ],
    )

    assert label_maps.target_labels[0, 2] == ArteryClass.BOUNDARY
    assert label_maps.target_labels[1, 1] == ArteryClass.LUMEN
    np.testing.assert_array_equal(
        label_maps.appearance_labels,
        label_maps.target_labels,
    )


def test_later_plaque_layer_wins_at_overlaps() -> None:
    artery = np.full((2, 2), ArteryClass.LUMEN, dtype=np.uint8)
    overlap = np.array([[True, False], [False, False]])

    label_maps = compose_label_maps(
        artery,
        [
            PlaqueLayer(overlap, ArteryClass.PLAQUE),
            PlaqueLayer(overlap, ArteryClass.BOUNDARY),
        ],
    )

    assert label_maps.target_labels[0, 0] == ArteryClass.BOUNDARY
    assert label_maps.appearance_labels[0, 0] == AppearanceKind.BOUNDARY


def test_appearance_can_differ_from_target_class() -> None:
    artery = np.full((2, 2), ArteryClass.LUMEN, dtype=np.uint8)
    artifact = np.array([[True, False], [False, False]])

    label_maps = compose_label_maps(
        artery,
        [
            PlaqueLayer(
                artifact,
                ArteryClass.LUMEN,
                AppearanceKind.SHADOW,
            )
        ],
    )

    assert label_maps.target_labels[0, 0] == ArteryClass.LUMEN
    assert label_maps.appearance_labels[0, 0] == AppearanceKind.SHADOW


def test_plaque_layer_rejects_non_anatomical_target() -> None:
    with pytest.raises(ValueError, match="boundary, lumen, or plaque"):
        PlaqueLayer(np.ones((2, 2), bool), ArteryClass.BACKGROUND)
