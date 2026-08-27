import numpy as np
import pytest

from constraints.generators.composition import compose_target_labels
from constraints.generators.types import ArteryClass, PlaqueLayer


def test_fake_layers_resolve_to_their_configured_anatomical_classes() -> None:
    artery = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
    boundary_fake = np.array([[False, False, True], [False, False, False]])
    lumen_fake = np.array([[False, False, False], [False, True, False]])

    labels = compose_target_labels(
        artery,
        [
            PlaqueLayer(boundary_fake, ArteryClass.BOUNDARY),
            PlaqueLayer(lumen_fake, ArteryClass.LUMEN),
        ],
    )

    assert labels[0, 2] == ArteryClass.BOUNDARY
    assert labels[1, 1] == ArteryClass.LUMEN
    assert set(np.unique(labels)) <= {0, 1, 2, 3}


def test_later_plaque_layer_wins_at_overlaps() -> None:
    artery = np.full((2, 2), ArteryClass.LUMEN, dtype=np.uint8)
    overlap = np.array([[True, False], [False, False]])

    labels = compose_target_labels(
        artery,
        [
            PlaqueLayer(overlap, ArteryClass.PLAQUE),
            PlaqueLayer(overlap, ArteryClass.BOUNDARY),
        ],
    )

    assert labels[0, 0] == ArteryClass.BOUNDARY


def test_plaque_layer_rejects_non_anatomical_target() -> None:
    with pytest.raises(ValueError, match="boundary, lumen, or plaque"):
        PlaqueLayer(np.ones((2, 2), bool), ArteryClass.BACKGROUND)
