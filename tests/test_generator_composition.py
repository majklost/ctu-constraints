import numpy as np

from constraints.generators.composition import compose_layers
from constraints.generators.layer_generators import (
    LayerPatch,
    MaskLayer,
    normalize_layer_output,
)
from constraints.generators.types import AppearanceKind, ArteryClass


def test_label_and_grayscale_patches_are_independent() -> None:
    artery = np.array([[2, 2, 2]], dtype=np.uint8)
    labels = np.array([[ArteryClass.PLAQUE, -1, -1]], dtype=np.int8)
    image = np.array([[np.nan, 0.3, 0.7]], dtype=np.float32)

    result = compose_layers(artery, (LayerPatch(labels, image),))

    np.testing.assert_array_equal(result.target_labels, [[3, 2, 2]])
    np.testing.assert_allclose(result.image, [[0.25, 0.3, 0.7]])


def test_later_patch_wins_only_where_it_is_not_transparent() -> None:
    artery = np.full((1, 2), ArteryClass.LUMEN, dtype=np.uint8)
    first = LayerPatch(
        np.array([[3, 3]], dtype=np.int8),
        np.array([[0.2, 0.4]], dtype=np.float32),
    )
    second = LayerPatch(
        np.array([[1, -1]], dtype=np.int8),
        np.array([[np.nan, 0.8]], dtype=np.float32),
    )

    result = compose_layers(artery, (first, second))

    np.testing.assert_array_equal(result.target_labels, [[1, 3]])
    np.testing.assert_allclose(result.image, [[0.2, 0.8]])


def test_mask_layer_is_the_uniform_convenience_form() -> None:
    patch = normalize_layer_output(
        MaskLayer(
            np.array([[True, False]]),
            ArteryClass.LUMEN,
            AppearanceKind.SHADOW,
        )
    )

    np.testing.assert_array_equal(patch.labels, [[2, -1]])
    assert patch.image[0, 0] == np.float32(0.05)
    assert np.isnan(patch.image[0, 1])


def test_gradient_is_kept_as_real_grayscale_values() -> None:
    gradient = np.linspace(0.1, 0.9, 5, dtype=np.float32)[None]
    patch = LayerPatch(np.full((1, 5), -1, dtype=np.int8), gradient)

    result = compose_layers(np.full((1, 5), 2, dtype=np.uint8), (patch,))

    np.testing.assert_allclose(result.image, gradient)
