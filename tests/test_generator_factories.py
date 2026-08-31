import numpy as np
import torch

from constraints.datatools.label_schema import LabelSchema
from constraints.generators.composition import compose_layers
from constraints.generators.factories import (
    create_layer_collection,
    get_source_config,
)
from constraints.generators.layer_generators import (
    LayerPatch,
    PowerPlaqueSamplingRanges,
    bubble_cavity_layer_backup,
    create_empty_artery,
    power_layer_backup,
    register_layer_resolver,
    resolve_layer_patch,
    wall_attenuation_layer_backup,
)
from constraints.generators.recipe_backups import LayerBackup
from constraints.generators.source import create_source
from constraints.generators.types import (
    EmptyArteryConfig,
    FloatRange,
    SourceConfig,
)
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_with_wall,
)


def test_factory_creates_collection_from_source_root(tmp_path) -> None:
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=2,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )
    create_source(root, config)

    folder = create_layer_collection(
        root,
        "three-big-blobs",
        power_layer_backup((PowerPlaqueSamplingRanges(),) * 3, seed=12),
    )

    assert get_source_config(root) == config
    assert np.load(folder / "labels.npy").shape == (2, 65, 65)
    assert np.load(folder / "image.npy").shape == (2, 65, 65)


def test_registered_resolver_can_store_independent_patches(tmp_path) -> None:
    @register_layer_resolver("test-independent-patches-v1")
    def separated(context, params):
        shape = context.source_config.empty_artery.image_size
        labels = np.full(shape, -1, dtype=np.int8)
        labels[1, 1] = 3
        image = np.full(shape, np.nan, dtype=np.float32)
        image[1, 1:3] = [0.2, 0.8]
        return LayerPatch(labels, image)

    root = tmp_path / "source"
    create_source(root, SourceConfig(1, EmptyArteryConfig(20, 5, (65, 65))))

    folder = create_layer_collection(
        root,
        "separated",
        LayerBackup("test-independent-patches-v1"),
    )

    labels = np.load(folder / "labels.npy")
    image = np.load(folder / "image.npy")
    assert (labels[0] >= 0).sum() == 1
    assert np.isfinite(image[0]).sum() == 2


def test_bubble_cavity_layer_is_deterministic_and_best_effort(tmp_path) -> None:
    config = SourceConfig(num_elements=1)
    plaque_range = PowerPlaqueSamplingRanges(
        angular_width_rad=FloatRange.fixed(2.5),
        inward_depth_fraction=FloatRange.fixed(0.6),
        wall_depth_fraction=FloatRange.fixed(0.2),
    )
    backup = bubble_cavity_layer_backup(
        plaque_range,
        seed=17,
        bubbles_per_kind=100,
        maximum_attempts=1,
        plaque_blur_sigma_px=1.5,
    )

    first = resolve_layer_patch(backup, tmp_path, config, 0)
    second = resolve_layer_patch(backup, tmp_path, config, 0)

    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.image, second.image, equal_nan=True)
    plaque_image = first.image[first.labels >= 0]
    assert np.any((plaque_image > 0.25) & (plaque_image < 1.0))


def test_bubble_cavity_ground_truth_never_creates_floating_plaque(tmp_path) -> None:
    config = SourceConfig(num_elements=10)
    artery = create_empty_artery(config.empty_artery)
    backup = bubble_cavity_layer_backup(
        PowerPlaqueSamplingRanges(
            angular_width_rad=FloatRange(2.0, 2.6),
            inward_depth_fraction=FloatRange(0.45, 0.65),
            wall_depth_fraction=FloatRange(0.05, 0.25),
        ),
        seed=81,
        bubbles_per_kind=3,
        minimum_plaque_separation_px=5,
    )

    for index in range(config.num_elements):
        patch = resolve_layer_patch(backup, tmp_path, config, index)
        labels = compose_layers(artery, (patch,)).target_labels
        violation, details = does_violation_occur_with_wall(
            torch.from_numpy(labels), LabelSchema.as_artery()
        )
        assert not violation, details


def test_hard_bubble_render_does_not_leave_plaque_inside_bites(tmp_path) -> None:
    config = SourceConfig(num_elements=1)
    backup = bubble_cavity_layer_backup(
        PowerPlaqueSamplingRanges(
            angular_width_rad=FloatRange.fixed(2.5),
            inward_depth_fraction=FloatRange.fixed(0.6),
            wall_depth_fraction=FloatRange.fixed(0.2),
        ),
        seed=81,
        bubbles_per_kind=3,
        plaque_blur_sigma_px=0,
        bubble_blur_sigma_px=0,
    )

    patch = resolve_layer_patch(backup, tmp_path, config, 0)
    bite_pixels = patch.labels == 2

    assert np.any(bite_pixels)
    assert np.all(patch.image[bite_pixels] == 0.25)


def test_wall_attenuation_preserves_gt_wall_and_extends_only_image(tmp_path) -> None:
    config = SourceConfig(num_elements=8)
    artery = create_empty_artery(config.empty_artery)
    sampling = PowerPlaqueSamplingRanges(
        angular_width_rad=FloatRange(0.5, 1.0),
        inward_depth_fraction=FloatRange(0.15, 0.3),
        wall_depth_fraction=FloatRange.fixed(0),
    )
    backup = wall_attenuation_layer_backup(
        (sampling,) * 4,
        seed=93,
        residual_wall_px=(4, 5, 8, 12),
        gradient_length_px=FloatRange(10, 24),
    )

    for index in range(config.num_elements):
        patch = resolve_layer_patch(backup, tmp_path, config, index)
        repeated = resolve_layer_patch(backup, tmp_path, config, index)
        assert np.array_equal(patch.labels, repeated.labels)
        assert np.array_equal(patch.image, repeated.image, equal_nan=True)
        assert np.any(np.isfinite(patch.image) & (patch.labels == -1))

        labels = compose_layers(artery, (patch,)).target_labels
        violation, details = does_violation_occur_with_wall(
            torch.from_numpy(labels), LabelSchema.as_artery()
        )
        assert not violation, details
