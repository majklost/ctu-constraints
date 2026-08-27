import numpy as np
import pytest

from constraints.generators.composition import PlaqueLayer, compose_target_labels
from constraints.generators.factories import (
    PreviewPlaqueLayer,
    preview_artificial_sample,
)
from constraints.generators.parametrization.plaque_generators import (
    create_empty_artery,
)
from constraints.generators.source import sample_power_plaque_mask
from constraints.generators.types import (
    ArteryClass,
    DeformationConfig,
    EmptyArteryConfig,
    FloatRange,
    PowerPlaqueSamplingRanges,
    RigidBounds,
    RigidRejectionConfig,
    SourceConfig,
)


def test_preview_sample_requires_no_storage_and_returns_resolved_values() -> None:
    source_config = SourceConfig(
        num_elements=10,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )

    sample = preview_artificial_sample(
        source_config,
        deformation_config=DeformationConfig(
            scales=8,
            magnitude=0,
            integrations=2,
            fractal_mode="upsample",
        ),
        seed=23,
        sample_index=4,
    )

    assert sample.image.shape == source_config.image_size
    assert sample.image.dtype == np.float32
    assert sample.target_labels.shape == source_config.image_size
    assert sample.target_labels.dtype == np.uint8
    assert ArteryClass.PLAQUE in sample.target_labels
    assert len(sample.plaque_parameters) == 1
    assert sample.plaque_parameters_by_layer == {"preview": sample.plaque_parameters}
    assert sample.deformation_field is not None
    assert sample.deformation_field.shape == (2, *source_config.image_size)
    assert sample.deformation_validation is not None
    assert sample.deformation_validation.accepted
    assert sample.rigid_parameters is None


def test_preview_sample_accepts_real_and_fake_plaque_layers() -> None:
    source_config = SourceConfig(
        num_elements=10,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )
    real_ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0),
        angular_width_rad=FloatRange.fixed(0.4),
        inward_depth_fraction=FloatRange.fixed(0.2),
        wall_depth_fraction=FloatRange.fixed(0.2),
        shape_power=FloatRange.fixed(0.5),
    )
    fake_ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(np.pi),
        angular_width_rad=FloatRange.fixed(0.4),
        inward_depth_fraction=FloatRange.fixed(0.2),
        wall_depth_fraction=FloatRange.fixed(0.2),
        shape_power=FloatRange.fixed(0.5),
    )

    sample = preview_artificial_sample(
        source_config,
        seed=23,
        sample_index=4,
        plaque_layers=(
            PreviewPlaqueLayer("real", real_ranges),
            PreviewPlaqueLayer("fake", fake_ranges, ArteryClass.LUMEN),
        ),
    )

    real = sample_power_plaque_mask(source_config, real_ranges, seed=23, sample_index=4)
    fake = sample_power_plaque_mask(source_config, fake_ranges, seed=24, sample_index=4)
    expected = compose_target_labels(
        create_empty_artery(
            source_config.empty_artery,
            source_config.image_size,
        ),
        (
            PlaqueLayer("real", real.mask, ArteryClass.PLAQUE),
            PlaqueLayer("fake", fake.mask, ArteryClass.LUMEN),
        ),
    )

    np.testing.assert_array_equal(sample.target_labels, expected)
    assert sample.plaque_parameters_by_layer == {
        "real": real.parameters,
        "fake": fake.parameters,
    }


def test_preview_sample_accepts_and_applies_rigid_configuration() -> None:
    source_config = SourceConfig(
        num_elements=10,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )
    baseline = preview_artificial_sample(source_config, seed=23, sample_index=4)

    sample = preview_artificial_sample(
        source_config,
        rigid_config=RigidBounds(
            angle=FloatRange.fixed(0),
            dx=FloatRange.fixed(2),
            dy=FloatRange.fixed(-1),
        ),
        rigid_rejection=RigidRejectionConfig(
            minimum_foreground_margin_px=1,
            max_attempts=1,
        ),
        seed=23,
        sample_index=4,
    )

    np.testing.assert_array_equal(sample.rigid_parameters, [0, 2, -1])
    np.testing.assert_array_equal(
        sample.target_labels[0:-1, 2:], baseline.target_labels[1:, 0:-2]
    )


def test_preview_sample_applies_rigid_rejection_configuration() -> None:
    source_config = SourceConfig(
        num_elements=1,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )

    with pytest.raises(RuntimeError, match="failed to sample valid rigid"):
        preview_artificial_sample(
            source_config,
            rigid_config=RigidBounds(
                angle=FloatRange.fixed(0),
                dx=FloatRange.fixed(20),
                dy=FloatRange.fixed(0),
            ),
            rigid_rejection=RigidRejectionConfig(
                minimum_foreground_margin_px=1,
                max_attempts=1,
            ),
            seed=23,
        )
