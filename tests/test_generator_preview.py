import numpy as np
import pytest

from constraints.generators.factories import preview_artificial_sample
from constraints.generators.parametrization import create_power_plaque_mask
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    DeformationConfig,
    EmptyArteryConfig,
    FloatRange,
    PlaqueLayer,
    PowerPlaqueSamplingRanges,
    RigidConfig,
    RigidRejectionConfig,
)


def _sample_mask(
    artery_config: EmptyArteryConfig,
    ranges: PowerPlaqueSamplingRanges,
    num: int,
    *,
    seed: int,
    lumen_radius_px: float | None = None,
) -> np.ndarray:
    if lumen_radius_px is None:
        lumen_radius_px = artery_config.lumen_radius_px
    parameters = ranges.sample(
        num,
        lumen_radius_px=lumen_radius_px,
        wall_thickness_px=artery_config.wall_thickness_px,
        rng=np.random.default_rng(seed),
    )
    return create_power_plaque_mask(
        parameters,
        artery_config,
        lumen_radius_px=lumen_radius_px,
    )


def test_preview_sample_requires_no_storage() -> None:
    artery_config = EmptyArteryConfig(10, 3, (33, 33))
    plaque_mask = _sample_mask(
        artery_config,
        PowerPlaqueSamplingRanges(),
        1,
        seed=23,
    )

    sample = preview_artificial_sample(
        artery_config,
        (PlaqueLayer(plaque_mask),),
        deformation_config=DeformationConfig(
            scales=8,
            magnitude=0,
            integrations=2,
            fractal_mode="upsample",
        ),
        seed=23,
        sample_index=4,
    )

    assert sample.image.shape == artery_config.image_size
    assert sample.image.dtype == np.float32
    assert sample.target_labels.shape == artery_config.image_size
    assert sample.target_labels.dtype == np.uint8
    assert sample.appearance_labels.dtype == np.uint8
    assert ArteryClass.PLAQUE in sample.target_labels
    assert sample.deformation_field is not None
    assert sample.deformation_field.shape == (2, *artery_config.image_size)
    assert sample.deformation_validation is not None
    assert sample.deformation_validation.accepted
    assert sample.rigid_parameters is None


def test_one_range_can_create_multiple_plaques_in_one_preview_layer() -> None:
    artery_config = EmptyArteryConfig(20, 5, (65, 65))
    ranges = PowerPlaqueSamplingRanges(
        angular_width_rad=FloatRange.fixed(0.25),
        inward_depth_fraction=FloatRange.fixed(0.2),
        wall_depth_fraction=FloatRange.fixed(0.2),
    )
    parameters = ranges.sample(
        8,
        lumen_radius_px=artery_config.lumen_radius_px,
        wall_thickness_px=artery_config.wall_thickness_px,
        rng=np.random.default_rng(8),
    )
    mask = create_power_plaque_mask(parameters, artery_config)

    sample = preview_artificial_sample(
        artery_config,
        (PlaqueLayer(mask, ArteryClass.PLAQUE),),
        seed=23,
    )

    assert len(parameters) == 8
    np.testing.assert_array_equal(
        sample.target_labels[mask],
        ArteryClass.PLAQUE,
    )


def test_preview_accepts_ordered_real_and_fake_plaque_layers() -> None:
    artery_config = EmptyArteryConfig(10, 3, (33, 33))
    overlap = np.zeros(artery_config.image_size, dtype=bool)
    overlap[16, 16] = True

    sample = preview_artificial_sample(
        artery_config,
        (
            PlaqueLayer(overlap, ArteryClass.PLAQUE),
            PlaqueLayer(
                overlap,
                ArteryClass.LUMEN,
                AppearanceKind.PLAQUE,
            ),
        ),
        seed=23,
    )

    assert sample.target_labels[16, 16] == ArteryClass.LUMEN
    assert sample.appearance_labels[16, 16] == AppearanceKind.PLAQUE
    assert sample.image[16, 16] == 1.0


def test_preview_renders_non_anatomical_appearance_kind() -> None:
    artery_config = EmptyArteryConfig(10, 3, (33, 33))
    shadow = np.zeros(artery_config.image_size, dtype=bool)
    shadow[16, 16] = True
    intensities = {**DEFAULT_CLASS_INTENSITIES, AppearanceKind.SHADOW: 0.05}

    sample = preview_artificial_sample(
        artery_config,
        (
            PlaqueLayer(
                shadow,
                ArteryClass.LUMEN,
                AppearanceKind.SHADOW,
            ),
        ),
        seed=23,
        class_intensities=intensities,
    )

    assert sample.target_labels[16, 16] == ArteryClass.LUMEN
    assert sample.appearance_labels[16, 16] == AppearanceKind.SHADOW
    assert sample.image[16, 16] == pytest.approx(0.05)


def test_preview_sample_accepts_and_applies_rigid_configuration() -> None:
    artery_config = EmptyArteryConfig(10, 3, (33, 33))
    baseline = preview_artificial_sample(artery_config, seed=23, sample_index=4)

    sample = preview_artificial_sample(
        artery_config,
        rigid_config=RigidConfig(
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
    artery_config = EmptyArteryConfig(10, 3, (33, 33))

    with pytest.raises(RuntimeError, match="failed to sample valid rigid"):
        preview_artificial_sample(
            artery_config,
            rigid_config=RigidConfig(
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
