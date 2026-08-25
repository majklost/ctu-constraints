import numpy as np
import pytest
from scipy.ndimage import binary_dilation, generate_binary_structure

from constraints.generators.parametrization.plaque_generators import (
    PowerPlaqueParameters,
    create_anatomical_target_label_mask,
    create_artery_label_mask,
    create_grayscale_image_from_label_mask,
    create_power_plaque,
)
from constraints.generators.parametrization.plaque_samplers import (
    FloatRange,
    PowerPlaqueSamplingRanges,
    sample_power_plaque_parameter_batch,
    sample_power_plaque_parameters,
)
from constraints.generators.types import ArteryClass, ArterySpec, PlaqueSpec


def test_create_artery_label_mask_creates_background_wall_and_lumen() -> None:
    labels = create_artery_label_mask(
        ArterySpec(
            image_size=(65, 65),
            center_yx_px=(32, 32),
            lumen_radius_px=20,
            wall_thickness_px=5,
        )
    )

    assert labels[32, 32] == ArteryClass.LUMEN
    assert labels[32, 54] == ArteryClass.BOUNDARY
    assert labels[0, 0] == ArteryClass.BACKGROUND


def test_power_plaque_touches_lumen_and_wall_but_not_background() -> None:
    lumen_radius = 30.0
    plaque = create_power_plaque(
        PowerPlaqueParameters(
            angle_rad=0.0,
            angular_width_rad=np.deg2rad(40),
            inward_depth_px=10.0,
            wall_depth_px=3.0,
        ),
        lumen_radius_px=lumen_radius,
    )
    labels = create_artery_label_mask(
        ArterySpec(
            image_size=(97, 97),
            center_yx_px=(48, 48),
            lumen_radius_px=lumen_radius,
            wall_thickness_px=8,
            plaques=(plaque,),
        )
    )

    plaque_mask = labels == ArteryClass.PLAQUE
    adjacent = binary_dilation(plaque_mask, structure=generate_binary_structure(2, 2))
    assert np.any(adjacent & (labels == ArteryClass.LUMEN))
    assert np.any(adjacent & (labels == ArteryClass.BOUNDARY))
    assert not np.any(adjacent & (labels == ArteryClass.BACKGROUND))


def test_custom_radial_functions_can_cross_wrapped_angle_boundary() -> None:
    plaque = PlaqueSpec(
        angle_rad=np.pi,
        angular_width_rad=np.deg2rad(30),
        inner_radius=lambda offset: np.full_like(offset, 15.0),
        outer_radius=lambda offset: np.full_like(offset, 22.0),
    )
    labels = create_artery_label_mask(
        ArterySpec(
            image_size=(65, 65),
            center_yx_px=(32, 32),
            lumen_radius_px=20,
            wall_thickness_px=6,
            plaques=(plaque,),
        )
    )

    assert labels[32, 12] == ArteryClass.PLAQUE
    assert labels[32, 52] == ArteryClass.LUMEN


def test_renderer_rejects_plaque_reaching_background() -> None:
    plaque = PlaqueSpec(
        angle_rad=0,
        angular_width_rad=0.5,
        inner_radius=lambda offset: 18.0,
        outer_radius=lambda offset: 25.0,
    )

    with pytest.raises(ValueError, match="must preserve wall"):
        create_artery_label_mask(
            ArterySpec(
                image_size=(65, 65),
                lumen_radius_px=20,
                wall_thickness_px=5,
                plaques=(plaque,),
            )
        )


def test_wallless_artery_allows_plaque_to_reach_background_boundary() -> None:
    lumen_radius = 20.0
    parameters = sample_power_plaque_parameters(
        PowerPlaqueSamplingRanges(),
        lumen_radius_px=lumen_radius,
        wall_thickness_px=0,
        rng=np.random.default_rng(11),
    )
    plaque = create_power_plaque(parameters, lumen_radius_px=lumen_radius)

    labels = create_artery_label_mask(
        ArterySpec(
            image_size=(65, 65),
            center_yx_px=(32, 32),
            lumen_radius_px=lumen_radius,
            wall_thickness_px=0,
            plaques=(plaque,),
        )
    )

    assert parameters.wall_depth_px == 0
    assert not np.any(labels == ArteryClass.BOUNDARY)
    assert np.any(labels == ArteryClass.PLAQUE)


def test_sampling_resolves_relative_depths_to_pixels() -> None:
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange(0.25, 0.25),
        angular_width_rad=FloatRange(0.5, 0.5),
        inward_depth_fraction=FloatRange(0.2, 0.2),
        wall_depth_fraction=FloatRange(0.4, 0.4),
        shape_power=FloatRange(0.75, 0.75),
    )

    parameters = sample_power_plaque_parameters(
        ranges,
        lumen_radius_px=50,
        wall_thickness_px=10,
        rng=np.random.default_rng(123),
    )

    assert parameters == PowerPlaqueParameters(
        angle_rad=0.25,
        angular_width_rad=0.5,
        inward_depth_px=10,
        wall_depth_px=4,
        shape_power=0.75,
    )


def test_sampling_repeats_with_same_seed() -> None:
    ranges = PowerPlaqueSamplingRanges()

    first = sample_power_plaque_parameter_batch(
        ranges,
        4,
        lumen_radius_px=73,
        wall_thickness_px=12,
        rng=np.random.default_rng(42),
    )
    second = sample_power_plaque_parameter_batch(
        ranges,
        4,
        lumen_radius_px=73,
        wall_thickness_px=12,
        rng=np.random.default_rng(42),
    )

    assert first == second
    assert len(first) == 4


def test_batch_sampling_accepts_one_range_configuration_per_plaque() -> None:
    first_ranges = PowerPlaqueSamplingRanges(angle_rad=FloatRange.fixed(0.25))
    second_ranges = PowerPlaqueSamplingRanges(angle_rad=FloatRange.fixed(1.25))

    samples = sample_power_plaque_parameter_batch(
        (first_ranges, second_ranges),
        2,
        lumen_radius_px=73,
        wall_thickness_px=12,
        rng=np.random.default_rng(42),
    )

    assert tuple(sample.angle_rad for sample in samples) == (0.25, 1.25)


def test_batch_sampling_rejects_wrong_number_of_range_configurations() -> None:
    with pytest.raises(ValueError, match="expected 2.*got 1"):
        sample_power_plaque_parameter_batch(
            (PowerPlaqueSamplingRanges(),),
            2,
            lumen_radius_px=73,
            wall_thickness_px=12,
            rng=np.random.default_rng(42),
        )


def test_sampling_stays_inside_ranges() -> None:
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange(-1, 1),
        angular_width_rad=FloatRange(0.2, 0.8),
        inward_depth_fraction=FloatRange(0.1, 0.4),
        wall_depth_fraction=FloatRange(0.2, 0.7),
        shape_power=FloatRange(0.3, 1.5),
    )
    samples = sample_power_plaque_parameter_batch(
        ranges,
        100,
        lumen_radius_px=50,
        wall_thickness_px=10,
        rng=np.random.default_rng(7),
    )

    assert all(-1 <= sample.angle_rad <= 1 for sample in samples)
    assert all(0.2 <= sample.angular_width_rad <= 0.8 for sample in samples)
    assert all(5 <= sample.inward_depth_px <= 20 for sample in samples)
    assert all(2 <= sample.wall_depth_px <= 7 for sample in samples)
    assert all(0.3 <= sample.shape_power <= 1.5 for sample in samples)


def test_fake_plaque_is_separate_for_synthesis_and_boundary_in_target() -> None:
    lumen_radius = 30.0
    fake_plaque = create_power_plaque(
        PowerPlaqueParameters(
            angle_rad=0,
            angular_width_rad=np.deg2rad(35),
            inward_depth_px=8,
            wall_depth_px=3,
        ),
        lumen_radius_px=lumen_radius,
    )
    synthesis_mask = create_artery_label_mask(
        ArterySpec(
            image_size=(97, 97),
            center_yx_px=(48, 48),
            lumen_radius_px=lumen_radius,
            wall_thickness_px=8,
            fake_plaques=(fake_plaque,),
        )
    )

    target_mask = create_anatomical_target_label_mask(synthesis_mask)

    assert np.any(synthesis_mask == ArteryClass.FAKE_PLAQUE)
    assert not np.any(target_mask == ArteryClass.FAKE_PLAQUE)
    assert np.all(
        target_mask[synthesis_mask == ArteryClass.FAKE_PLAQUE]
        == ArteryClass.BOUNDARY
    )


def test_fake_and_real_plaque_can_have_same_grayscale_intensity() -> None:
    label_mask = np.array(
        [[ArteryClass.PLAQUE, ArteryClass.FAKE_PLAQUE]], dtype=np.uint8
    )

    image = create_grayscale_image_from_label_mask(
        label_mask,
        {
            ArteryClass.PLAQUE: 0.7,
            ArteryClass.FAKE_PLAQUE: 0.7,
        },
    )

    np.testing.assert_array_equal(image, np.array([[0.7, 0.7]], dtype=np.float32))
