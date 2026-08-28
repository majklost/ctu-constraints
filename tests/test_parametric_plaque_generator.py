import numpy as np
import pytest
from scipy.ndimage import binary_dilation, generate_binary_structure

from constraints.generators.parametrization.plaque_generators import (
    _create_plaque_mask,
    _PlaqueSpec,
    create_empty_artery,
    create_power_plaque_mask,
)
from constraints.generators.types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    PowerPlaqueParameters,
    PowerPlaqueSamplingRanges,
)


def test_create_empty_artery_creates_background_wall_and_lumen() -> None:
    config = EmptyArteryConfig(20, 5, (65, 65))

    labels = create_empty_artery(config)

    assert labels[32, 32] == ArteryClass.LUMEN
    assert labels[32, 54] == ArteryClass.BOUNDARY
    assert labels[0, 0] == ArteryClass.BACKGROUND


def test_power_plaque_touches_lumen_and_wall_but_not_background() -> None:
    config = EmptyArteryConfig(30, 8, (97, 97))
    mask = create_power_plaque_mask(
        (
            PowerPlaqueParameters(
                angle_rad=0.0,
                angular_width_rad=np.deg2rad(40),
                inward_depth_px=10.0,
                wall_depth_px=3.0,
            ),
        ),
        config,
    )
    labels = create_empty_artery(config)
    labels[mask] = ArteryClass.PLAQUE

    adjacent = binary_dilation(mask, structure=generate_binary_structure(2, 2))
    assert np.any(adjacent & (labels == ArteryClass.LUMEN))
    assert np.any(adjacent & (labels == ArteryClass.BOUNDARY))
    assert not np.any(adjacent & (labels == ArteryClass.BACKGROUND))


def test_internal_plaque_spec_supports_wrapped_angle_boundary() -> None:
    config = EmptyArteryConfig(20, 6, (65, 65))
    plaque = _PlaqueSpec(
        angle_rad=np.pi,
        angular_width_rad=np.deg2rad(30),
        inner_radius=lambda offset: np.full_like(offset, 15.0),
        outer_radius=lambda offset: np.full_like(offset, 22.0),
    )

    mask = _create_plaque_mask((plaque,), config)

    assert mask[32, 12]
    assert not mask[32, 52]


def test_renderer_rejects_plaque_reaching_background() -> None:
    config = EmptyArteryConfig(20, 5, (65, 65))
    plaque = _PlaqueSpec(
        angle_rad=0,
        angular_width_rad=0.5,
        inner_radius=lambda offset: 18.0,
        outer_radius=lambda offset: 25.0,
    )

    with pytest.raises(ValueError, match="must preserve wall"):
        _create_plaque_mask((plaque,), config)


def test_wallless_artery_allows_plaque_to_reach_outer_boundary() -> None:
    config = EmptyArteryConfig(20, 0, (65, 65))
    parameters = PowerPlaqueSamplingRanges().sample(
        1,
        lumen_radius_px=config.lumen_radius_px,
        wall_thickness_px=config.wall_thickness_px,
        rng=np.random.default_rng(11),
    )

    mask = create_power_plaque_mask(parameters, config)

    assert parameters[0].wall_depth_px == 0
    assert mask.any()


def test_sampling_resolves_relative_depths_to_pixels() -> None:
    ranges = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange.fixed(0.25),
        angular_width_rad=FloatRange.fixed(0.5),
        inward_depth_fraction=FloatRange.fixed(0.2),
        wall_depth_fraction=FloatRange.fixed(0.4),
        shape_power=FloatRange.fixed(0.75),
        offset_px_lumen=FloatRange.fixed(-5),
    )

    parameters = ranges.sample(
        1,
        lumen_radius_px=50,
        wall_thickness_px=10,
        rng=np.random.default_rng(123),
    )

    assert parameters == (
        PowerPlaqueParameters(
            angle_rad=0.25,
            angular_width_rad=0.5,
            inward_depth_px=10,
            wall_depth_px=4,
            shape_power=0.75,
            offset_px_lumen=-5,
        ),
    )


def test_negative_lumen_offset_moves_plaque_inside_lumen() -> None:
    config = EmptyArteryConfig(20, 5, (65, 65))
    parameters = (
        PowerPlaqueParameters(
            angle_rad=0,
            angular_width_rad=0.5,
            inward_depth_px=2,
            wall_depth_px=1,
            offset_px_lumen=-5,
        ),
    )

    mask = create_power_plaque_mask(parameters, config)

    assert mask[32, 47]  # radius 15: shifted plaque centerline
    assert not mask[32, 52]  # radius 20: actual lumen boundary


def test_lumen_offset_must_leave_a_positive_base_radius() -> None:
    config = EmptyArteryConfig(20, 5, (65, 65))
    parameters = (PowerPlaqueParameters(0, 0.5, 2, 1, offset_px_lumen=-20),)

    with pytest.raises(ValueError, match="offset_px_lumen"):
        create_power_plaque_mask(parameters, config)


def test_sampling_multiple_parameters_repeats_with_same_seed() -> None:
    ranges = PowerPlaqueSamplingRanges()

    first = ranges.sample(
        8,
        lumen_radius_px=73,
        wall_thickness_px=12,
        rng=np.random.default_rng(42),
    )
    second = ranges.sample(
        8,
        lumen_radius_px=73,
        wall_thickness_px=12,
        rng=np.random.default_rng(42),
    )

    assert first == second
    assert len(first) == 8


@pytest.mark.parametrize("num", [-1, 1.5, True])
def test_sampling_rejects_invalid_count(num) -> None:
    with pytest.raises(ValueError, match="num"):
        PowerPlaqueSamplingRanges().sample(
            num,
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
    samples = ranges.sample(
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


def test_multiple_parameters_are_rasterized_into_one_union_mask() -> None:
    config = EmptyArteryConfig(20, 5, (65, 65))
    first = PowerPlaqueParameters(0, 0.4, 5, 1)
    second = PowerPlaqueParameters(np.pi, 0.4, 5, 1)

    combined = create_power_plaque_mask((first, second), config)
    first_only = create_power_plaque_mask((first,), config)
    second_only = create_power_plaque_mask((second,), config)

    np.testing.assert_array_equal(combined, first_only | second_only)
