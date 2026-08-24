import numpy as np
import pytest
from scipy.ndimage import binary_dilation, generate_binary_structure

from constraints.generators.parametrization.plaque_generators import (
    PowerPlaqueParameters,
    create_power_plaque,
    render_artery,
)
from constraints.generators.types import ArteryClass, ArterySpec, PlaqueSpec


def test_render_artery_creates_background_wall_and_lumen() -> None:
    labels = render_artery(
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
    labels = render_artery(
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
    labels = render_artery(
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
        outer_radius=lambda offset: 26.0,
    )

    with pytest.raises(ValueError, match="must remain inside"):
        render_artery(
            ArterySpec(
                image_size=(65, 65),
                lumen_radius_px=20,
                wall_thickness_px=5,
                plaques=(plaque,),
            )
        )
