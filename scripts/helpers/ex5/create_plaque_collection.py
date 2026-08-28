import numpy as np

from constraints import get_data_folder
from constraints.generators.factories import create_layer_collection
from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    power_layer_backup,
)
from constraints.generators.types import FloatRange


def create_fake_similar():
    folder = get_data_folder() / "artificial" / "samples5000"

    fake_plaque_range = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange(np.pi / 3, 2 * np.pi - 2 * np.pi / 3),
        angular_width_rad=FloatRange.fixed(np.pi / 5),
        inward_depth_fraction=FloatRange(0.2, 0.3),
        shape_power=FloatRange.fixed(0.5),
        wall_depth_fraction=FloatRange.fixed(0),
        offset_px_lumen=FloatRange.fixed(-5),
    )

    create_layer_collection(
        folder,
        "FloatingFakeSimilarTwoPlaque",
        power_layer_backup((fake_plaque_range,) * 2, seed=53),
    )


if __name__ == "__main__":
    create_fake_similar()
