import numpy as np

from constraints import get_data_folder
from constraints.generators.factories import create_plaque_collection
from constraints.generators.types import FloatRange, PowerPlaqueSamplingRanges


def create_fake_similar():
    FOLDER = get_data_folder() / "artificial" / "samples5000"

    fake_plaque_range = PowerPlaqueSamplingRanges(
        angle_rad=FloatRange(np.pi / 3, 2 * np.pi - 2 * np.pi / 3),
        angular_width_rad=FloatRange.fixed(np.pi / 5),
        inward_depth_fraction=FloatRange(0.2, 0.3),
        shape_power=FloatRange.fixed(0.5),
        wall_depth_fraction=FloatRange.fixed(0),
        offset_px_lumen=FloatRange.fixed(-5),
    )

    create_plaque_collection(
        FOLDER,
        "FloatingFakeSimilarTwoPlaque",
        fake_plaque_range,
        seed=53,
    )


if __name__ == "__main__":
    create_fake_similar()
