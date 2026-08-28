import numpy as np

from constraints import get_data_folder
from constraints.generators.factories import create_rigid_collection
from constraints.generators.types import (
    FloatRange,
    PowerPlaqueSamplingRanges,
    RigidConfig,
)


def create_rotonly():
    FOLDER = get_data_folder() / "artificial" / "samples5000"/"deformation/default"

    rc = RigidConfig(dx=FloatRange.fixed(0), dy=FloatRange.fixed(0))

    create_rigid_collection(FOLDER, "FloatingFakeSimilarTwoPlaque", rc, seed=52)


if __name__ == "__main__":
    create_rotonly()
