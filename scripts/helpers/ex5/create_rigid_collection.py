import numpy as np

from constraints import get_data_folder
from constraints.generators.factories import create_rigid_collection
from constraints.generators.types import (
    FloatRange,
    RigidConfig,
)


def create_rotonly():
    folder = get_data_folder() / "artificial" / "samples5000"

    rc = RigidConfig(dx=FloatRange.fixed(0), dy=FloatRange.fixed(0))

    create_rigid_collection(
        folder,
        "FloatingFakeSimilarTwoPlaque",
        deformation="default",
        config=rc,
        seed=52,
    )


if __name__ == "__main__":
    create_rotonly()
