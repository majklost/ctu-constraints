# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ctu-constraints
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from constraints.generators.factories import preview_artificial_sample
from constraints.generators.parametrization import create_power_plaque_mask
from constraints.generators.types import (
    ArteryClass,
    EmptyArteryConfig,
    FloatRange,
    PlaqueLayer,
    PowerPlaqueSamplingRanges,
)

from constraints import get_experiment_folder

FOLDER = get_experiment_folder("ex5/fake")
print(FOLDER.absolute)

# %%
IMAGE_SIZE = (256, 256)
LUMEN_RADIUS_PX = 73.0
WALL_THICKNESS_PX = 12.0
WALL_DEPTH_PX = 0
rng = np.random.default_rng(25)

# %%
fake_plaque_range = PowerPlaqueSamplingRanges(
    inward_depth_fraction=FloatRange(0.12, 0.15),
    shape_power=FloatRange.fixed(2),
    wall_depth_fraction=FloatRange.fixed(0.1),
)
plaque_range1 = PowerPlaqueSamplingRanges(
    angle_rad=FloatRange(-np.pi / 3, -np.pi / 10),
    angular_width_rad=FloatRange.fixed(np.pi / 5),
    inward_depth_fraction=FloatRange(0.2, 0.3),
    shape_power=FloatRange.fixed(0.5),
    wall_depth_fraction=FloatRange.fixed(0),
)
plaque_range2 = PowerPlaqueSamplingRanges(
    angle_rad=FloatRange(np.pi / 10, np.pi / 3),
    angular_width_rad=FloatRange.fixed(np.pi / 5),
    inward_depth_fraction=FloatRange(0.2, 0.3),
    shape_power=FloatRange.fixed(0.5),
    wall_depth_fraction=FloatRange.fixed(0),
)

# %%
artery_config = EmptyArteryConfig(LUMEN_RADIUS_PX, WALL_THICKNESS_PX, IMAGE_SIZE)
fake_lumen_radius_px = LUMEN_RADIUS_PX - 5
fake_params = fake_plaque_range.sample(
    8,
    lumen_radius_px=fake_lumen_radius_px,
    wall_thickness_px=WALL_THICKNESS_PX,
    rng=rng,
)
real_params = plaque_range1.sample(
    1, lumen_radius_px=LUMEN_RADIUS_PX, wall_thickness_px=WALL_THICKNESS_PX, rng=rng
) + plaque_range2.sample(
    1, lumen_radius_px=LUMEN_RADIUS_PX, wall_thickness_px=WALL_THICKNESS_PX, rng=rng
)
layers = (
    PlaqueLayer(
        create_power_plaque_mask(
            fake_params, artery_config, lumen_radius_px=fake_lumen_radius_px
        ),
        ArteryClass.LUMEN,
    ),
    PlaqueLayer(create_power_plaque_mask(real_params, artery_config)),
)
sample = preview_artificial_sample(artery_config, layers, seed=25)
label_map = sample.target_labels
plt.imshow(label_map)
plt.savefig(FOLDER / ("mask" + str(fake_params[0].shape_power)))


# %%
img = sample.image
plt.imshow(img, cmap="grey")
plt.savefig(FOLDER / ("image" + str(fake_params[0].shape_power)))

# %%
