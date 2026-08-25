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

from constraints.generators.parametrization import (
    PowerPlaqueParameters,
    create_artery_label_mask,
    create_power_plaque,
)
from constraints.generators.parametrization.plaque_samplers import (
    FloatRange,
    PowerPlaqueSamplingRanges,
    sample_power_plaque_parameter_batch,
    sample_power_plaque_parameters,
)
from constraints.generators.types import ArterySpec

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
fake_plaque_range = PowerPlaqueSamplingRanges(inward_depth_fraction=FloatRange(0.12,0.15),shape_power=FloatRange.fixed(2),wall_depth_fraction=FloatRange.fixed(0.1))
plaque_range1 = PowerPlaqueSamplingRanges(angle_rad=FloatRange(-np.pi/3,-np.pi/10),angular_width_rad=FloatRange.fixed(np.pi/5),inward_depth_fraction=FloatRange(0.2,0.3),shape_power=FloatRange.fixed(0.5),wall_depth_fraction=FloatRange.fixed(0))
plaque_range2 = PowerPlaqueSamplingRanges(angle_rad=FloatRange(np.pi/10,np.pi/3),angular_width_rad=FloatRange.fixed(np.pi/5),inward_depth_fraction=FloatRange(0.2,0.3),shape_power=FloatRange.fixed(0.5),wall_depth_fraction=FloatRange.fixed(0))

# %%
fake_params =sample_power_plaque_parameter_batch(fake_plaque_range,8,lumen_radius_px=LUMEN_RADIUS_PX-5,wall_thickness_px=WALL_THICKNESS_PX,rng=rng)
real_params =sample_power_plaque_parameter_batch((plaque_range1,plaque_range2),2,lumen_radius_px=LUMEN_RADIUS_PX,wall_thickness_px=WALL_THICKNESS_PX,rng=rng)
plaques = map(lambda x: create_power_plaque(x,LUMEN_RADIUS_PX),real_params)
fake_plaques = map(lambda x: create_power_plaque(x,LUMEN_RADIUS_PX-5),fake_params)

label_map = create_artery_label_mask(
            ArterySpec(
                image_size=IMAGE_SIZE,
                lumen_radius_px=LUMEN_RADIUS_PX,
                wall_thickness_px=WALL_THICKNESS_PX,
                plaques=plaques,
                fake_plaques=fake_plaques
            )
        )
plt.imshow(label_map)
plt.savefig(FOLDER/("mask"+str(fake_params[0].shape_power)))




# %%
from constraints.generators.types import ArteryClass

class_intensities = {ArteryClass.PLAQUE: 255.,
                      ArteryClass.FAKE_PLAQUE: 255.,
                      ArteryClass.BACKGROUND:0.,
                      ArteryClass.LUMEN: 60.,
                      ArteryClass.BOUNDARY: 200. 
                      }


# %%
from constraints.generators.parametrization.plaque_generators import create_grayscale_image_from_label_mask

img = create_grayscale_image_from_label_mask(label_map,class_intensities)
plt.imshow(img,cmap="grey")
plt.savefig(FOLDER/("image"+str(fake_params[0].shape_power)))

# %%
