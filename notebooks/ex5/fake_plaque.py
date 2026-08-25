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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from constraints.generators.parametrization import (
    PowerPlaqueParameters,
    create_power_plaque,
    render_artery,
)
from constraints.generators.types import ArterySpec

# %%
IMAGE_SIZE = (256, 256)
LUMEN_RADIUS_PX = 73.0
WALL_THICKNESS_PX = 12.0
