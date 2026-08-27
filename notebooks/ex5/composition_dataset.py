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

# %% [markdown]
"""
# Composed artificial source dataset

Select named plaque collections and compose them lazily with the reusable
empty-artery source.
"""

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from constraints import get_data_folder
from constraints.datatools.datasets import ComposedArtificialDataset

# %%
source_root = get_data_folder() / "artificial" / "demo"
dataset = ComposedArtificialDataset(
    source_root, plaques=("2blobs",), deformation="validated-default"
)
print(f"Loaded {len(dataset)} samples from {source_root}")

# %%
label_cmap = ListedColormap(
    [
        "black",  # background
        "firebrick",  # boundary
        "seagreen",  # lumen
        "royalblue",  # plaque
    ]
)

sample_indices = range(6)
fig, axes = plt.subplots(len(sample_indices), 2, figsize=(8, 18))
for row, sample_index in enumerate(sample_indices):
    sample = dataset[sample_index]
    axes[row, 0].imshow(sample["image"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Sample {sample_index}: image")
    axes[row, 1].imshow(
        sample["target_labels"],
        cmap=label_cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[row, 1].set_title(f"Sample {sample_index}: target labels")
    for axis in axes[row]:
        axis.axis("off")
fig.tight_layout()

# %% [markdown]
#

# %% [markdown]
"""
Fake plaques can be selected independently and assigned their anatomical target:

```python
dataset = ComposedArtificialDataset(
    source_root,
    plaques=("2blobs",),
    fake_plaques={
        "floating-plaque": ArteryClass.LUMEN,
        "wall-artifact": ArteryClass.BOUNDARY,
    },
)
```

Those masks retain plaque-like appearance in the image while resolving to the
configured class in `target_labels`.
"""

# %% [markdown]
# # Tuning parameters

# %%
from constraints.generators.factories import preview_artificial_sample
from constraints.generators.parametrization import create_power_plaque_mask
from constraints.generators.types import (
    AppearanceKind,
    SourceConfig,
    PowerPlaqueSamplingRanges,
    DeformationConfig,
    DeformationRejectionConfig,
    RigidConfig,
    RigidRejectionConfig,
    FloatRange,
    PlaqueLayer,
    ArteryClass,
)
from matplotlib import pyplot as plt
import numpy as np

# %%
sc = SourceConfig(1)
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

fake_plaque_range = PowerPlaqueSamplingRanges(
    inward_depth_fraction=FloatRange(0.12, 0.15),
    shape_power=FloatRange.fixed(2),
    wall_depth_fraction=FloatRange.fixed(0.1),
)


dc = DeformationConfig()
drc = DeformationRejectionConfig()
rc = RigidConfig()

rng = np.random.default_rng(25)
artery_config = sc.empty_artery
real_parameters = plaque_range1.sample(
    1,
    lumen_radius_px=artery_config.lumen_radius_px,
    wall_thickness_px=artery_config.wall_thickness_px,
    rng=rng,
) + plaque_range2.sample(
    1,
    lumen_radius_px=artery_config.lumen_radius_px,
    wall_thickness_px=artery_config.wall_thickness_px,
    rng=rng,
)
fake_lumen_radius_px = artery_config.lumen_radius_px - 5
fake_parameters = fake_plaque_range.sample(
    8,
    lumen_radius_px=fake_lumen_radius_px,
    wall_thickness_px=artery_config.wall_thickness_px,
    rng=rng,
)
plaque_layers = (
    PlaqueLayer(
        create_power_plaque_mask(
            fake_parameters, artery_config, lumen_radius_px=fake_lumen_radius_px
        ),
        ArteryClass.BOUNDARY,
        AppearanceKind.PLAQUE,
    ),
    PlaqueLayer(create_power_plaque_mask(real_parameters, artery_config)),
)


# %%
rn = np.random.randint(0, 1000)
sample = preview_artificial_sample(
    artery_config,
    plaque_layers,
    deformation_config=dc,
    deformation_rejection=drc,
    rigid_config=rc,
    seed=rn,
)
plt.imshow(sample.target_labels)

# %%
