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

Select named layer collections and compose them lazily with the reusable
empty-artery source.
"""

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgba
from constraints import get_data_folder
from constraints.datatools.datasets import ComposedArtificialDataset, SavedLayer

# %%
from constraints.generators.recipes import Recipe
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_no_wall,
    does_violation_occur_with_wall,
)
from constraints.utils import get_repo_root

RECIPE = "wall_attenuation"
# RECIPE = "bubble_cavities_overlap_gradient"
# RECIPE = "tworeal_fake_similar_offset_minus3"

source_root = get_data_folder() / "artificial" / "samples5000"
recipe_path = (
    get_repo_root() / f"recipes/artificial/{RECIPE}.json"
)
dataset = ComposedArtificialDataset.from_recipe(
    source_root, Recipe.load_json(recipe_path)
)
print(f"Loaded {len(dataset)} samples from {source_root}")

# %%
for i in range(len(dataset)):
    d = dataset[i]
    out = does_violation_occur_with_wall(d["target_labels"], dataset.label_schema)
    if out[0]:
        print(f"Sample {i}, violations:")
        for s in out[1]:
            print(s)
        plt.imshow(d["target_labels"])
        plt.show()

# %%
import torch
import numpy as np
def create_segmentation_overlay(
    image: torch.Tensor | np.ndarray,
    label_map: torch.Tensor | np.ndarray,
    cmap: ListedColormap | list | dict,
    alpha: float = 0.4,
    background_label: int = 0,
) -> np.ndarray:
    """Blends a grayscale image with a segmentation label map into an RGB image.

    Args:
        image: Grayscale tensor/array of shape (H, W) or (1, H, W), range
          [0, 1].
        label_map: Integer tensor/array of shape (H, W) containing class labels.
        cmap: Matplotlib ListedColormap, list/tuple of colors, or dict mapping
          label->color.
        alpha: Opacity factor for segmentation mask (0.0 to 1.0).
        background_label: Label index treated as transparent/unmasked background
          (set to None to color all labels).

    Returns:
        np.ndarray: Blended RGB float32 array of shape (H, W, 3) in range [0,
        1].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.detach().cpu().numpy()

    image = np.squeeze(image).astype(np.float32)
    label_map = np.squeeze(label_map).astype(np.int64)

    # Convert grayscale (H, W) to 3-channel RGB (H, W, 3)
    rgb_base = np.stack([image] * 3, axis=-1)
    rgb_base = np.clip(rgb_base, 0.0, 1.0)

    # Build color lookup table: shape (max_label + 1, 4) in RGBA
    max_label = int(label_map.max())
    if isinstance(cmap, dict):
        lut = np.zeros((max(max_label + 1, max(cmap.keys()) + 1), 4), dtype=np.float32)
        for lbl, color in cmap.items():
            lut[lbl] = to_rgba(color)
    elif isinstance(cmap, ListedColormap):
        colors = cmap.colors
        lut = np.array([to_rgba(c) for c in colors], dtype=np.float32)
    else:
        lut = np.array([to_rgba(c) for c in cmap], dtype=np.float32)

    # Map labels to RGB overlay
    overlay_rgb = lut[label_map, :3]

    # Create mask for pixels to blend (exclude background)
    if background_label is not None:
        mask = (label_map != background_label)[..., None]
    else:
        mask = np.ones((*label_map.shape, 1), dtype=bool)

    # Alpha blending on segmented regions
    blended = np.where(mask, (1.0 - alpha) * rgb_base + alpha * overlay_rgb, rgb_base)
    return np.clip(blended, 0.0, 1.0)

# %%


sample_indices = range(5,11)
fig, axes = plt.subplots(2,len(sample_indices), figsize=(18,8))

cmap = ListedColormap(
    [dataset.label_schema.colors[i] for i in sorted(dataset.label_schema.colors)]
)
for col, sample_index in enumerate(sample_indices):
    sample = dataset[sample_index]
    axes[0,col].imshow(sample["image"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0,col].set_title(f"Sample {sample_index}")
    axes[1,col].imshow(
        create_segmentation_overlay(sample["image"],sample["target_labels"],cmap,.8),
        cmap=cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[1,col].set_title(f"target labels")
    for axis in axes[:,col]:
        axis.axis("off")
fig.tight_layout()
fig.savefig(get_repo_root()/"reports/august26/images"/f"{RECIPE}.png")

# %% [markdown]
#

# %% [markdown]
"""
Layer collections already contain their independent image and label patches:

```python
dataset = ComposedArtificialDataset(
    source_root,
    layers=(
        SavedLayer("2blobs"),
        SavedLayer("floating-plaque"),
        SavedLayer("wall-artifact"),
    ),
)
```

The resolver that creates each collection decides both the grayscale pixels and
the labels; the Recipe only orders named layers.
"""

# %% [markdown]
# # Tuning parameters

# %%
import numpy as np
from matplotlib import pyplot as plt

from constraints.generators.factories import preview_artificial_sample
from constraints.generators.layer_generators import (
    MaskLayer,
    PowerPlaqueSamplingRanges,
    create_power_plaque_mask,
    normalize_layer_output,
)
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    DeformationConfig,
    DeformationRejectionConfig,
    FloatRange,
    RigidConfig,
    RigidRejectionConfig,
    SourceConfig,
)

# %%
rn = np.random.randint(0, 1000)
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
    offset_px_lumen=FloatRange.fixed(-5),
)


dc = DeformationConfig()
drc = DeformationRejectionConfig()
rc = RigidConfig()

rng = np.random.default_rng(rn)
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
fake_parameters = fake_plaque_range.sample(
    5,
    lumen_radius_px=artery_config.lumen_radius_px,
    wall_thickness_px=artery_config.wall_thickness_px,
    rng=rng,
)
layers = (
    normalize_layer_output(
        MaskLayer(
            create_power_plaque_mask(fake_parameters, artery_config),
            ArteryClass.LUMEN,
            AppearanceKind.PLAQUE,
        )
    ),
    normalize_layer_output(
        MaskLayer(create_power_plaque_mask(real_parameters, artery_config))
    ),
)

sample = preview_artificial_sample(
    artery_config,
    layers,
    deformation_config=dc,
    deformation_rejection=drc,
    # rigid_config=rc,
    seed=rn,
)
plt.imshow(sample.target_labels)
plt.show()
plt.imshow(sample.image, cmap="grey")


# %%

# %%
