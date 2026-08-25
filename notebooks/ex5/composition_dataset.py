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
    source_root,
    plaques=("2blobs",),
    deformation="default"
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
