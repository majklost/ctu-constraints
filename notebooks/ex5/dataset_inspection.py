# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
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
# View the dataset and changes
"""

# %%
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from constraints import show_torch_image, show_torch_mask,get_data_folder
from constraints.datatools.datasets import CachedArtificialDataset


# %%
DATAF = get_data_folder()
DATAS = DATAF/"artificial"/"downloaded"/"deformed"/"trn"
print(list(DATAF.iterdir()))

# %%
dset = CachedArtificialDataset(DATAS,return_transform=True,return_template_sdf=True)
sample = dset[0]
show_torch_mask(sample['template'])

# %%
# show_torch_image(sample['image'],cmap="grey")



show_torch_mask(sample['target_labels'])
print(sample["target_labels"].unique())

# %%
show_torch_image(sample['image'],cmap="grey")

# %%
