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

from constraints.computers.metric_computers import StagedMetricComputer
from constraints.transforms.transformers import RigidTransformer

# %%


plt.imshow(torch.rand((200, 200, 3)).numpy())

# %%
pd.DataFrame({"a": [1, 2], "b": [3, 4]})

# %%
print(torch.cuda.is_available())

# %%
print("TEST")

# %%
