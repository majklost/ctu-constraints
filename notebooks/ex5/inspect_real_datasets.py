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
from pathlib import Path
import h5py
from constraints import get_data_folder
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # CSV

# %%
CSV = get_data_folder() / "real/CSV_2026/CSV2026_Dataset_Annot"
assert CSV.is_dir()

# %%
CSV_IMAGES = CSV/"images"
CSV_LABELS = CSV/"labels"
assert CSV_IMAGES.is_dir()
assert CSV_LABELS.is_dir()

# %%
image_fnames = list(sorted(CSV_IMAGES.iterdir()))
label_fnames = list(sorted(CSV_LABELS.iterdir()))

print(len(image_fnames), len(label_fnames))





# %%
sample_id = 1
image_fname = image_fnames[sample_id]
label_fname = label_fnames[sample_id]
print(image_fname.name,label_fname.name)
image_file = h5py.File(image_fname)
label_file = h5py.File(label_fname)
print(label_file.keys())
image = np.array(image_file["trans_img"])
label = np.array(label_file["trans_mask"],dtype=np.uint8)

plt.imshow(image,cmap="gray")
plt.show()
plt.imshow(label)
print(np.unique_values(label))
plt.show()
plt.imshow(label==255)



# %%
for i in range(len(image_fnames)):
    image_fname = image_fnames[i].name
    label_fname = label_fnames[i].name
    assert image_fname == label_fname.replace("_label",""),"name differs"

# %%
