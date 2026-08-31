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
import polars as pl

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

# %% [markdown]
# # ACDC

# %%
ACDC_INIT = get_data_folder() /"real"/"acdc"
ACDC = get_data_folder() /"real"/"acdc"/"ACDC_preprocessed"/"ACDC_training_slices"
assert ACDC.is_dir()

# %%
slices = list(ACDC.iterdir())
print(len(slices))

# %%
sample_id =1
file=h5py.File(slices[sample_id])
print(file.keys())
img = np.array(file['image'])
label = np.array(file['label'])
scribble = np.array(file['scribble'])

plt.imshow(img,cmap="gray")
plt.show()
plt.imshow(label)
plt.show()
plt.imshow(scribble)
plt.show()

print(np.unique(label))

# %%
print(img.shape)
print(label.shape)

# %%
fig,axs = plt.subplots(1,5)
for i in range(4):
    axs[i].imshow(label==i)
axs[4].imshow(img)
plt.show()
slices[0].relative_to(ACDC_INIT)

# %%
df =pl.DataFrame({"path": map(lambda s: str(s.relative_to(ACDC_INIT)) ,slices)})
pattern = r"patient(?<patient>\d+)_frame(?<frame>\d+)_slice_(?<slice>\d+)\.h5$"

result = (
    df.with_columns(
        pl.col("path").str.extract_groups(pattern).alias("extracted")
    )
    .unnest("extracted")
    .with_columns(
        pl.col("patient", "frame", "slice").cast(pl.Int64)
    )
)


df =result.sort(pl.col("patient"),pl.col("frame"),pl.col("slice"))
print(len(df))
df.head()

# %%
from constraints.datatools.utils.acdc_utils import is_annular


def open_acdc(data_path: Path):
    file = h5py.File(ACDC_INIT/ data_path)
    assert "label" in file.keys()
    assert "image" in file.keys()
    image = np.array(file["image"])
    label = np.array(file['label'])
    return image,label


def show_pair(data_path:Path):
    im, label = open_acdc(data_path)
    fig = plt.figure(f"pair {data_path}",(5,10))
    axs = fig.subplots(1,2)
    axs[0].imshow(im,cmap="gray")
    axs[1].imshow(label==2)
    plt.show()    

def check_annularity(data_path:Path):
    im,label = open_acdc(data_path)
    return is_annular(label == 2)
selected = df["path"][20]
print(check_annularity(selected))
show_pair(selected)

# %%
df = df.with_columns(annular_myocardium=pl.col("path").map_elements(check_annularity,pl.Boolean))
df.head()

# %%
print(df.select(pl.col("annular_myocardium").sum()))
print(len(df))

# %%
# print(df.select(pl.col("patient").n_unique()))
slice_summary = (
    df.group_by(["patient", "frame"])
    .agg(
        num_slices=pl.len(),
        min_slice=pl.col("slice").min(),
        max_slice=pl.col("slice").max()
    )
    .sort(["patient", "frame"])
)
slice_summary

# %%
