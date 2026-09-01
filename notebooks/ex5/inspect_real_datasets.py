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
from constraints.utils import get_repo_root
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
import h5py
import matplotlib.pyplot as plt
import numpy as np

sample_id = 0
image_fname = image_fnames[sample_id]
label_fname = label_fnames[sample_id]
print(image_fname.name, label_fname.name)

with (
    h5py.File(image_fname, "r") as image_file,
    h5py.File(label_fname, "r") as label_file,
):
    print("Label keys:", list(label_file.keys()))
    image = np.array(image_file["trans_img"])
    label = np.array(label_file["trans_mask"], dtype=np.uint8)

print("Unique label values:", np.unique(label))

# Create 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Transformed Image
axes[0].imshow(image, cmap="gray")
axes[0].set_title(f"Image ({image_fname.name})")

# 2. Label Mask
im1 = axes[1].imshow(label, cmap="viridis")
axes[1].set_title("Label Mask")

# 3. Label == 255 Binary Mask
axes[2].imshow(label == 255, cmap="gray")
axes[2].set_title("Mask == 255")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

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

# %% [markdown]
# ## Image sizes

# %%
def acdc_shape(data_path: Path) -> tuple[int, int]:
    with h5py.File(ACDC_INIT / data_path, "r") as file:
        image_shape = tuple(file["image"].shape)
        label_shape = tuple(file["label"].shape)
    assert image_shape == label_shape
    assert len(image_shape) == 2
    return image_shape


shapes = [acdc_shape(path) for path in df["path"]]
shape_counts = (
    pl.DataFrame(
        {
            "image_height": [shape[0] for shape in shapes],
            "image_width": [shape[1] for shape in shapes],
        }
    )
    .group_by("image_height", "image_width")
    .len(name="num_images")
    .sort("num_images", descending=True)
)
print(f"All ACDC images have one size: {len(shape_counts) == 1}")
print(shape_counts)

# %% [markdown]
# ## Patient-disjoint annular-myocardium splits
#
# Split patients rather than individual slices so that slices from one patient
# cannot leak between train, validation, and test sets.

# %%
SPLIT_SEED = 42
MANIFEST_DIR = get_repo_root() / "dataset_manifests" / "acdc"

annular_df = df.filter(pl.col("annular_myocardium"))
patient_ids = np.array(sorted(annular_df["patient"].unique().to_list()))
np.random.default_rng(SPLIT_SEED).shuffle(patient_ids)

num_train_patients = int(0.70 * len(patient_ids))
num_val_patients = int(0.20 * len(patient_ids))
split_patients = {
    "trn": patient_ids[:num_train_patients],
    "val": patient_ids[
        num_train_patients : num_train_patients + num_val_patients
    ],
    "test": patient_ids[num_train_patients + num_val_patients :],
}

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
split_dfs = {}
for split_name, patients in split_patients.items():
    split_df = (
        annular_df.filter(pl.col("patient").is_in(patients))
        .sort("patient", "frame", "slice")
    )
    split_df.write_csv(MANIFEST_DIR / f"{split_name}.csv")
    split_dfs[split_name] = split_df
    print(
        split_name,
        f"patients={split_df['patient'].n_unique()}",
        f"slices={len(split_df)}",
    )

assert sum(len(split_df) for split_df in split_dfs.values()) == len(annular_df)
assert all(
    set(split_patients[left]).isdisjoint(split_patients[right])
    for left, right in (("trn", "val"), ("trn", "test"), ("val", "test"))
)

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
