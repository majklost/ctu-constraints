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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Inspect checkpoint constraint violations

Load the best inference checkpoint for one run, reproduce one of its dataset
splits, and inspect every U-Net prediction rejected by
`does_violation_occur_with_wall`.

`checkpoint_validation` reproduces the split used to select the best
checkpoint, including the run's `swap_splits` setting.
"""

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json

import matplotlib.pyplot as plt
import polars
import torch
from IPython.display import display
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from constraints.datatools.datasets import ComposedArtificialDataset
from constraints.devices import resolve_compute_device
from constraints.generators.recipes import Recipe
from constraints.lightning_wrappers.modules import UnetLightning
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_with_wall,
)
from constraints.models.segmentator import set_segmentator_encoder_weights
from constraints.utils import get_repo_root

# %%
RUN_ID = "s0x3j84i"
# RUN_ID = "ct6wtl0n"
EXPERIMENT = "ex4"
EXPERIMENT_FILE = "initial_decoupled_new"

# Options: checkpoint_validation, checkpoint_train, original_validation,
# original_train.
DATASET_SPLIT = "checkpoint_validation"
BATCH_SIZE = 32

repo_root = get_repo_root()
run_dir = (
    repo_root / "synced" / "weights" / EXPERIMENT / EXPERIMENT_FILE / RUN_ID
)
metadata_path = run_dir / "metadata.json"
weights_path = run_dir / "weights.ckpt"

metadata = json.loads(metadata_path.read_text())
config = metadata["command_line_arguments"]

print(f"Run: {metadata['wandb']['name']} ({RUN_ID})")
print(
    "Best checkpoint: "
    f"{metadata['checkpoint_selection']['metric']}="
    f"{metadata['checkpoint_selection']['value']:.6f}"
)

# %% [markdown]
"""
## Reproduce the selected dataset split

This run was trained with `swap_splits=True`. Therefore its checkpoint
validation split is the source dataset's original training split. The mapping
below handles that from metadata rather than hard-coding it.
"""

# %%
recipe = Recipe.load_json(repo_root / config["recipe"])
source_root = recipe.resolve_source_root()

original_train_indices = polars.read_csv(
    source_root / "splits" / "trn_samples.csv"
)["sample_index"].to_list()
original_validation_indices = polars.read_csv(
    source_root / "splits" / "val_samples.csv"
)["sample_index"].to_list()

checkpoint_train_indices = original_train_indices
checkpoint_validation_indices = original_validation_indices
if config["swap_splits"]:
    checkpoint_train_indices, checkpoint_validation_indices = (
        checkpoint_validation_indices,
        checkpoint_train_indices,
    )

split_indices = {
    "checkpoint_validation": checkpoint_validation_indices,
    "checkpoint_train": checkpoint_train_indices,
    "original_validation": original_validation_indices,
    "original_train": original_train_indices,
}
if DATASET_SPLIT not in split_indices:
    raise ValueError(
        f"Unknown DATASET_SPLIT={DATASET_SPLIT!r}; "
        f"choose one of {tuple(split_indices)}"
    )

dataset = ComposedArtificialDataset.from_recipe(
    source_root,
    recipe,
    sample_list=split_indices[DATASET_SPLIT],
)
print(f"Loaded {len(dataset)} samples from {DATASET_SPLIT}")

# %% [markdown]
"""
## Load the best weights

Inference weights are deliberately excluded from Mutagen synchronization. If
this assertion fails, copy `weights.ckpt` from the training machine into the
run directory printed below.
"""

# %%
assert weights_path.is_file(), (
    f"Checkpoint not found: {weights_path}\n"
    "Copy weights.ckpt from the machine on which this run was trained."
)

device = resolve_compute_device()

# The checkpoint replaces every model parameter, so avoid downloading an
# otherwise unused ImageNet initialization while constructing the U-Net.
set_segmentator_encoder_weights(None)
model = UnetLightning(
    label_schema=dataset.label_schema,
    learning_rate=config["learning_rate"],
)
checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval().to(device)

print(f"Loaded {weights_path} on {device}")

# %% [markdown]
"""
## Find violating predictions

Ground-truth masks that already violate the same constraints are reported and
excluded from the prediction analysis. Each violating prediction is retained
as a compact `uint8` tensor so the viewer shows the exact mask that triggered
the recorded violation rather than running the model again with a different
batch size.
"""

# %%
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

violations = []
gt_violations = []
dataset_position = 0

with torch.inference_mode():
    for batch in loader:
        predictions = model(batch["image"].to(device)).argmax(dim=1).cpu()

        for prediction, sample_id, target_labels in zip(
            predictions,
            batch["sample_id"],
            batch["target_labels"],
            strict=True,
        ):
            current_position = dataset_position
            dataset_position += 1

            gt_violated, gt_details = does_violation_occur_with_wall(
                target_labels,
                dataset.label_schema,
            )
            if gt_violated:
                gt_violations.append(
                    {
                        "dataset_position": current_position,
                        "sample_id": sample_id,
                        "details": tuple(gt_details),
                    }
                )
                continue

            occurred, details = does_violation_occur_with_wall(
                prediction,
                dataset.label_schema,
            )
            if occurred:
                violations.append(
                    {
                        "dataset_position": current_position,
                        "sample_id": sample_id,
                        "details": tuple(details),
                        "prediction": prediction.to(torch.uint8),
                    }
                )

evaluated_samples = len(dataset) - len(gt_violations)
print(
    f"Found {len(violations)}/{evaluated_samples} violating predictions "
    f"({len(violations) / max(evaluated_samples, 1):.2%})"
)
print(
    f"Excluded {len(gt_violations)}/{len(dataset)} samples whose ground truth "
    "already violates the constraints"
)

# %%
violation_summary = polars.DataFrame(
    {
        "violation_index": range(len(violations)),
        "sample_id": [item["sample_id"] for item in violations],
        "details": [" | ".join(item["details"]) for item in violations],
    }
)
display(violation_summary)

# %% [markdown]
"""
## View violations

Pass a row number from `violation_summary` to `show_violation`. The figure
shows the input, ground truth, and checkpoint prediction using identical class
colors.
"""

# %%
from typing import List, Union

from matplotlib import gridspec

from constraints.visu.helpers import create_segmentation_overlay


cmap = ListedColormap(
    [
        dataset.label_schema.colors[index]
        for index in sorted(dataset.label_schema.colors)
    ]
)

def show_violations(violation_indices: Union[int, List[int]]) -> None:
    if isinstance(violation_indices, int):
        violation_indices = [violation_indices]

    num_rows = len(violation_indices)
    if num_rows == 0:
        return

    # Compact figure size: width=10, height proportional (~3.2 inches per row)
    fig = plt.figure(figsize=(10, 3.2 * num_rows))
    gs = gridspec.GridSpec(
        num_rows, 3, figure=fig, hspace=0.35, wspace=0.08, top=0.95, bottom=0.05
    )

    mask_options = {
        "cmap": cmap,
        "vmin": 0,
        "vmax": dataset.label_schema.num_classes - 1,
        "interpolation": "nearest",
    }

    for row_idx, v_idx in enumerate(violation_indices):
        record = violations[v_idx]
        sample = dataset[record["dataset_position"]]
        prediction = record["prediction"]

        occurred, details = does_violation_occur_with_wall(
            prediction,
            dataset.label_schema,
        )
        if not occurred or tuple(details) != record["details"]:
            raise RuntimeError(
                f"Constraint results changed for index {v_idx} (sample {record['sample_id']})."
            )

        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax2 = fig.add_subplot(gs[row_idx, 2])

        # 1. Grayscale Image
        img = sample["image"].squeeze()
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(
            f"Sample {record['sample_id']} (#{v_idx})",
            fontsize=10,
            fontweight="bold",
        )

        # 2. Ground Truth Overlay
        gt_overlay = create_segmentation_overlay(
            sample["image"], sample["target_labels"], cmap, alpha=0.7
        )
        ax1.imshow(gt_overlay, **mask_options)
        ax1.set_title("Ground truth", fontsize=10)

        # 3. Prediction Overlay
        pred_overlay = create_segmentation_overlay(
            sample["image"], prediction, cmap, alpha=0.7
        )
        ax2.imshow(pred_overlay, **mask_options)
        ax2.set_title("Prediction", fontsize=10)

        for ax in (ax0, ax1, ax2):
            ax.axis("off")

        # Display violation reason spanning across the row under the middle plot
        details_text = " • ".join(record["details"])
        ax1.text(
            0.5,
            -0.12,
            f"Violations: {details_text}",
            transform=ax1.transAxes,
            fontsize=8.5,
            ha="center",
            va="top",
            wrap=True,
        )

    plt.show()

# %%
import polars as pl
if violations:
    sample_ids_to_plot = ["2888", "723", "1275"]
    indices = [
        i
        for i, item in enumerate(violations)
        if str(item.get("sample_id")) in sample_ids_to_plot
    ]
    show_violations(indices)
else:
    print("No violating predictions to display.")

# %%
for i in range(min(50, len(violations))):
    show_violation(i)

# %%
