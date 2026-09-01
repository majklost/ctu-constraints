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
# Inspect ACDC checkpoint annularity violations

Load the best inference checkpoint for ACDC U-Net run `zipjfhp2`, reproduce a
dataset split, find every non-annular predicted myocardium mask, and inspect
the corresponding image, ground truth, and prediction.

`checkpoint_validation` reproduces the validation split used to select the
best checkpoint, including the run's `swap_splits` setting.
"""

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json
from textwrap import fill

import matplotlib.pyplot as plt
import polars as pl
import torch
from IPython.display import display
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from constraints import get_data_folder
from constraints.datatools.datasets import ACDCSliceMyocardiumOnlyDataset
from constraints.devices import resolve_compute_device
from constraints.lightning_wrappers.modules import UnetLightning
from constraints.losses_metrics.metrics import ACDCAnnularityViolationCounter
from constraints.models.segmentator import set_segmentator_encoder_weights
from constraints.utils import get_repo_root
from constraints.visu.helpers import create_segmentation_overlay

# %%
RUN_ID = "zipjfhp2"
EXPERIMENT = "ex5"
EXPERIMENT_FILE = "initial_decoupled_new"

# Options: checkpoint_validation, checkpoint_train, original_validation,
# original_train, test.
DATASET_SPLIT = "checkpoint_validation"
BATCH_SIZE = 16
MIN_HOLE_AREA = 10

repo_root = get_repo_root()
run_dir = (
    repo_root / "synced" / "weights" / EXPERIMENT / EXPERIMENT_FILE / RUN_ID
)
metadata_path = run_dir / "metadata.json"
weights_path = run_dir / "weights.ckpt"

metadata = json.loads(metadata_path.read_text())
config = metadata["command_line_arguments"]

if config["dataset"] != "acdc" or config["mode"] != "UNET":
    raise ValueError(
        "This notebook requires an ACDC UNET run, got "
        f"dataset={config['dataset']!r}, mode={config['mode']!r}."
    )

print(f"Run: {metadata['wandb']['name']} ({RUN_ID})")
print(
    "Best checkpoint: "
    f"{metadata['checkpoint_selection']['metric']}="
    f"{metadata['checkpoint_selection']['value']:.6f}"
)

# %% [markdown]
"""
## Reproduce the selected ACDC split

The paths stored in the training metadata belong to the training machine, so
the notebook deliberately resolves the tracked manifests and ACDC data from
this local checkout.
"""

# %%
manifest_dir = repo_root / "dataset_manifests" / "acdc"
original_train_df = pl.read_csv(manifest_dir / "trn.csv")
original_validation_df = pl.read_csv(manifest_dir / "val.csv")
test_df = pl.read_csv(manifest_dir / "test.csv")

checkpoint_train_df = original_train_df
checkpoint_validation_df = original_validation_df
if config["swap_splits"]:
    checkpoint_train_df, checkpoint_validation_df = (
        checkpoint_validation_df,
        checkpoint_train_df,
    )

split_dfs = {
    "checkpoint_validation": checkpoint_validation_df,
    "checkpoint_train": checkpoint_train_df,
    "original_validation": original_validation_df,
    "original_train": original_train_df,
    "test": test_df,
}
if DATASET_SPLIT not in split_dfs:
    raise ValueError(
        f"Unknown DATASET_SPLIT={DATASET_SPLIT!r}; "
        f"choose one of {tuple(split_dfs)}"
    )

split_df = split_dfs[DATASET_SPLIT]
image_size = tuple(config["acdc_image_size"])
dataset = ACDCSliceMyocardiumOnlyDataset(
    get_data_folder() / "real" / "acdc",
    split_df,
    image_size=image_size,
)
print(
    f"Loaded {len(dataset)} slices from {split_df['patient'].n_unique()} "
    f"patients in {DATASET_SPLIT}; image size={image_size}"
)

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
    "Copy weights.ckpt from the machine on which run "
    f"{RUN_ID} was trained into this directory."
)

device = resolve_compute_device()

# The checkpoint replaces every parameter, so avoid downloading an otherwise
# unused ImageNet initialization while constructing the U-Net.
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
## Find non-annular myocardium predictions

The same `ACDCAnnularityViolationCounter` used by training metrics classifies
both predictions and ground truth. Ground-truth violations are reported and
excluded from the prediction violation rate.
"""

# %%
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)
annularity_counter = ACDCAnnularityViolationCounter(
    dataset.label_schema,
    min_hole_area=MIN_HOLE_AREA,
)

violations = []
gt_violations = []
dataset_position = 0

with torch.inference_mode():
    for batch in loader:
        predictions = model(batch["image"].to(device)).argmax(dim=1).cpu()
        prediction_results = annularity_counter.classify(predictions)
        gt_results = annularity_counter.classify(batch["target_labels"])

        for prediction, sample_id, prediction_result, gt_result in zip(
            predictions,
            batch["sample_id"],
            prediction_results,
            gt_results,
            strict=True,
        ):
            current_position = dataset_position
            dataset_position += 1

            gt_violated, gt_details = gt_result
            if gt_violated:
                gt_violations.append(
                    {
                        "dataset_position": current_position,
                        "sample_id": sample_id,
                        "details": gt_details,
                    }
                )
                continue

            occurred, details = prediction_result
            if occurred:
                violations.append(
                    {
                        "dataset_position": current_position,
                        "sample_id": sample_id,
                        "details": details,
                        "prediction": prediction.to(torch.uint8),
                    }
                )

evaluated_samples = len(dataset) - len(gt_violations)
print(
    f"Found {len(violations)}/{evaluated_samples} non-annular predictions "
    f"({len(violations) / max(evaluated_samples, 1):.2%})"
)
print(
    f"Excluded {len(gt_violations)}/{len(dataset)} samples whose resized "
    "ground truth is not annular"
)

# %%
violation_summary = pl.DataFrame(
    {
        "violation_index": range(len(violations)),
        "sample_id": [record["sample_id"] for record in violations],
        "details": [" | ".join(record["details"]) for record in violations],
    }
)
display(violation_summary)

# %%
violation_summary = violation_summary.with_columns(
    open_ring = pl.col("details").str.contains("Open-ring/missing-hole violation")
)
display(violation_summary)

print(violation_summary.select(pl.col("open_ring").sum()))

open_ring_sample = violation_summary.filter(pl.col("open_ring"))
display(open_ring_sample)
open_ring_indices = open_ring_sample["violation_index"].to_list()

# %% [markdown]
"""
## View violations

Pass one index or a list of indices from `violation_summary` to
`show_violations`. Predictions are stored during the scan, so the viewer shows
the exact mask that was classified rather than rerunning the model.
"""

# %%
cmap = ListedColormap(
    [
        dataset.label_schema.colors[index]
        for index in sorted(dataset.label_schema.colors)
    ]
)

def show_violations(
    violation_indices: int | list[int],
    rows_per_figure: int = 3,
) -> None:
    if isinstance(violation_indices, int):
        violation_indices = [violation_indices]
    if not violation_indices:
        return
    if rows_per_figure <= 0:
        raise ValueError("rows_per_figure must be positive.")
    if any(index < 0 or index >= len(violations) for index in violation_indices):
        raise IndexError(f"Violation index must be in [0, {len(violations) - 1}].")

    for page_start in range(0, len(violation_indices), rows_per_figure):
        page_indices = violation_indices[page_start : page_start + rows_per_figure]
        fig, axes = plt.subplots(
            len(page_indices),
            3,
            figsize=(13.5, 3.7 * len(page_indices)),
            squeeze=False,
            layout="constrained",
            gridspec_kw={"wspace": 0.02, "hspace": 0.08},
        )

        for row, violation_index in enumerate(page_indices):
            record = violations[violation_index]
            sample = dataset[record["dataset_position"]]
            prediction = record["prediction"]

            occurred, details = annularity_counter.classify(
                prediction.unsqueeze(0)
            )[0]
            if not occurred or details != record["details"]:
                raise RuntimeError(
                    "Annularity result changed for violation index "
                    f"{violation_index} ({record['sample_id']})."
                )

            image = sample["image"].squeeze()
            gt_overlay = create_segmentation_overlay(
                sample["image"],
                sample["target_labels"],
                cmap,
                alpha=0.7,
            )
            prediction_overlay = create_segmentation_overlay(
                sample["image"],
                prediction,
                cmap,
                alpha=1,
            )

            axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(
                f"{record['sample_id']} (violation #{violation_index})",
                fontsize=10,
            )
            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].set_title("Ground-truth myocardium", fontsize=10)
            axes[row, 2].imshow(prediction_overlay)
            details_text = fill(" • ".join(record["details"]), width=52)
            axes[row, 2].set_title(
                f"Non-annular prediction\n{details_text}",
                fontsize=9,
            )

            for axis in axes[row]:
                axis.axis("off")

        plt.show()
        plt.close(fig)

# %%
if violations:
    # show_violations(list(range(min(10, len(violations)))))
    show_violations(open_ring_indices)
else:
    print("No non-annular predictions to display.")

# %%
# Inspect any specific rows from violation_summary, for example:
# show_violations([10, 25, 40])
