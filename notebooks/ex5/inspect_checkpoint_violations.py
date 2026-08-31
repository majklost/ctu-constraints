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
# RUN_ID = "s0x3j84i"
RUN_ID = "ct6wtl0n"
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
cmap = ListedColormap(
    [
        dataset.label_schema.colors[index]
        for index in sorted(dataset.label_schema.colors)
    ]
)


def show_violation(violation_index: int) -> None:
    record = violations[violation_index]
    sample = dataset[record["dataset_position"]]
    prediction = record["prediction"]

    occurred, details = does_violation_occur_with_wall(
        prediction,
        dataset.label_schema,
    )
    if not occurred or tuple(details) != record["details"]:
        raise RuntimeError(
            "Constraint results changed since this prediction was scanned. "
            "Rerun the violation scan cell before displaying it."
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sample["image"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Image (sample {record['sample_id']})")

    mask_options = {
        "cmap": cmap,
        "vmin": 0,
        "vmax": dataset.label_schema.num_classes - 1,
        "interpolation": "nearest",
    }
    axes[1].imshow(sample["target_labels"], **mask_options)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, **mask_options)
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    fig.suptitle("\n".join(record["details"]), wrap=True)
    fig.tight_layout()
    plt.show()

# %%
if violations:
    show_violation(0)
else:
    print("No violating predictions to display.")

# %%
for i in range(min(50, len(violations))):
    show_violation(i)

# %%
