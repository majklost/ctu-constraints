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
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from constraints import REPO_ROOT, get_experiment_folder
from constraints.datatools.label_schema import LabelSchema
from constraints.generators.factories import (
    get_source_config,
    preview_artificial_sample,
)
from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    SavedLayer,
    bubble_cavity_layer_backup,
    power_layer_backup,
    wall_attenuation_layer_backup,
)
from constraints.generators.recipe_backups import (
    DeformationBackup,
    RigidBackup,
    SavedDeformation,
    SavedRigid,
)
from constraints.generators.recipes import Recipe
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    DeformationConfig,
    DeformationRejectionConfig,
    FloatRange,
    NoiseConfig,
    RigidConfig,
)
from constraints.losses_metrics.constraint_function import (
    does_violation_occur_with_wall,
)

FOLDER = get_experiment_folder("ex5/fake")
print(FOLDER.absolute)

# %%
fake_plaque_range = PowerPlaqueSamplingRanges(
    angle_rad=FloatRange(np.pi / 3, 2 * np.pi - 2 * np.pi / 3),
    angular_width_rad=FloatRange.fixed(np.pi / 5),
    inward_depth_fraction=FloatRange(0.2, 0.3),
    shape_power=FloatRange.fixed(0.5),
    wall_depth_fraction=FloatRange.fixed(0),
    offset_px_lumen=FloatRange.fixed(-3),
)
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
dc = DeformationConfig()
nc = NoiseConfig(speckle_std=0.7)
rc = RigidConfig(dx=FloatRange.fixed(0), dy=FloatRange.fixed(0))

# %%
recipe = Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(
            backup=power_layer_backup(
                (fake_plaque_range,) * 2,
                seed=53,
                target_class=ArteryClass.LUMEN,
                appearance=AppearanceKind.PLAQUE,
            ),
        ),
        SavedLayer(backup=power_layer_backup((plaque_range1, plaque_range2), seed=25)),
    ),
    deformation=SavedDeformation(backup=DeformationBackup(dc, seed=27)),
    rigid=SavedRigid(backup=RigidBackup(rc, seed=52)),
    noise=nc,
)
sample = preview_artificial_sample(
    recipe=recipe,
    sample_index=np.random.randint(0, 100),
)
label_map = sample.target_labels
plt.imshow(label_map)
plt.show()
plt.imshow(sample.image, cmap="gray")
plt.show()
print(
    does_violation_occur_with_wall(torch.from_numpy(label_map), LabelSchema.as_artery())
)


# %%
# Give every generated artifact a stable name only after the visual tuning is done.
# Fresh names keep this trial independent of older datasets on the cluster.
cluster_recipe = recipe.with_names(
    layers={
        0: "fake-similar-offset-minus-3-v1",
        1: "two-real-separated-v1",
    },
    deformation="default-deformation-v1",
    rigid="rotation-only-v1",
)
recipe_path = REPO_ROOT / "recipes/artificial/tworeal_fake_similar_offset_minus3.json"
cluster_recipe.save_json(recipe_path)
print(recipe_path)

# %% [markdown]
# The JSON above is the portable cluster input. It contains the relative source
# path and all backups needed to recreate missing layer, deformation, and rigid
# artifacts. On the cluster, from the repository root, run:
#
# ```bash
# .venv/bin/python scripts/ensure_recipe.py \
#     recipes/artificial/tworeal_fake_similar_offset_minus3.json \
#     --device cuda
# ```
# Then pass that same JSON to the experiment, for example:
#
# ```bash
# .venv/bin/python experiments/ex5/initial_decoupled_new.py \
#     --recipe recipes/artificial/tworeal_fake_similar_offset_minus3.json \
#     --mode UNET --modality deformed
# ```
#
# The base source (`data/artificial/samples5000`) must already exist. `ensure`
# first checks every artifact, then creates missing ones. If a name already has a
# different definition it fails without changing anything; use a new name for a
# new trial. `--overwrite` is reserved for intentionally replacing all conflicting
# named artifacts that have backups.

# %%
# The same validation/materialization can be tested locally before committing.
# report = cluster_recipe.ensure(device="cpu")
# print(report)

# %% [markdown]
# ## Figure for report

# %%
from constraints.visu.helpers import create_segmentation_overlay
from constraints.utils import get_repo_root
recipes =[
    Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(backup=power_layer_backup((plaque_range1, plaque_range2), seed=25)),
    ),

),
Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(backup=power_layer_backup((plaque_range1, plaque_range2), seed=25)),
    ),
    deformation=SavedDeformation(backup=DeformationBackup(dc, seed=27)),

),
Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(backup=power_layer_backup((plaque_range1, plaque_range2), seed=25)),
    ),
    deformation=SavedDeformation(backup=DeformationBackup(dc, seed=27)),
    rigid=SavedRigid(backup=RigidBackup(rc, seed=52)),
),
Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(backup=power_layer_backup((plaque_range1, plaque_range2), seed=25)),
    ),
    deformation=SavedDeformation(backup=DeformationBackup(dc, seed=27)),
    rigid=SavedRigid(backup=RigidBackup(rc, seed=52)),
    noise=nc,
)
] 
# sample = preview_artificial_sample(
#     recipe=recipe,
#     sample_index=5,
# )
samples = list(map(lambda x: preview_artificial_sample(recipe=x, sample_index=5),recipes))
cmap = ListedColormap(
    [LabelSchema.as_artery().colors[i] for i in sorted(LabelSchema.as_artery().colors) ]
)
fig, axes = plt.subplots(2,4, figsize=(18,8))
for col, sample_index in enumerate(range(len(samples))):
    sample = samples[sample_index]
    axes[0,col].imshow(sample.image.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0,col].set_title(f"Sample {sample_index}")
    axes[1,col].imshow(
        create_segmentation_overlay(sample.image,sample.target_labels,cmap,1),
        cmap=cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    axes[1,col].set_title(f"target labels")
    for axis in axes[:,col]:
        axis.axis("off")
fig.tight_layout()
fig.savefig(get_repo_root()/"reports/august26/images"/f"dataset.png")



# %% [markdown]
# # Bites and holes
# Start with one large plaque produced by the existing power-profile generator.
# Random circular bubbles are then intersected with the plaque:
#
# - a **bite** touches the original main lumen and is labelled as lumen;
# - a **hole** is fully surrounded by plaque and remains labelled as plaque;
# - both are rendered with the same lumen-like appearance;
# - enclosed bubbles may overlap one another;
# - every hole retains at least five plaque pixels between it and final lumen.
#
# Thus local appearance does not determine the target class. The network has to
# learn which individual bubble directly touches the original visible lumen.
# Sampling is best-effort: a difficult sample is kept even when fewer than the
# requested number of one kind can be placed.
#
# If the U-Net predicts an enclosed hole as lumen, it creates a second lumen
# component and therefore a topological violation. This is intended to be part
# of the training distribution, not only an out-of-distribution test.
#
# Final plaque and bubble appearances are rendered sharply first. Symmetric
# Gaussian smoothing is then applied to the image only, so ambiguity is confined
# to a narrow boundary rather than leaving plaque-like stripes inside GT lumen.

# %%
cavity_plaque_range = PowerPlaqueSamplingRanges(
    angle_rad=FloatRange(-np.pi / 5, np.pi / 5),
    angular_width_rad=FloatRange(2 * np.pi / 3, 5 * np.pi / 6),
    inward_depth_fraction=FloatRange(0.45, 0.65),
    wall_depth_fraction=FloatRange(0.05, 0.25),
    shape_power=FloatRange(0.4, 0.8),
)

# This is a complete recipe: the custom layer is generated before the same
# deformation, rotation, and image noise used by the other artificial datasets.
cavity_recipe = Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(
            backup=bubble_cavity_layer_backup(
                cavity_plaque_range,
                seed=81,
                bubbles_per_kind=4,
                radius_px=FloatRange(6.0, 13.0),
                minimum_plaque_separation_px=5,
                plaque_blur_sigma_px=1.5,
                bubble_blur_sigma_px=1.5,
            )
        ),
    ),
    deformation=SavedDeformation(backup=DeformationBackup(dc, seed=27)),
    rigid=SavedRigid(backup=RigidBackup(rc, seed=52)),
    noise=nc,
)

# %%
cavity_source_config = get_source_config(cavity_recipe.resolve_source_root())
bubble_index = np.random.randint(0, cavity_source_config.num_elements)
bubble_sample = preview_artificial_sample(
    recipe=cavity_recipe,
    sample_index=bubble_index,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(bubble_sample.target_labels, vmin=0, vmax=3)
axes[0].set_title("Deformed target labels")
axes[1].imshow(bubble_sample.image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Smoothed plaque + bubbles + noise")
for axis in axes:
    axis.axis("off")
plt.show()

print(
    does_violation_occur_with_wall(
        torch.from_numpy(bubble_sample.target_labels), LabelSchema.as_artery()
    )
)

# %%
cavity_cluster_recipe = cavity_recipe.with_names(
    layers={0: "bubble-cavities-smoothed-topology-safe-v3"},
    deformation="default-deformation-v1",
    rigid="rotation-only-v1",
)
cavity_recipe_path = (
    REPO_ROOT / "recipes/artificial/bubble_cavities_overlap_gradient.json"
)
cavity_cluster_recipe.save_json(cavity_recipe_path)
print(cavity_recipe_path)

# %% [markdown]
# The recipe above is ready for `scripts/ensure_recipe.py`. Bubble placement does
# not abort materialization when one requested category cannot be filled; that
# sample simply contains fewer bubbles of that kind. Before training, inspect the
# achieved distribution and verify that deformed ground truths remain valid.
#
# ```bash
# .venv/bin/python scripts/ensure_recipe.py \
#     recipes/artificial/bubble_cavities_overlap_gradient.json \
#     --device cuda
#
# .venv/bin/python experiments/ex5/initial_decoupled_new.py \
#     --recipe recipes/artificial/bubble_cavities_overlap_gradient.json \
#     --mode UNET --modality deformed
# ```

# %% [markdown]
# # Wall Attenuation
# Let plaque grow into the boundary, leaving, for example, only five boundary
# pixels in the target (so the boundary is still an annular ring).
# Make the plaque larger in the grayscale image, reaching the outer
# edge. Use an intensity gradient to make the plaque/boundary transition unclear.
# There should be many plaques with different growth into the boundary, but the
# structure has to remain annular.
#
# In grayscale, the gradient should start inside the plaque. Its length differs
# between plaques and slowly brings the appearance to background intensity. The
# pixels corresponding to boundary will therefore be in the lower part of the
# gradient or black and indistinguishable from background.
#
# In GT, there is boundary behind every plaque. Each plaque independently uses
# $$\text{wall\_residual\_px} \in \{4, 5, 8, 12\}\text{ px},$$
# so the boundary remains an annular structure. In the image, the plaque extends
# farther than its target and fades to background intensity at the artery's
# exterior. The sampled fade begins inside the labelled plaque for the shorter
# residual-wall choices.
# Candidate deformation fields are stress-tested with a four-pixel annular wall
# and rejected if nearest-neighbor resampling opens a gap. This preserves the
# difficult four-pixel GT cases without thickening the other residual walls.
# The U-Net may omit the remaining boundary and connect plaque directly to the
# background, producing a topological violation.
#

# %%
attenuation_plaque_range = PowerPlaqueSamplingRanges(
    angle_rad=FloatRange(-np.pi, np.pi),
    angular_width_rad=FloatRange(np.pi / 6, np.pi / 3),
    inward_depth_fraction=FloatRange(0.15, 0.35),
    # Ignored by this layer: exact residual wall widths determine GT wall depth.
    wall_depth_fraction=FloatRange.fixed(0),
    shape_power=FloatRange(0.4, 0.8),
)

attenuation_recipe = Recipe(
    source="artificial/samples5000",
    layers=(
        SavedLayer(
            backup=wall_attenuation_layer_backup(
                (attenuation_plaque_range,) * 4,
                seed=93,
                residual_wall_px=(4, 5, 8, 12),
                gradient_length_px=FloatRange(10, 24),
            )
        ),
    ),
    deformation=SavedDeformation(
        backup=DeformationBackup(
            dc,
            rejection=DeformationRejectionConfig(
                preserved_wall_thickness_px=4,
                max_attempts=100,
            ),
            seed=27,
        )
    ),
    rigid=SavedRigid(backup=RigidBackup(rc, seed=52)),
    noise=nc,
)

# %%
attenuation_source_config = get_source_config(attenuation_recipe.resolve_source_root())
attenuation_index = np.random.randint(0, attenuation_source_config.num_elements)
attenuation_sample = preview_artificial_sample(
    recipe=attenuation_recipe,
    sample_index=attenuation_index,
)

attenuation_classes = ("background", "boundary", "lumen", "plaque")
attenuation_colors = ("black", "tab:orange", "tab:blue", "tab:red")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(
    attenuation_sample.target_labels,
    cmap=ListedColormap(attenuation_colors),
    vmin=-0.5,
    vmax=3.5,
)
axes[0].set_title("GT: residual wall remains")
axes[0].legend(
    handles=[
        Patch(color=color, label=name)
        for name, color in zip(attenuation_classes, attenuation_colors, strict=True)
    ],
    loc="lower right",
)
axes[1].imshow(attenuation_sample.image, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Image: plaque fades into background")
for axis in axes:
    axis.axis("off")
plt.show()

print(
    does_violation_occur_with_wall(
        torch.from_numpy(attenuation_sample.target_labels),
        LabelSchema.as_artery(),
    )
)

# %%
attenuation_cluster_recipe = attenuation_recipe.with_names(
    layers={0: "wall-attenuation-four-plaques-v2"},
    deformation="wall-preserving-4px-deformation-v1",
    rigid="rotation-only-v1",
)
attenuation_recipe_path = REPO_ROOT / "recipes/artificial/wall_attenuation.json"
attenuation_cluster_recipe.save_json(attenuation_recipe_path)
print(attenuation_recipe_path)

# %% [markdown]
# Materialize and train this experiment on the cluster with:
#
# ```bash
# .venv/bin/python scripts/ensure_recipe.py \
#     recipes/artificial/wall_attenuation.json \
#     --device cuda
#
# .venv/bin/python experiments/ex5/initial_decoupled_new.py \
#     --recipe recipes/artificial/wall_attenuation.json \
#     --mode UNET --modality deformed
# ```
