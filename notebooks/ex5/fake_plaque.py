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
# Make the plaque larger in the grayscale image, potentially reaching the outer
# edge. Use an intensity gradient to make the plaque/boundary transition unclear.
#
# The U-Net may omit the remaining boundary and connect plaque directly to the
# background, producing a topological violation.
#

# %% [markdown]
#
