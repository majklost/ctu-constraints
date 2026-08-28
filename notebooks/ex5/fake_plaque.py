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
from constraints.generators.factories import preview_artificial_sample
from constraints.generators.recipe_backups import (
    DeformationBackup,
    PowerPlaqueBackup,
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
    PowerPlaqueSamplingRanges,
    RigidConfig,
    SavedPlaque,
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
    plaques=(
        SavedPlaque(
            target_class=ArteryClass.LUMEN,
            appearance=AppearanceKind.PLAQUE,
            backup=PowerPlaqueBackup((fake_plaque_range,) * 2, seed=53),
        ),
        SavedPlaque(
            backup=PowerPlaqueBackup((plaque_range1, plaque_range2), seed=25)
        ),
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
    plaques={
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
# path and all backups needed to recreate missing plaque, deformation, and rigid
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
