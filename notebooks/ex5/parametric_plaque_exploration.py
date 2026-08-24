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
# # Parametric plaque dataset pipeline
#
# The new artificial-data pipeline has three separate responsibilities:
#
# 1. generate anatomically valid circular templates with parametric plaques;
# 2. deform and cache clean samples;
# 3. apply a fixed per-sample rigid transform and image-only artifacts when the
#    cached dataset is loaded.
#
# Keeping anatomy, spatial transformations, and image corruption separate makes
# it possible to validate every anatomical mask while changing the image
# difficulty independently.
#
# ```text
# Parametric template
#         |
#         v
# Anatomical validation
#         |
#         v
# Smooth deformation
#         |
#         v
# Discretization and anatomical validation
#         |
#         v
# Clean image, mask and SDF saved to disk
#         |
#         v
# Fixed per-sample rigid transform during loading
#         |
#         v
# Fake plaques, dropout and speckle (image only)
#         |
#         v
# Training sample
# ```
#
# The mask and SDF are changed only by spatial transformations. Fake plaques,
# dropout rectangles, and speckle modify the image but never its target.

# %% [markdown]
# ## 1. Parametric vessel and plaque geometry
#
# NumPy is sufficient for the core generator. OpenCV can later rasterize
# irregular polygons, but the canonical representation should remain a set of
# parameters rather than a rasterized image. Shapely is useful only if more
# complicated Boolean polygon geometry is required. Rasterio adds little for
# non-geospatial medical masks.
#
# Describe a circular vessel in polar coordinates around its centre. Let
# `r_inner` be the lumen radius and `r_outer` the outside radius of the wall.
# Describe each plaque using:
#
# - angular position;
# - angular width;
# - protrusion into the lumen;
# - embedding depth into the wall;
# - optional shape exponent or boundary smoothness;
# - optional intensity parameters for later image synthesis.
#
# The approximate radial support of a plaque is:
#
# ```text
# r_inner - protrusion <= radius <= r_inner + wall_embed
# ```
#
# This makes attachment to the wall a construction rule. Plaque count, size,
# position, and angular separation can be sampled independently from configured
# ranges.
#
# ```python
# @dataclass(frozen=True)
# class PlaqueSpec:
#     angle: float
#     angular_width: float
#     protrusion: float
#     wall_embed: float
#     shape_power: float = 2.0
#
#
# @dataclass(frozen=True)
# class VesselTemplateSpec:
#     center_yx: tuple[float, float]
#     lumen_radius: float
#     wall_thickness: float
#     plaques: tuple[PlaqueSpec, ...]
# ```
#
# Prefer dimensions relative to the lumen radius where practical so the same
# distribution works at different output resolutions.

# %% [markdown]
# ## 2. Anatomical validity by construction
#
# Encode the assumptions checked by `does_violation_occur_with_wall` directly
# into the sampling bounds. For every significant plaque, require:
#
# - positive protrusion into the lumen, so it touches the lumen;
# - positive embedding into the wall, so it touches the wall;
# - an intact outer wall layer between the plaque and background;
# - pixel area greater than the constraint checker's `blob_threshold`;
# - enough residual lumen that it stays one connected component.
#
# Useful bounds are:
#
# ```text
# protrusion >= minimum protrusion
# protrusion <= maximum fraction of lumen radius
# wall_embed >= minimum wall embedding
# wall_embed <= wall thickness - outer wall safety margin
# plaque pixel area >= blob threshold
# ```
#
# Multiple plaques also need rules for minimum angular separation, overlap,
# maximum combined angular coverage, and maximum total lumen obstruction. If
# overlap is forbidden, sample plaque angles with bounded rejection sampling.
#
# The rasterized label map must contain exactly one label per pixel. Plaque
# pixels replace lumen/wall pixels according to the chosen priority, but may
# never overwrite the protected outer-wall margin.

# %% [markdown]
# ## 3. Validation checkpoints
#
# Construction rules reduce failures but do not replace validation. Rasterizing
# and interpolating continuous geometry can create one-pixel gaps or disconnected
# components.
#
# Run `does_violation_occur_with_wall` at two points:
#
# 1. after rasterizing the parametric template;
# 2. after deformation and conversion back to a discrete label map.
#
# Also verify:
#
# - expected shape and class IDs;
# - exactly one class at every pixel;
# - all required classes are present;
# - foreground is separated from the image boundary by a safety margin;
# - every requested plaque exceeds its configured minimum area.
#
# During development, save rejected parameter specifications and validation
# messages. Rejection should become rare after calibrating valid ranges. Always
# use a finite attempt limit and report infeasible configurations clearly.

# %% [markdown]
# ## 4. Deformation and cached clean samples
#
# Apply a smooth deformation to the valid template. Prefer an integrated,
# diffeomorphic field and monitor its Jacobian determinant because folding can
# change topology. After interpolation, convert the mask to discrete labels and
# run the anatomical validation again.
#
# Cache the expensive, foundational outputs:
#
# - clean deformed image;
# - discrete deformed target mask;
# - SDF recomputed from that final discrete mask;
# - sampled plaque/template parameters;
# - deformation parameters or field when needed for analysis;
# - template identifier or full specification.
#
# Recomputing the stored SDF after discretizing the deformed mask gives the best
# agreement between target labels and distances.

# %% [markdown]
# ## 5. Fixed per-sample rigid transformation
#
# Sample one rigid transform for each deformed sample and save:
#
# ```text
# rigid.npy, shape [N, 3]
# columns: angle_radians, dx_pixels, dy_pixels
# ```
#
# Store metadata defining angle and translation units, coordinate order,
# rotation direction, and transform direction. This prevents ambiguity when
# pixel translations are converted to the normalized coordinates expected by
# `affine_grid`.
#
# Use bounded rejection sampling against the actual deformed foreground mask:
#
# 1. sample `(angle, dx, dy)` from configured ranges;
# 2. transform the mask with nearest-neighbour interpolation;
# 3. reject if foreground is lost, touches the edge, or violates constraints;
# 4. fail clearly after a finite number of attempts.
#
# Testing the transformed mask is safer than using only its rotated bounding
# box. Require a two- or three-pixel border margin to account for image
# interpolation near the edge.
#
# During dataset loading apply the same stored transform to:
#
# - image with bilinear interpolation;
# - discrete target mask with nearest-neighbour interpolation;
# - SDF with bilinear interpolation.
#
# Bilinear SDF transformation is efficient and normally accurate for rigid
# transforms. An optional strict mode can recompute it from the transformed mask
# when exact raster consistency is more important than loading speed.

# %% [markdown]
# ## 6. Ordering of image-only artifacts
#
# Generate all image artifacts after the fixed rigid transform. At that point
# the anatomy has its final image location, so no newly exposed region lacks the
# expected image formation or noise.
#
# ### Fake plaques
#
# Fake plaques are distractors and modify only the image. Generate them using
# the transformed mask as a placement guide:
#
# 1. find the transformed wall/lumen interface or another configured region;
# 2. choose a deterministic location near it;
# 3. synthesize a structured low-frequency plaque-like blob;
# 4. blend it into the image;
# 5. leave the mask and SDF unchanged.
#
# This makes distractors spatially plausible without labeling them as anatomy.
# Difficulty can be calibrated by varying location, size, contrast, edge
# softness, and frequency content.
#
# ### Black dropout rectangles
#
# Add dropout after the rigid transform because it represents an image/probe
# coordinate suppression artifact. It obscures the final image without changing
# anatomical labels. Its location, size, count, and intensity should come from
# an explicit configuration.
#
# ### Speckle
#
# Add speckle last, after interpolation and structured artifacts. Adding it
# before the rigid transform would rotate and interpolate the noise, introduce
# artificial spatial correlation, and leave translated-in padding with a
# different noise distribution.
#
# The runtime order is therefore:
#
# ```text
# load clean image/mask/SDF
#     -> apply fixed rigid transform to all three
#     -> add fake plaque to image
#     -> add dropout rectangle to image
#     -> add speckle to image
# ```
#
# Fake plaques and dropout can be swapped if treated as independent acquisition
# effects. Speckle should remain last so it affects the final visible result.

# %% [markdown]
# ## 7. Runtime generation with fixed outputs
#
# Applying artifacts in `CachedArtificialDataset.__getitem__` does not require
# them to change between epochs. Derive independent deterministic seeds from the
# dataset seed, original sample index, and operation-specific stream:
#
# ```python
# seed = SeedSequence([dataset_seed, sample_index, artifact_stream])
# ```
#
# Suggested streams:
#
# ```text
# 0: rigid parameter generation
# 1: fake-plaque parameters
# 2: dropout parameters
# 3: speckle realization
# ```
#
# Separate streams prevent enabling one artifact from changing another and make
# results independent of DataLoader worker count and access order.
#
# Two deliberately different modes can be supported:
#
# - `fixed`: seed from dataset seed, sample index, and stream;
# - `per_epoch`: additionally include the epoch number.
#
# Use `fixed` for the controlled experiment matrix. Per-epoch corruption tests
# augmentation robustness and answers a different experimental question.

# %% [markdown]
# ## 8. Responsibility split
#
# Dataset-generation code owns:
#
# - template and plaque parameter sampling;
# - anatomical validation;
# - smooth deformation and post-deformation validation;
# - clean image synthesis and mask/SDF generation;
# - saving sampled anatomical parameters.
#
# A rigid-parameter preparation script owns:
#
# - sampling fixed rigid parameters for every cached sample;
# - rejection based on the transformed foreground margin;
# - saving the parameter array and convention metadata.
#
# `CachedArtificialDataset` and composable image transforms own:
#
# - applying the stored rigid transform consistently;
# - adding deterministic fake plaques;
# - adding deterministic dropout rectangles;
# - adding deterministic speckle;
# - returning the final image with its matching mask and SDF.
#
# Each artifact should be an independent callable with its own configuration and
# random generator. The dataset composes them in a declared order instead of
# accumulating artifact-specific branches in `__getitem__`.

# %% [markdown]
# ## 9. Calibration and tests
#
# Before producing the full experiment matrix:
#
# 1. sample many templates and measure rejection reasons;
# 2. visualize extrema of plaque count, size, separation, and protrusion;
# 3. sweep deformation magnitude, integration count, and spatial scale;
# 4. measure post-deformation anatomical-validity rate;
# 5. verify that every accepted rigid transform preserves the border margin;
# 6. compare transformed SDFs with SDFs recomputed from transformed masks;
# 7. verify deterministic reads across repeated calls and DataLoader workers;
# 8. run a pilot difficulty sweep for fake plaques and dropout.
#
# Important automated tests include:
#
# - templates satisfy all anatomical constraints across many seeds;
# - impossible ranges fail after the configured attempt limit;
# - nearest-neighbour mask transforms introduce no invalid class values;
# - image, mask, and SDF receive the identical rigid transform;
# - image artifacts never modify masks or SDFs;
# - random streams are deterministic and independent;
# - accepted transforms never place foreground on the image boundary.

# %% [markdown]
# ## 10. Power-profile plaque example
#
# The following example uses the implemented parametric renderer. The plaque is
# described by two radial functions around the lumen boundary. The inner
# function protrudes into the lumen, while the outer function embeds the plaque
# into the wall without reaching the background.
#
# For the power-profile factory, the normalized profile is
#
# ```text
# profile(u) = (1 - u^2)^shape_power,  |u| <= 1
# ```
#
# where `u=-1` and `u=1` are the angular endpoints. A power of `0.5` gives the
# square-root profile associated with an ellipse in local angular/radial
# coordinates. The comparison below keeps all dimensions fixed and changes only
# `shape_power`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from constraints.generators.parametrization import (
    PowerPlaqueParameters,
    create_power_plaque,
    render_artery,
)
from constraints.generators.types import ArterySpec

# %%
IMAGE_SIZE = (256, 256)
LUMEN_RADIUS_PX = 60.0
WALL_THICKNESS_PX = 12.0
PLAQUE_ANGLE_RAD = np.deg2rad(-60)
PLAQUE_ANGULAR_WIDTH_RAD = np.deg2rad(55)
INWARD_DEPTH_PX = 10.0
WALL_DEPTH_PX = 1

shape_powers = (0.25, 0.11, 1.0, 2.0)

label_maps = []
for shape_power in shape_powers:
    parameters = PowerPlaqueParameters(
        angle_rad=PLAQUE_ANGLE_RAD,
        angular_width_rad=PLAQUE_ANGULAR_WIDTH_RAD,
        inward_depth_px=INWARD_DEPTH_PX,
        wall_depth_px=WALL_DEPTH_PX,
        shape_power=shape_power,
    )
    plaque = create_power_plaque(parameters, lumen_radius_px=LUMEN_RADIUS_PX)
    label_maps.append(
        render_artery(
            ArterySpec(
                image_size=IMAGE_SIZE,
                lumen_radius_px=LUMEN_RADIUS_PX,
                wall_thickness_px=WALL_THICKNESS_PX,
                plaques=(plaque,),
            )
        )
    )

# %%
class_names = ("background", "wall", "lumen", "plaque")
class_colors = ("black", "tab:red", "tab:green", "tab:blue")
label_cmap = ListedColormap(class_colors)

fig, axes = plt.subplots(
    1, len(shape_powers), figsize=(16, 4), constrained_layout=True
)
for axis, labels, shape_power in zip(axes, label_maps, shape_powers, strict=True):
    axis.imshow(labels, cmap=label_cmap, vmin=0, vmax=3, interpolation="nearest")
    axis.set_title(f"shape_power = {shape_power}")
    axis.axis("off")

fig.legend(
    handles=[
        Patch(facecolor=color, label=name)
        for name, color in zip(class_names, class_colors, strict=True)
    ],
    loc="outside lower center",
    ncols=4,
)
plt.show()

# %% [markdown]
# The next plot shows the underlying one-dimensional profiles directly. Lower
# powers retain more depth near the angular endpoints and therefore produce a
# fuller plaque. Higher powers concentrate the plaque around its central angle.

# %%
u = np.linspace(-1, 1, 501)

fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
for shape_power in shape_powers:
    profile = np.clip(1 - u**2, 0, None) ** shape_power
    axis.plot(u, profile, label=f"shape_power = {shape_power}")

axis.set(
    xlabel="normalized angular displacement u",
    ylabel="fraction of maximum radial depth",
    title="Power plaque profiles",
    xlim=(-1, 1),
    ylim=(0, 1.05),
)
axis.grid(alpha=0.25)
axis.legend()
plt.show()

# %% [markdown]
# A fully custom plaque can use arbitrary vectorized boundary functions instead
# of the power-profile factory. Such a `PlaqueSpec` is a runtime object; save the
# parameters or recipe used to construct its functions rather than attempting
# to serialize the callables themselves.

