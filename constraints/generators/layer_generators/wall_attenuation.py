"""Plaques whose image attenuates through a wall preserved in ground truth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

import numpy as np

from ..recipe_backups import LayerBackup
from ..rendering import DEFAULT_CLASS_INTENSITIES
from ..types import AppearanceKind, ArteryClass, FloatRange
from .power import PowerPlaqueParameters, PowerPlaqueSamplingRanges
from .rasterizer import polar_grid
from .registry import register_layer_resolver
from .types import TRANSPARENT_LABEL, LayerPatch, LayerResolverContext

DEFAULT_GRADIENT_LENGTH_PX = FloatRange(8.0, 20.0)


def wall_attenuation_layer_backup(
    plaque_ranges: PowerPlaqueSamplingRanges | tuple[PowerPlaqueSamplingRanges, ...],
    *,
    seed: int,
    residual_wall_px: Sequence[float] = (4.0, 5.0, 8.0, 12.0),
    gradient_length_px: FloatRange = DEFAULT_GRADIENT_LENGTH_PX,
) -> LayerBackup:
    """Describe plaques with a preserved GT wall and misleading image extent.

    One residual-wall value and one gradient length are sampled independently
    for each plaque. The label plaque stops before the vessel exterior, whereas
    its grayscale appearance fades from plaque intensity to background intensity
    and reaches the exterior.
    """
    ranges = (
        (plaque_ranges,)
        if isinstance(plaque_ranges, PowerPlaqueSamplingRanges)
        else plaque_ranges
    )
    if not ranges:
        raise ValueError("wall attenuation requires at least one plaque range")
    residuals = tuple(float(value) for value in residual_wall_px)
    if not residuals or any(not isfinite(value) or value < 0 for value in residuals):
        raise ValueError("residual_wall_px must contain finite non-negative values")
    if gradient_length_px.minimum <= 0:
        raise ValueError("gradient_length_px must be positive")
    return LayerBackup(
        "wall-attenuation-v1",
        {
            "seed": seed,
            "plaque_sampling": [item.to_dict() for item in ranges],
            "residual_wall_px": list(residuals),
            "gradient_length_px": gradient_length_px.to_dict(),
        },
    )


@register_layer_resolver("wall-attenuation-v1")
def _resolve_wall_attenuation(
    context: LayerResolverContext, params: Mapping[str, Any]
) -> LayerPatch:
    values = _parse_params(params)
    artery = context.source_config.empty_artery
    wall_thickness = artery.wall_thickness_px
    if any(value > wall_thickness for value in values["residual_wall_px"]):
        raise ValueError("residual wall cannot exceed source wall thickness")

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [values["seed"], context.sample_index, 0x415454454E554154]
        )
    )
    parameters: list[PowerPlaqueParameters] = []
    gradient_lengths: list[float] = []
    for sampling in values["plaque_ranges"]:
        sampled = sampling.sample(
            1,
            lumen_radius_px=artery.lumen_radius_px,
            wall_thickness_px=wall_thickness,
            rng=rng,
        )[0]
        residual = float(rng.choice(values["residual_wall_px"]))
        wall_depth = wall_thickness - residual - sampled.offset_px_lumen
        if wall_depth < 0:
            raise ValueError(
                "residual wall and lumen offset leave no non-negative wall depth"
            )
        parameters.append(replace(sampled, wall_depth_px=wall_depth))
        gradient_lengths.append(values["gradient_length_px"].sample(rng))

    radius, angle = polar_grid(artery.image_size)
    label_mask = np.zeros(artery.image_size, dtype=bool)
    image_alpha = np.zeros(artery.image_size, dtype=np.float64)
    lumen_radius = artery.lumen_radius_px
    outer_artery = lumen_radius + wall_thickness

    for plaque, gradient_length in zip(parameters, gradient_lengths, strict=True):
        delta = np.arctan2(
            np.sin(angle - plaque.angle_rad), np.cos(angle - plaque.angle_rad)
        )
        support = np.abs(delta) <= plaque.angular_width_rad / 2
        normalized = 2 * delta / plaque.angular_width_rad
        profile = np.clip(1 - normalized**2, 0.0, None) ** plaque.shape_power
        base_radius = lumen_radius + plaque.offset_px_lumen
        inner = base_radius - plaque.inward_depth_px * profile
        label_outer = base_radius + plaque.wall_depth_px * profile
        visual_outer = base_radius + (outer_artery - base_radius) * profile

        label_mask |= support & (radius >= inner) & (radius <= label_outer)
        visual_mask = support & (radius >= inner) & (radius <= visual_outer)
        alpha = np.clip((visual_outer - radius) / gradient_length, 0.0, 1.0)
        image_alpha = np.maximum(image_alpha, np.where(visual_mask, alpha, 0.0))

    labels = np.full(artery.image_size, TRANSPARENT_LABEL, dtype=np.int8)
    labels[label_mask] = ArteryClass.PLAQUE
    image = np.full(artery.image_size, np.nan, dtype=np.float32)
    visual_mask = image_alpha > 0
    plaque_intensity = DEFAULT_CLASS_INTENSITIES[AppearanceKind.PLAQUE]
    background_intensity = DEFAULT_CLASS_INTENSITIES[AppearanceKind.BACKGROUND]
    image[visual_mask] = background_intensity + image_alpha[visual_mask] * (
        plaque_intensity - background_intensity
    )
    return LayerPatch(labels, image)


def _parse_params(params: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "seed",
        "plaque_sampling",
        "residual_wall_px",
        "gradient_length_px",
    }
    if params.keys() != expected:
        raise ValueError("invalid wall-attenuation-v1 resolver params")
    try:
        seed = params["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        sampling = params["plaque_sampling"]
        residuals = params["residual_wall_px"]
        if not isinstance(sampling, list) or not sampling:
            raise ValueError("plaque_sampling must be a non-empty list")
        if not isinstance(residuals, list) or not residuals:
            raise ValueError("residual_wall_px must be a non-empty list")
        residual_values = tuple(float(value) for value in residuals)
        if any(not isfinite(value) or value < 0 for value in residual_values):
            raise ValueError("invalid residual wall value")
        gradient = FloatRange.from_dict(params["gradient_length_px"])
        if gradient.minimum <= 0:
            raise ValueError("gradient length must be positive")
        return {
            "seed": seed,
            "plaque_ranges": tuple(
                PowerPlaqueSamplingRanges.from_dict(item) for item in sampling
            ),
            "residual_wall_px": residual_values,
            "gradient_length_px": gradient,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid wall-attenuation-v1 resolver params") from error
