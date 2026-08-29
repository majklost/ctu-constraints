"""Plaques with smooth, connectivity-ambiguous bites and holes."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    generate_binary_structure,
    label,
)

from ..recipe_backups import LayerBackup
from ..rendering import (
    DEFAULT_CLASS_INTENSITIES,
    create_grayscale_image_from_label_mask,
)
from ..types import AppearanceKind, ArteryClass, FloatRange
from .artery import create_empty_artery
from .power import PowerPlaqueSamplingRanges, sample_power_plaque_mask
from .registry import register_layer_resolver
from .types import TRANSPARENT_LABEL, LayerPatch, LayerResolverContext


def bubble_cavity_layer_backup(
    plaque_range: PowerPlaqueSamplingRanges,
    *,
    seed: int,
    bubbles_per_kind: int = 3,
    radius_px: FloatRange | None = None,
    maximum_attempts: int = 2_000,
    minimum_plaque_separation_px: int = 5,
    plaque_blur_sigma_px: float = 1.5,
    bubble_blur_sigma_px: float = 1.5,
) -> LayerBackup:
    """Describe one valid plaque with lumen bites and enclosed visual holes."""
    radius_px = FloatRange(6.0, 13.0) if radius_px is None else radius_px
    return LayerBackup(
        "bubble-cavities-v2",
        {
            "seed": seed,
            "plaque_sampling": plaque_range.to_dict(),
            "bubbles_per_kind": bubbles_per_kind,
            "radius_px": radius_px.to_dict(),
            "maximum_attempts": maximum_attempts,
            "minimum_plaque_separation_px": minimum_plaque_separation_px,
            "plaque_blur_sigma_px": plaque_blur_sigma_px,
            "bubble_blur_sigma_px": bubble_blur_sigma_px,
        },
    )


@register_layer_resolver("bubble-cavities-v2")
def _resolve_bubble_cavities(
    context: LayerResolverContext, params: Mapping[str, Any]
) -> LayerPatch:
    values = _parse_params(params)
    artery = create_empty_artery(context.source_config.empty_artery)
    plaque = sample_power_plaque_mask(
        context.source_config,
        values["plaque_range"],
        seed=values["seed"],
        sample_index=context.sample_index,
    ).mask
    visible_lumen = (artery == ArteryClass.LUMEN) & ~plaque
    visible_wall = (artery == ArteryClass.BOUNDARY) & ~plaque
    connectivity = generate_binary_structure(2, 2)

    rng = np.random.default_rng(
        np.random.SeedSequence([values["seed"], context.sample_index, 0x425542424C45])
    )
    y_grid, x_grid = np.ogrid[: artery.shape[0], : artery.shape[1]]
    plaque_y, plaque_x = np.nonzero(plaque)

    # Bites are sampled first. A candidate is accepted only if carving it into
    # lumen leaves one connected plaque component that remains attached to wall.
    bites = np.zeros_like(plaque)
    bite_count = 0
    for _ in range(values["maximum_attempts"]):
        if bite_count >= values["bubbles_per_kind"]:
            break
        cavity, bubble = _sample_cavity(
            plaque, plaque_y, plaque_x, y_grid, x_grid, values["radius_px"], rng
        )
        cavity_edge = binary_dilation(cavity, structure=connectivity)
        if (
            not _is_usable_cavity(cavity, bubble)
            or np.any(cavity_edge & visible_wall)
            or not np.any(cavity_edge & visible_lumen)
        ):
            continue
        proposed_bites = bites | cavity
        if not _is_valid_remaining_plaque(
            plaque & ~proposed_bites, visible_wall, connectivity
        ):
            continue
        bites = proposed_bites
        bite_count += 1

    # Holes are sampled relative to the final lumen created above. They may
    # overlap one another, but a plaque barrier of the configured thickness is
    # required between every hole and lumen/bite target pixel.
    final_lumen = visible_lumen | bites
    remaining_plaque = plaque & ~bites
    holes = np.zeros_like(plaque)
    hole_count = 0
    for _ in range(values["maximum_attempts"]):
        if hole_count >= values["bubbles_per_kind"]:
            break
        cavity, bubble = _sample_cavity(
            remaining_plaque,
            plaque_y,
            plaque_x,
            y_grid,
            x_grid,
            values["radius_px"],
            rng,
        )
        cavity_edge = binary_dilation(cavity, structure=connectivity)
        clearance = _dilate(
            cavity,
            connectivity,
            values["minimum_plaque_separation_px"],
        )
        if (
            not _is_usable_cavity(cavity, bubble)
            or np.any(cavity_edge & visible_wall)
            or np.any(clearance & final_lumen)
        ):
            continue
        holes |= cavity
        hole_count += 1

    labels = np.full(artery.shape, TRANSPARENT_LABEL, dtype=np.int8)
    labels[plaque] = ArteryClass.PLAQUE
    labels[bites] = ArteryClass.LUMEN
    visual_bubbles = bites | holes
    image = _render_smoothed_image(
        artery,
        plaque,
        visual_bubbles,
        values["plaque_blur_sigma_px"],
        values["bubble_blur_sigma_px"],
    )
    return LayerPatch(labels, image)


def _sample_cavity(
    support: np.ndarray,
    plaque_y: np.ndarray,
    plaque_x: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray,
    radius_px: FloatRange,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    center_index = int(rng.integers(len(plaque_y)))
    center_y = plaque_y[center_index]
    center_x = plaque_x[center_index]
    radius = radius_px.sample(rng)
    bubble = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2 <= radius**2
    return bubble & support, bubble


def _is_usable_cavity(cavity: np.ndarray, bubble: np.ndarray) -> bool:
    return bool(cavity.sum() >= 50 and cavity.sum() >= 0.55 * bubble.sum())


def _is_valid_remaining_plaque(
    plaque: np.ndarray, visible_wall: np.ndarray, connectivity: np.ndarray
) -> bool:
    _, component_count = label(plaque, structure=connectivity)
    return bool(
        component_count == 1
        and np.any(binary_dilation(plaque, structure=connectivity) & visible_wall)
    )


def _dilate(mask: np.ndarray, connectivity: np.ndarray, iterations: int) -> np.ndarray:
    if iterations == 0:
        return mask
    return binary_dilation(mask, structure=connectivity, iterations=iterations)


def _render_smoothed_image(
    artery: np.ndarray,
    plaque: np.ndarray,
    visual_bubbles: np.ndarray,
    plaque_sigma_px: float,
    bubble_sigma_px: float,
) -> np.ndarray:
    """Render hard final appearances, then smooth their image boundaries."""
    image = create_grayscale_image_from_label_mask(
        artery, DEFAULT_CLASS_INTENSITIES
    )
    plaque_intensity = DEFAULT_CLASS_INTENSITIES[AppearanceKind.PLAQUE]
    lumen_intensity = DEFAULT_CLASS_INTENSITIES[AppearanceKind.LUMEN]
    plaque_alpha = _smoothed_mask(plaque, plaque_sigma_px)
    image += plaque_alpha * (plaque_intensity - image)
    bubble_alpha = _smoothed_mask(visual_bubbles, bubble_sigma_px)
    image += bubble_alpha * (lumen_intensity - image)
    return image.astype(np.float32, copy=False)


def _smoothed_mask(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    values = mask.astype(np.float32)
    if sigma_px == 0:
        return values
    return gaussian_filter(values, sigma=sigma_px, mode="nearest")


def _parse_params(params: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "seed",
        "plaque_sampling",
        "bubbles_per_kind",
        "radius_px",
        "maximum_attempts",
        "minimum_plaque_separation_px",
        "plaque_blur_sigma_px",
        "bubble_blur_sigma_px",
    }
    if params.keys() != expected:
        raise ValueError("invalid bubble-cavities-v2 resolver params")
    try:
        values = {
            "seed": _non_negative_integer(params["seed"], "seed"),
            "bubbles_per_kind": _non_negative_integer(
                params["bubbles_per_kind"], "bubbles_per_kind"
            ),
            "maximum_attempts": _positive_integer(
                params["maximum_attempts"], "maximum_attempts"
            ),
            "minimum_plaque_separation_px": _non_negative_integer(
                params["minimum_plaque_separation_px"],
                "minimum_plaque_separation_px",
            ),
            "plaque_range": PowerPlaqueSamplingRanges.from_dict(
                params["plaque_sampling"]
            ),
            "radius_px": FloatRange.from_dict(params["radius_px"]),
            "plaque_blur_sigma_px": _non_negative_finite(
                params["plaque_blur_sigma_px"], "plaque_blur_sigma_px"
            ),
            "bubble_blur_sigma_px": _non_negative_finite(
                params["bubble_blur_sigma_px"], "bubble_blur_sigma_px"
            ),
        }
        if values["radius_px"].minimum <= 0:
            raise ValueError("radius_px must be positive")
        return values
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid bubble-cavities-v2 resolver params") from error


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: Any, name: str) -> int:
    value = _non_negative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
