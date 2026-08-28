"""Resolver registry plus materialization of generated layer patches."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from ..recipe_backups import LayerBackup
from ..rendering import DEFAULT_CLASS_INTENSITIES
from ..types import AppearanceKind, ArteryClass, SourceConfig
from .power import PowerPlaqueSamplingRanges, sample_power_plaque_mask
from .types import (
    TRANSPARENT_LABEL,
    LayerCollection,
    LayerOutput,
    LayerPatch,
    LayerResolver,
    LayerResolverContext,
    MaskLayer,
)

_RESOLVERS: dict[str, LayerResolver] = {}


def register_layer_resolver(name: str):
    """Register a stable resolver name; re-registration supports autoreload."""
    if not isinstance(name, str) or not name:
        raise ValueError("resolver name must be a non-empty string")

    def register(function: LayerResolver) -> LayerResolver:
        _RESOLVERS[name] = function
        return function

    return register


def resolve_layer_patch(
    backup: LayerBackup,
    source_root: Path,
    source_config: SourceConfig,
    sample_index: int,
) -> LayerPatch:
    try:
        resolver = _RESOLVERS[backup.resolver]
    except KeyError as error:
        raise ValueError(f"unknown layer resolver: {backup.resolver}") from error
    output = resolver(
        LayerResolverContext(Path(source_root), source_config, sample_index),
        backup.params,
    )
    return normalize_layer_output(output)


def normalize_layer_output(output: LayerOutput) -> LayerPatch:
    if isinstance(output, LayerPatch):
        return output
    if not isinstance(output, MaskLayer):
        raise TypeError("layer resolver must return LayerPatch or MaskLayer")
    labels = np.full(output.mask.shape, TRANSPARENT_LABEL, dtype=np.int8)
    labels[output.mask] = output.target_class
    image = np.full(output.mask.shape, np.nan, dtype=np.float32)
    appearance = output.appearance or AppearanceKind(output.target_class.value)
    image[output.mask] = DEFAULT_CLASS_INTENSITIES[appearance]
    return LayerPatch(labels, image)


def power_layer_backup(
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
    lumen_radius_px: float | None = None,
    target_class: ArteryClass = ArteryClass.PLAQUE,
    appearance: AppearanceKind | None = None,
) -> LayerBackup:
    ranges = (PowerPlaqueSamplingRanges(),) if ranges is None else ranges
    ranges = (ranges,) if isinstance(ranges, PowerPlaqueSamplingRanges) else ranges
    if not ranges:
        raise ValueError("power layer requires at least one sampling range")
    groups: list[dict[str, Any]] = []
    for item in ranges:
        sampling = item.to_dict()
        if groups and groups[-1]["sampling"] == sampling:
            groups[-1]["count"] += 1
        else:
            groups.append({"count": 1, "sampling": sampling})
    return LayerBackup(
        "power-v2",
        {
            "seed": seed,
            "lumen_radius_px": lumen_radius_px,
            "ranges": groups,
            "target_class": ArteryClass(target_class).name.lower(),
            "appearance": (
                None if appearance is None else AppearanceKind(appearance).name.lower()
            ),
        },
    )


@register_layer_resolver("power-v2")
def _resolve_power(
    context: LayerResolverContext, params: Mapping[str, Any]
) -> MaskLayer:
    expected = {
        "seed",
        "lumen_radius_px",
        "ranges",
        "target_class",
        "appearance",
    }
    if params.keys() != expected or not isinstance(params["ranges"], list):
        raise ValueError("invalid power-v2 resolver params")
    seed = params["seed"]
    lumen_radius_px = params["lumen_radius_px"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("power-v2 seed must be a non-negative integer")
    if lumen_radius_px is not None and (
        not isinstance(lumen_radius_px, (int, float))
        or not isfinite(lumen_radius_px)
        or lumen_radius_px <= 0
    ):
        raise ValueError("power-v2 lumen_radius_px must be positive")
    try:
        ranges = tuple(
            sampling
            for group in params["ranges"]
            for sampling in _power_range_group(group)
        )
        target_class = ArteryClass[params["target_class"].upper()]
        appearance = (
            None
            if params["appearance"] is None
            else AppearanceKind[params["appearance"].upper()]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid power-v2 resolver params") from error
    sample = sample_power_plaque_mask(
        context.source_config,
        ranges,
        seed=seed,
        sample_index=context.sample_index,
        lumen_radius_px=lumen_radius_px,
    )
    return MaskLayer(sample.mask, target_class, appearance)


def _power_range_group(value: Any) -> tuple[PowerPlaqueSamplingRanges, ...]:
    if (
        not isinstance(value, dict)
        or value.keys() != {"count", "sampling"}
        or isinstance(value["count"], bool)
        or not isinstance(value["count"], int)
        or value["count"] <= 0
    ):
        raise ValueError("invalid power-v2 range group")
    return (PowerPlaqueSamplingRanges.from_dict(value["sampling"]),) * value["count"]


def materialize_layer_collection(
    source_root: Path,
    name: str,
    source_config: SourceConfig,
    backup: LayerBackup,
) -> Path:
    folder = Path(source_root) / "layers" / name
    if folder.exists():
        raise FileExistsError(f"layer collection already exists: {name}")
    folder.mkdir(parents=True)
    labels_tmp = folder / ".labels.npy.tmp"
    image_tmp = folder / ".image.npy.tmp"
    shape = (source_config.num_elements, *source_config.empty_artery.image_size)
    labels = np.lib.format.open_memmap(
        labels_tmp, mode="w+", dtype=np.int8, shape=shape
    )
    image = np.lib.format.open_memmap(
        image_tmp, mode="w+", dtype=np.float32, shape=shape
    )
    try:
        for index in range(source_config.num_elements):
            patch = resolve_layer_patch(backup, source_root, source_config, index)
            if patch.labels.shape != shape[1:]:
                raise ValueError("resolved layer shape does not match source")
            labels[index], image[index] = patch.labels, patch.image
        labels.flush()
        image.flush()
        labels_tmp.replace(folder / "labels.npy")
        image_tmp.replace(folder / "image.npy")
    except BaseException:
        shutil.rmtree(folder)
        raise
    finally:
        del labels, image
    return folder


def load_layer_collection(
    source_root: Path, name: str, source_config: SourceConfig
) -> LayerCollection:
    if not name or Path(name).name != name:
        raise ValueError("layer collection name must be a filename component")
    folder = Path(source_root) / "layers" / name
    labels = np.load(folder / "labels.npy", mmap_mode="r")
    image = np.load(folder / "image.npy", mmap_mode="r")
    expected = (source_config.num_elements, *source_config.empty_artery.image_size)
    if labels.shape != expected or labels.dtype != np.int8:
        raise ValueError(f"invalid labels in layer collection {name!r}")
    if image.shape != expected or image.dtype != np.float32:
        raise ValueError(f"invalid image in layer collection {name!r}")
    return LayerCollection(labels, image)
