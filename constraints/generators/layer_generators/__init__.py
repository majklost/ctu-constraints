"""Public API for procedural and persisted composition layers."""

from .artery import create_empty_artery
from .power import (
    PowerPlaqueParameters,
    PowerPlaqueSample,
    PowerPlaqueSamplingRanges,
    create_power_plaque,
    create_power_plaque_mask,
    sample_power_plaque_mask,
)
from .rasterizer import CyclicRasterizer, PlaqueSpec
from .registry import (
    load_layer_collection,
    materialize_layer_collection,
    normalize_layer_output,
    power_layer_backup,
    register_layer_resolver,
    resolve_layer_patch,
)
from .types import (
    TRANSPARENT_LABEL,
    LayerCollection,
    LayerGenerator,
    LayerOutput,
    LayerPatch,
    LayerResolverContext,
    MaskLayer,
    SavedLayer,
)

__all__ = [
    "CyclicRasterizer",
    "LayerCollection",
    "LayerGenerator",
    "LayerOutput",
    "LayerPatch",
    "LayerResolverContext",
    "MaskLayer",
    "PlaqueSpec",
    "PowerPlaqueParameters",
    "PowerPlaqueSample",
    "PowerPlaqueSamplingRanges",
    "SavedLayer",
    "TRANSPARENT_LABEL",
    "create_empty_artery",
    "create_power_plaque",
    "create_power_plaque_mask",
    "load_layer_collection",
    "materialize_layer_collection",
    "normalize_layer_output",
    "power_layer_backup",
    "register_layer_resolver",
    "resolve_layer_patch",
    "sample_power_plaque_mask",
]
