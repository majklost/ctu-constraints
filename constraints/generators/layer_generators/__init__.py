"""Public API for procedural and persisted composition layers."""

from .artery import create_empty_artery
from .bubble_cavities import bubble_cavity_layer_backup
from .power import (
    PowerPlaqueParameters,
    PowerPlaqueSample,
    PowerPlaqueSamplingRanges,
    create_power_plaque,
    create_power_plaque_mask,
    power_layer_backup,
    sample_power_plaque_mask,
)
from .rasterizer import CyclicRasterizer, PlaqueSpec
from .registry import (
    load_layer_collection,
    materialize_layer_collection,
    normalize_layer_output,
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
from .wall_attenuation import wall_attenuation_layer_backup

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
    "bubble_cavity_layer_backup",
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
    "wall_attenuation_layer_backup",
]
