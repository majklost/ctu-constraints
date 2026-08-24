from typing import Any

_LEGACY_GENERATOR_EXPORTS = {
    "ROT_ONLY",
    "RigidSampleBounds",
    "ArteryGeneratorRigid",
    "ArteryGeneratorDeformed",
}

__all__ = sorted(_LEGACY_GENERATOR_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load legacy generators only when their public names are requested."""
    if name not in _LEGACY_GENERATOR_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import generators

    return getattr(generators, name)
