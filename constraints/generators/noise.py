"""Deterministic image-noise generation for artificial samples."""

import numpy as np
from numpy.typing import NDArray

from .types import NoiseConfig


def apply_speckle_noise(
    image: np.ndarray,
    config: NoiseConfig,
    *,
    sample_index: int,
) -> NDArray[np.float32]:
    """Apply deterministic Gaussian speckle to an image.

    The result is stable for a given configuration and source sample index and
    does not read or mutate NumPy's global random state. Multiplicative mode
    scales noise by image intensity; additive mode uses a constant noise scale.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("speckle-noise image must have shape [H, W]")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("speckle-noise image must have a numeric dtype")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("sample_index must be a non-negative integer")

    result = image.astype(np.float32, copy=True)
    if config.speckle_std == 0:
        return result

    seed = np.random.SeedSequence([0x535045434B4C45, config.seed, sample_index])
    rng = np.random.default_rng(seed)
    speckle = rng.standard_normal(result.shape, dtype=np.float32)
    if config.speckle_mode == "multiplicative":
        result += result * speckle * config.speckle_std
    else:
        result += speckle * config.speckle_std
    np.clip(result, 0.0, 1.0, out=result)
    return result
