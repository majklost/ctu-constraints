import numpy as np
import pytest

from constraints.generators.noise import apply_speckle_noise
from constraints.generators.types import NoiseConfig


def test_speckle_noise_is_deterministic_without_global_rng_side_effects() -> None:
    image = np.full((16, 16), 0.5, dtype=np.float32)
    config = NoiseConfig(speckle_std=0.2, seed=123)
    expected_global_draws = np.random.RandomState(7).random_sample(2)
    np.random.seed(7)

    first = apply_speckle_noise(image, config, sample_index=4)
    second = apply_speckle_noise(image, config, sample_index=4)
    global_draws = np.random.random(2)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(global_draws, expected_global_draws)
    assert first.dtype == np.float32
    assert np.all((0 <= first) & (first <= 1))


def test_speckle_noise_varies_by_seed_and_sample_index() -> None:
    image = np.full((8, 8), 0.5, dtype=np.float32)

    baseline = apply_speckle_noise(
        image, NoiseConfig(speckle_std=0.2, seed=1), sample_index=0
    )
    other_seed = apply_speckle_noise(
        image, NoiseConfig(speckle_std=0.2, seed=2), sample_index=0
    )
    other_sample = apply_speckle_noise(
        image, NoiseConfig(speckle_std=0.2, seed=1), sample_index=1
    )

    assert not np.array_equal(baseline, other_seed)
    assert not np.array_equal(baseline, other_sample)


def test_zero_speckle_std_returns_float32_copy() -> None:
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)

    result = apply_speckle_noise(image, NoiseConfig(seed=3), sample_index=0)

    np.testing.assert_array_equal(result, image)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, image)


def test_multiplicative_is_default_and_additive_uses_constant_scale() -> None:
    image = np.full((8, 8), 0.25, dtype=np.float32)
    multiplicative_config = NoiseConfig(speckle_std=0.01, seed=6)
    additive_config = NoiseConfig(
        speckle_std=0.01,
        speckle_mode="additive",
        seed=6,
    )

    multiplicative = apply_speckle_noise(image, multiplicative_config, sample_index=2)
    additive = apply_speckle_noise(image, additive_config, sample_index=2)

    assert multiplicative_config.speckle_mode == "multiplicative"
    np.testing.assert_allclose(
        multiplicative - image,
        image * (additive - image),
        rtol=2e-5,
        atol=2e-8,
    )


@pytest.mark.parametrize("speckle_std", [-0.1, np.inf, np.nan])
def test_noise_config_rejects_invalid_speckle_std(speckle_std) -> None:
    with pytest.raises(ValueError, match="speckle_std"):
        NoiseConfig(speckle_std=speckle_std)


def test_noise_config_rejects_invalid_speckle_mode() -> None:
    with pytest.raises(ValueError, match="speckle_mode"):
        NoiseConfig(speckle_mode="other")  # type: ignore[arg-type]
