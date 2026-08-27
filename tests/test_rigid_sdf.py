import numpy as np

from constraints.generators.factories import preview_artificial_sample
from constraints.generators.rigid import apply_rigid
from constraints.generators.types import EmptyArteryConfig
from constraints.utils import signed_distance_scipy


def _foreground_channels(labels: np.ndarray) -> np.ndarray:
    return np.stack([labels == class_id for class_id in (1, 2, 3)], axis=-1)


def test_rigidly_transformed_sdf_approximates_strict_recomputation() -> None:
    sample = preview_artificial_sample(
        EmptyArteryConfig(20, 5, (65, 65)),
        seed=31,
    )
    parameters = (0.15, 3.0, -2.0)
    sdf_before = signed_distance_scipy(_foreground_channels(sample.target_labels))
    rigid_labels = np.rint(
        apply_rigid(sample.target_labels, *parameters, method="nearest")
    ).astype(np.uint8)
    strict_sdf = signed_distance_scipy(_foreground_channels(rigid_labels))

    transformed_sdf = np.stack(
        [
            apply_rigid(
                sdf_before[..., channel],
                *parameters,
                method="linear",
                padding_mode="border",
            )
            for channel in range(sdf_before.shape[-1])
        ],
        axis=-1,
    )
    error = np.abs(transformed_sdf - strict_sdf)
    boundary_band = np.abs(strict_sdf) <= 10
    sign_disagreement = (transformed_sdf[boundary_band] < 0) != (
        strict_sdf[boundary_band] < 0
    )

    assert error[boundary_band].mean() < 0.5
    assert sign_disagreement.mean() < 0.02
