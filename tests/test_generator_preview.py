import numpy as np

from constraints.generators.factories import preview_artificial_sample
from constraints.generators.types import (
    ArteryClass,
    DeformationConfig,
    EmptyArteryConfig,
    SourceConfig,
)


def test_preview_sample_requires_no_storage_and_returns_resolved_values() -> None:
    source_config = SourceConfig(
        num_elements=10,
        image_size=(33, 33),
        empty_artery=EmptyArteryConfig(10, 3),
    )

    sample = preview_artificial_sample(
        source_config,
        deformation_config=DeformationConfig(
            scales=8,
            magnitude=0,
            integrations=2,
            fractal_mode="upsample",
        ),
        seed=23,
        sample_index=4,
    )

    assert sample.image.shape == source_config.image_size
    assert sample.image.dtype == np.float32
    assert sample.target_labels.shape == source_config.image_size
    assert sample.target_labels.dtype == np.uint8
    assert ArteryClass.PLAQUE in sample.target_labels
    assert len(sample.plaque_parameters) == 1
    assert sample.deformation_field is not None
    assert sample.deformation_field.shape == (2, *source_config.image_size)
    assert sample.deformation_validation is not None
    assert sample.deformation_validation.accepted
