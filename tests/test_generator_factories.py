import json

import numpy as np

from constraints.generators.factories import (
    create_plaque_collection,
    get_source_config,
)
from constraints.generators.source import create_source
from constraints.generators.types import (
    EmptyArteryConfig,
    PowerPlaqueSamplingRanges,
    SourceConfig,
)


def test_factory_creates_collection_from_source_root(tmp_path) -> None:
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=2,
        image_size=(65, 65),
        empty_artery=EmptyArteryConfig(20, 5),
    )
    create_source(root, config)

    masks_path, parameters_path = create_plaque_collection(
        root,
        "three-big-blobs",
        (PowerPlaqueSamplingRanges(),) * 3,
        seed=12,
    )

    assert get_source_config(root) == config
    assert np.load(masks_path).shape == (2, 65, 65)
    assert len(parameters_path.read_text().splitlines()) == 2
    first_record = json.loads(parameters_path.read_text().splitlines()[0])
    assert first_record["sample_index"] == 0
    assert len(first_record["plaques"]) == 3
