import numpy as np

from constraints.generators.factories import (
    create_layer_collection,
    get_source_config,
)
from constraints.generators.layer_generators import (
    LayerPatch,
    PowerPlaqueSamplingRanges,
    power_layer_backup,
    register_layer_resolver,
)
from constraints.generators.recipe_backups import LayerBackup
from constraints.generators.source import create_source
from constraints.generators.types import (
    EmptyArteryConfig,
    SourceConfig,
)


def test_factory_creates_collection_from_source_root(tmp_path) -> None:
    root = tmp_path / "source"
    config = SourceConfig(
        num_elements=2,
        empty_artery=EmptyArteryConfig(20, 5, (65, 65)),
    )
    create_source(root, config)

    folder = create_layer_collection(
        root,
        "three-big-blobs",
        power_layer_backup((PowerPlaqueSamplingRanges(),) * 3, seed=12),
    )

    assert get_source_config(root) == config
    assert np.load(folder / "labels.npy").shape == (2, 65, 65)
    assert np.load(folder / "image.npy").shape == (2, 65, 65)


def test_registered_resolver_can_store_independent_patches(tmp_path) -> None:
    @register_layer_resolver("test-independent-patches-v1")
    def separated(context, params):
        shape = context.source_config.empty_artery.image_size
        labels = np.full(shape, -1, dtype=np.int8)
        labels[1, 1] = 3
        image = np.full(shape, np.nan, dtype=np.float32)
        image[1, 1:3] = [0.2, 0.8]
        return LayerPatch(labels, image)

    root = tmp_path / "source"
    create_source(root, SourceConfig(1, EmptyArteryConfig(20, 5, (65, 65))))

    folder = create_layer_collection(
        root,
        "separated",
        LayerBackup("test-independent-patches-v1"),
    )

    labels = np.load(folder / "labels.npy")
    image = np.load(folder / "image.npy")
    assert (labels[0] >= 0).sum() == 1
    assert np.isfinite(image[0]).sum() == 2
