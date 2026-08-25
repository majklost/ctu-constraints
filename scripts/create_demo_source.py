"""Create a small source dataset for inspecting the generation framework."""

from pathlib import Path

from constraints.generators.factories import create_plaque_collection
from constraints.generators.source import create_source
from constraints.generators.types import PowerPlaqueSamplingRanges, SourceConfig

DEMO_ROOT = Path("data/artificial/demo")
DEMO_NUM_SAMPLES = 20
DEMO_SEED = 42


def create_demo_dataset(root: Path = DEMO_ROOT) -> None:
    """Create an empty-artery source and a two-plaque collection."""
    create_source(root, SourceConfig(num_elements=DEMO_NUM_SAMPLES))

    default_plaque = PowerPlaqueSamplingRanges()
    create_plaque_collection(
        root,
        "2blobs",
        (default_plaque, default_plaque),
        seed=DEMO_SEED,
    )


if __name__ == "__main__":
    create_demo_dataset()
