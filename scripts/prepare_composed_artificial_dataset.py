"""Prepare the first large lazily composed artificial dataset.

The geometry defaults mirror ``notebooks/ex5/composition_dataset.py``. The
script is restart-aware at artifact boundaries: completed collections are
validated and reused, while incompatible or partial outputs fail explicitly.
"""

import csv
import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import get_args

import numpy as np

from constraints.datatools.datasets import ComposedArtificialDataset
from constraints.datatools.datasets.types import SDFMode
from constraints.devices import DeviceSelection, resolve_compute_device
from constraints.generators.deformation import load_deformation_fields
from constraints.generators.factories import (
    create_deformation_collection,
    create_plaque_collection,
    create_rigid_collection,
    get_source_config,
)
from constraints.generators.recipes import Recipe
from constraints.generators.rigid import load_rigid_parameters
from constraints.generators.sdf_cache import SDFCacheConfig, create_sdf_cache
from constraints.generators.source import create_source
from constraints.generators.storage import write_json
from constraints.generators.types import (
    AppearanceKind,
    ArteryClass,
    DeformationConfig,
    DeformationRejectionConfig,
    FloatRange,
    PowerPlaqueSamplingRanges,
    RigidConfig,
    RigidRejectionConfig,
    SavedPlaque,
    SourceConfig,
)

DEFAULT_NUM_SAMPLES = 5_000
DEFAULT_TRAIN_SIZE = 2_000
DEFAULT_VAL_SIZE = 200
DEFAULT_SEED = 25

REAL_PLAQUES = "two-real-plaques"
FAKE_PLAQUES = "five-fake-plaques"
DEFORMATION = "default"
RIGID = "default"
RECIPE_NAME = "default"


def prepare_dataset(
    root: Path,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    device: DeviceSelection = "auto",
    sdf_mode: SDFMode | None = "scipy",
    sdf_batch_size: int = 16,
    include_rigid: bool = False,
) -> Recipe:
    """Create or validate every artifact needed by the configured dataset."""
    if train_size < 0 or val_size < 0 or train_size + val_size > num_samples:
        raise ValueError("train_size and val_size must fit within num_samples")
    source_config = SourceConfig(num_elements=num_samples)
    _print_storage_estimate(source_config, include_sdf=sdf_mode is not None)
    resolved_device = resolve_compute_device(device)
    real_ranges = _real_plaque_ranges()
    fake_range = _fake_plaque_range()
    fake_lumen_radius_px = source_config.empty_artery.lumen_radius_px - 10
    deformation_config = DeformationConfig()
    deformation_rejection = DeformationRejectionConfig()
    rigid_config = RigidConfig()
    rigid_rejection = RigidRejectionConfig()
    recipe = Recipe(
        plaques=(
            SavedPlaque(
                FAKE_PLAQUES,
                target_class=ArteryClass.LUMEN,
                appearance=AppearanceKind.PLAQUE,
            ),
            SavedPlaque(REAL_PLAQUES),
        ),
        deformation=DEFORMATION,
        rigid=RIGID if include_rigid else None,
    )
    definition = _json_normalize(
        {
            "format_version": 1,
            "source_config": source_config.to_dict(),
            "collections": {
                REAL_PLAQUES: {
                    "seed": seed,
                    "ranges": [asdict(item) for item in real_ranges],
                    "lumen_radius_px": source_config.empty_artery.lumen_radius_px,
                },
                FAKE_PLAQUES: {
                    "seed": seed + 1,
                    "ranges": [asdict(fake_range)] * 5,
                    "lumen_radius_px": fake_lumen_radius_px,
                },
                DEFORMATION: {
                    "seed": seed + 2,
                    "config": deformation_config.to_dict(),
                    "rejection": deformation_rejection.to_dict(),
                    "device": str(resolved_device),
                },
                "rigid": (
                    None
                    if not include_rigid
                    else {
                        "name": RIGID,
                        "seed": seed + 3,
                        "config": rigid_config.to_dict(),
                        "rejection": rigid_rejection.to_dict(),
                        "deformation": DEFORMATION,
                    }
                ),
            },
            "recipe": recipe.to_dict(),
            "splits": {
                "seed": seed + 4,
                "train_size": train_size,
                "val_size": val_size,
            },
            "sdf": (
                None if sdf_mode is None else SDFCacheConfig(mode=sdf_mode).to_dict()
            ),
        }
    )

    root = Path(root)
    preparation_path = root / "preparation.json"
    if root.exists():
        _validate_existing_preparation(preparation_path, definition)
        if get_source_config(root) != source_config:
            raise ValueError("existing source configuration does not match")
        print(f"Reusing compatible source root: {root}")
    else:
        print(f"Creating source root with {num_samples} samples: {root}")
        create_source(root, source_config)
        _write_preparation(preparation_path, definition, status="preparing")

    _ensure_plaque_collection(
        root,
        REAL_PLAQUES,
        real_ranges,
        seed=seed,
    )
    _ensure_plaque_collection(
        root,
        FAKE_PLAQUES,
        (fake_range,) * 5,
        seed=seed + 1,
        lumen_radius_px=fake_lumen_radius_px,
    )
    _ensure_deformation(
        root,
        source_config,
        deformation_config,
        deformation_rejection,
        seed=seed + 2,
        device=resolved_device,
    )
    if include_rigid:
        _ensure_rigid(
            root,
            source_config,
            rigid_config,
            rigid_rejection,
            seed=seed + 3,
        )

    recipe_path = root / "recipes" / f"{RECIPE_NAME}.json"
    if recipe_path.exists():
        if Recipe.load_json(recipe_path) != recipe:
            raise ValueError(f"existing recipe does not match: {recipe_path}")
    else:
        recipe.save_json(recipe_path)
        print(f"Saved recipe: {recipe_path}")

    _ensure_splits(
        root,
        num_samples=num_samples,
        train_size=train_size,
        val_size=val_size,
        seed=seed + 4,
    )

    if sdf_mode is not None:
        sdf_config = SDFCacheConfig(mode=sdf_mode)
        geometry_recipe = Recipe(
            plaques=recipe.plaques,
            deformation=recipe.deformation,
            class_intensities=recipe.class_intensities,
        )
        geometry_dataset = ComposedArtificialDataset.from_recipe(
            root,
            geometry_recipe,
        )
        identity = geometry_dataset.sdf_cache_identity(sdf_config)
        cache_folder = identity.cache_directory(root)
        if cache_folder.exists():
            _validate_sdf_cache(cache_folder, identity.digest, source_config)
            print(f"Reusing SDF cache: {cache_folder}")
        else:
            print(f"Creating {sdf_mode} SDF cache: {cache_folder}")
            create_sdf_cache(
                geometry_dataset,
                sdf_config,
                batch_size=sdf_batch_size,
                device=resolved_device,
            )

    _write_preparation(preparation_path, definition, status="complete")
    print(f"Dataset preparation complete: {root}")
    return recipe


def _real_plaque_ranges() -> tuple[PowerPlaqueSamplingRanges, ...]:
    shared = {
        "angular_width_rad": FloatRange.fixed(np.pi / 5),
        "inward_depth_fraction": FloatRange(0.2, 0.3),
        "wall_depth_fraction": FloatRange.fixed(0),
        "shape_power": FloatRange.fixed(0.5),
    }
    return (
        PowerPlaqueSamplingRanges(
            angle_rad=FloatRange(-np.pi / 3, -np.pi / 10),
            **shared,
        ),
        PowerPlaqueSamplingRanges(
            angle_rad=FloatRange(np.pi / 10, np.pi / 3),
            **shared,
        ),
    )


def _fake_plaque_range() -> PowerPlaqueSamplingRanges:
    return PowerPlaqueSamplingRanges(
        inward_depth_fraction=FloatRange(0.12, 0.15),
        wall_depth_fraction=FloatRange.fixed(0.1),
        shape_power=FloatRange.fixed(2),
    )


def _ensure_plaque_collection(
    root: Path,
    name: str,
    ranges: tuple[PowerPlaqueSamplingRanges, ...],
    *,
    seed: int,
    lumen_radius_px: float | None = None,
) -> None:
    masks_path = root / "plaques" / f"{name}.npy"
    parameters_path = root / "plaques" / f"{name}.jsonl"
    if _all_or_none_exist((masks_path, parameters_path), name):
        masks = np.load(masks_path, mmap_mode="r")
        source_config = get_source_config(root)
        expected_shape = (
            source_config.num_elements,
            *source_config.empty_artery.image_size,
        )
        if masks.shape != expected_shape or masks.dtype != np.bool_:
            raise ValueError(f"invalid plaque collection in {masks_path}")
        print(f"Reusing plaque collection: {name}")
        return
    print(f"Creating plaque collection: {name}")
    create_plaque_collection(
        root,
        name,
        ranges,
        seed=seed,
        lumen_radius_px=lumen_radius_px,
    )


def _ensure_deformation(
    root: Path,
    source_config: SourceConfig,
    config: DeformationConfig,
    rejection: DeformationRejectionConfig,
    *,
    seed: int,
    device: DeviceSelection,
) -> None:
    folder = root / "deformations" / DEFORMATION
    if folder.exists():
        load_deformation_fields(root / "deformations", DEFORMATION, source_config)
        print(f"Reusing deformation collection: {DEFORMATION}")
        return
    print(f"Creating deformation collection: {DEFORMATION}")
    create_deformation_collection(
        root,
        DEFORMATION,
        config,
        rejection,
        seed=seed,
        device=device,
    )


def _ensure_rigid(
    root: Path,
    source_config: SourceConfig,
    config: RigidConfig,
    rejection: RigidRejectionConfig,
    *,
    seed: int,
) -> None:
    folder = root / "deformations" / DEFORMATION
    parameters_path = folder / "rigid" / f"{RIGID}.npy"
    config_path = folder / "rigid" / f"{RIGID}.json"
    if _all_or_none_exist((parameters_path, config_path), RIGID):
        load_rigid_parameters(folder, RIGID, source_config)
        print(f"Reusing rigid collection: {RIGID}")
        return
    print(f"Creating rigid collection: {RIGID}")
    create_rigid_collection(
        root,
        RIGID,
        config,
        rejection,
        deformation=DEFORMATION,
        seed=seed,
    )


def _ensure_splits(
    root: Path,
    *,
    num_samples: int,
    train_size: int,
    val_size: int,
    seed: int,
) -> None:
    folder = root / "splits"
    train_path = folder / "trn_samples.csv"
    val_path = folder / "val_samples.csv"
    if _all_or_none_exist((train_path, val_path), "train/validation splits"):
        train_indices = _validate_split(train_path, train_size, num_samples)
        val_indices = _validate_split(val_path, val_size, num_samples)
        if set(train_indices) & set(val_indices):
            raise ValueError("training and validation splits overlap")
        print(f"Reusing split CSVs: {folder}")
        return
    indices = np.random.default_rng(seed).permutation(num_samples)
    _write_split(train_path, indices[:train_size])
    _write_split(val_path, indices[train_size : train_size + val_size])
    print(f"Saved split CSVs: {folder}")


def _write_split(path: Path, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index",))
        writer.writerows((int(index),) for index in indices)
    temporary.replace(path)


def _validate_split(
    path: Path, expected_size: int, num_samples: int
) -> tuple[int, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_index"]:
            raise ValueError(f"invalid split columns in {path}")
        indices = [int(row["sample_index"]) for row in reader]
    if len(indices) != expected_size or len(indices) != len(set(indices)):
        raise ValueError(f"invalid split size or duplicate indices in {path}")
    if any(index < 0 or index >= num_samples for index in indices):
        raise ValueError(f"split index outside source range in {path}")
    return tuple(indices)


def _validate_sdf_cache(
    folder: Path,
    expected_key: str,
    source_config: SourceConfig,
) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("cache_key") != expected_key
    ):
        raise ValueError(f"invalid SDF cache manifest in {folder}")
    values = np.load(folder / "sdf.npy", mmap_mode="r")
    expected_shape = (
        source_config.num_elements,
        3,
        *source_config.empty_artery.image_size,
    )
    if values.shape != expected_shape or values.dtype != np.float32:
        raise ValueError(f"invalid SDF cache array in {folder}")


def _validate_existing_preparation(path: Path, definition: dict) -> None:
    if not path.is_file():
        raise FileExistsError(
            "source root exists without this script's preparation manifest: "
            f"{path.parent}"
        )
    value = json.loads(path.read_text())
    if value.get("definition") != definition:
        raise ValueError("existing preparation uses a different dataset definition")


def _write_preparation(path: Path, definition: dict, *, status: str) -> None:
    write_json(
        path,
        {
            "format_name": "first-composed-artificial-dataset",
            "format_version": 1,
            "status": status,
            "definition": definition,
        },
    )


def _all_or_none_exist(paths: tuple[Path, ...], description: str) -> bool:
    exists = tuple(path.exists() for path in paths)
    if any(exists) and not all(exists):
        raise RuntimeError(f"partial {description} artifacts: {paths}")
    return all(exists)


def _json_normalize(value: dict) -> dict:
    return json.loads(json.dumps(value, sort_keys=True))


def _print_storage_estimate(config: SourceConfig, *, include_sdf: bool) -> None:
    height, width = config.empty_artery.image_size
    bytes_per_pixel = 2 + 2 * np.dtype(np.float32).itemsize
    if include_sdf:
        bytes_per_pixel += 3 * np.dtype(np.float32).itemsize
    gibibytes = config.num_elements * height * width * bytes_per_pixel / 1024**3
    print(f"Estimated generated-array storage: {gibibytes:.1f} GiB")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--sdf-mode", choices=get_args(SDFMode), default="scipy")
    parser.add_argument("--sdf-batch-size", type=int, default=16)
    parser.add_argument("--skip-sdf", action="store_true")
    parser.add_argument("--include-rigid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        args.root,
        num_samples=args.num_samples,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        device=args.device,
        sdf_mode=None if args.skip_sdf else args.sdf_mode,
        sdf_batch_size=args.sdf_batch_size,
        include_rigid=args.include_rigid,
    )


if __name__ == "__main__":
    main()
