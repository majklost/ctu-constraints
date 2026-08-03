"""Migrate cached artificial datasets to explicit-background masks/templates.

The migrated representation stores:
- mask.npy and template.npy as [background, boundary, lumen, plaque]
- sdf_scipy.npy and sdf_kornia.npy as foreground-only [boundary, lumen, plaque]

This keeps SDF losses on foreground structures while making the model-facing mask
and template representation explicit-background by default.
"""

from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from constraints.datatools.datasets import (
    ARTIFICIAL_MASK_NUM_CLASSES,
    ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS,
)
from constraints.utils import signed_distance_kornia, signed_distance_scipy


REQUIRED_FILES = ("img.npy", "mask.npy", "template.npy", "transform.npy")
OPTIONAL_COPY_FILES = ("sdf_scipy.npy",)


def _explicit_background(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        channel_axis = 0
    elif mask.ndim == 4:
        channel_axis = 1
    else:
        raise ValueError(f"Expected [C,H,W] or [N,C,H,W], got shape {mask.shape}")

    channels = mask.shape[channel_axis]
    if channels == ARTIFICIAL_MASK_NUM_CLASSES:
        return mask.astype(np.float32, copy=False)
    if channels != ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS:
        raise ValueError(
            f"Expected {ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS} or "
            f"{ARTIFICIAL_MASK_NUM_CLASSES} channels, got {channels} in shape {mask.shape}"
        )

    mask = mask.astype(np.float32, copy=False)
    if mask.ndim == 3:
        background = 1.0 - mask.max(axis=0, keepdims=True)
        return np.concatenate([np.clip(background, 0.0, 1.0), mask], axis=0).astype(np.float32)

    background = 1.0 - mask.max(axis=1, keepdims=True)
    return np.concatenate([np.clip(background, 0.0, 1.0), mask], axis=1).astype(np.float32)


def _foreground_channels(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] mask, got shape {mask.shape}")
    if mask.shape[1] == ARTIFICIAL_MASK_NUM_CLASSES:
        return mask[:, 1:]
    if mask.shape[1] == ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS:
        return mask
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def _write_array(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp_path, array.astype(np.float32, copy=False))
    os.replace(tmp_path.with_suffix(tmp_path.suffix + ".npy"), path)


def _copy_or_link(src: Path, dst: Path, overwrite: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
        dst.unlink()
    shutil.copy2(src, dst)


def _prepare_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    for filename in REQUIRED_FILES:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("img.npy", "transform.npy"):
        _copy_or_link(input_dir / filename, output_dir / filename, overwrite=overwrite)
    for filename in OPTIONAL_COPY_FILES:
        src = input_dir / filename
        if src.exists():
            _copy_or_link(src, output_dir / filename, overwrite=overwrite)


def _migrate_template(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    output_path = output_dir / "template.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    template = np.load(input_dir / "template.npy")
    _write_array(output_path, _explicit_background(template))


def _migrate_masks(input_dir: Path, output_dir: Path, overwrite: bool, chunk_size: int) -> None:
    output_path = output_dir / "mask.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    masks = np.load(input_dir / "mask.npy", mmap_mode="r")
    if masks.ndim != 4:
        raise ValueError(f"Expected mask.npy shape [N,C,H,W], got {masks.shape}")
    output_shape = (masks.shape[0], ARTIFICIAL_MASK_NUM_CLASSES, *masks.shape[-2:])
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )
    for start in tqdm(range(0, masks.shape[0], chunk_size), desc=f"masks {input_dir.name}"):
        end = min(start + chunk_size, masks.shape[0])
        output[start:end] = _explicit_background(np.array(masks[start:end]))
    output.flush()


def _regenerate_sdf(
    input_dir: Path,
    output_dir: Path,
    name: str,
    overwrite: bool,
    chunk_size: int,
) -> None:
    output_path = output_dir / f"sdf_{name}.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    masks = np.load(input_dir / "mask.npy", mmap_mode="r")
    output_shape = (masks.shape[0], ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, *masks.shape[-2:])
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )
    sdf_fn = signed_distance_kornia if name == "kornia" else signed_distance_scipy
    for start in tqdm(range(0, masks.shape[0], chunk_size), desc=f"sdf_{name} {input_dir.name}"):
        end = min(start + chunk_size, masks.shape[0])
        foreground = _foreground_channels(np.array(masks[start:end]))
        sdf = sdf_fn(torch.from_numpy(foreground))
        if isinstance(sdf, torch.Tensor):
            sdf = sdf.detach().cpu().numpy()
        output[start:end] = sdf.astype(np.float32, copy=False)
    output.flush()


def _write_manifest(input_dir: Path, output_dir: Path, args: Namespace) -> None:
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as file:
            manifest = json.load(file)
    else:
        manifest = {}
    manifest["explicit_background_migration"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source_dir": str(input_dir),
        "sdf_channels": "foreground_only",
        "recomputed_sdf_kornia": True,
        "recomputed_sdf_scipy": bool(args.recompute_scipy),
    }
    with open(output_dir / "manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)


def migrate_dataset_dir(input_dir: Path, output_dir: Path, args: Namespace) -> None:
    _prepare_output_dir(input_dir, output_dir, overwrite=args.overwrite)
    _migrate_template(input_dir, output_dir, overwrite=args.overwrite)
    _migrate_masks(input_dir, output_dir, overwrite=args.overwrite, chunk_size=args.chunk_size)
    _regenerate_sdf(input_dir, output_dir, "kornia", overwrite=args.overwrite, chunk_size=args.chunk_size)
    if args.recompute_scipy:
        _regenerate_sdf(input_dir, output_dir, "scipy", overwrite=args.overwrite, chunk_size=args.chunk_size)
    _write_manifest(input_dir, output_dir, args)


def _resolve_output_dir(input_dir: Path, args: Namespace, common_root: Path | None) -> Path:
    if args.in_place:
        return input_dir
    if args.output_root is None:
        return input_dir.with_name(input_dir.name + "_explicit_bg")
    if common_root is not None:
        return Path(args.output_root) / input_dir.relative_to(common_root)
    return Path(args.output_root) / input_dir.name


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--in_place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recompute_scipy", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    if args.in_place and args.output_root is not None:
        raise ValueError("Use either --in_place or --output_root, not both.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    common_root = None
    if args.output_root is not None and len(args.dataset_dirs) > 1:
        common_root = Path(os.path.commonpath([path.resolve() for path in args.dataset_dirs]))

    for dataset_dir in args.dataset_dirs:
        input_dir = dataset_dir.resolve()
        output_dir = _resolve_output_dir(input_dir, args, common_root).resolve()
        print(f"Migrating {input_dir} -> {output_dir}")
        migrate_dataset_dir(input_dir, output_dir, args)
