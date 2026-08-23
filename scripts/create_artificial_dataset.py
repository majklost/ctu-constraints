"""
Create a dataset with distance functions and masks for training - arficial ellipse dataset.
"""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constraints.datatools.datasets import (
    ARTIFICIAL_MASK_NUM_CLASSES,
    write_bad_indices,
)
from constraints.generators.generators import (
    ArteryGeneratorRigid,
    ArteryGeneratorDeformed,
    RigidSampleBounds,
    NO_RIGID,
    ROT_ONLY,
    SMALL,
)
from constraints.utils import (
    save_manifest,
    signed_distance_kornia,
    signed_distance_scipy,
    foreground_channels,
)


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return array.astype(np.float32, copy=False)


def _prepare_files(
    output_dir: Path,
    num_samples: int,
    category_shapes: dict[str, tuple[int, ...]],
) -> dict[str, np.memmap]:
    """Prepare one memory-mapped .npy file per category."""
    files: dict[str, np.memmap] = {}
    for name, shape in category_shapes.items():
        files[name] = np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, *shape),
        )
    return files


def _save_template(output_dir: Path, template: torch.Tensor):
    np.save(output_dir / "template.npy", _to_numpy(template))





def _resolve_rigid_sample_specs(rigid_mode: str) -> RigidSampleBounds:
    if rigid_mode == "rot":
        return ROT_ONLY
    elif rigid_mode == "small":
        return SMALL
    elif rigid_mode == "large":
        return RigidSampleBounds()
    elif rigid_mode == "none":
        return NO_RIGID
    else:
        raise ValueError(f"Unknown rigid mode: {rigid_mode}")


def create_rigid(args) -> None:
    sample_specs = _resolve_rigid_sample_specs(args.rigid_mode)
    output_dir = Path(args.output_dir)
    dataset = ArteryGeneratorRigid(
        fixed_seed=args.seed,
        num_samples=args.num_samples,
        speckle=0.2,
        sample_specs=sample_specs,
    )

    first_sample = dataset[0]
    img_shape = tuple(first_sample["img"].shape)
    mask_shape = tuple(first_sample["mask"].shape)
    sdf_shape = tuple(foreground_channels(first_sample["mask"]).shape)
    rigid_shape = tuple(first_sample["rigid"].shape)

    category_shapes = {
        "img": img_shape,
        "mask": mask_shape,
        "transform": rigid_shape,
    }
    if args.sdf_type in ("scipy", "both"):
        category_shapes["sdf_scipy"] = sdf_shape
    if args.sdf_type in ("kornia", "both"):
        category_shapes["sdf_kornia"] = sdf_shape

    files = _prepare_files(output_dir, args.num_samples, category_shapes)
    _save_template(output_dir, first_sample["template"])

    for idx in tqdm(range(len(dataset)), desc="Generating rigid dataset"):
        sample = dataset[idx]
        img, mask, template, rigid = (
            sample["img"],
            sample["mask"],
            sample["template"],
            sample["rigid"],  # (2, 3) rigid matrix
        )

        files["img"][idx] = _to_numpy(img)
        files["mask"][idx] = _to_numpy(mask)
        files["transform"][idx] = _to_numpy(rigid)
        foreground_mask = foreground_channels(mask)

        if args.sdf_type in ("scipy", "both"):
            files["sdf_scipy"][idx] = _to_numpy(signed_distance_scipy(foreground_mask))
        if args.sdf_type in ("kornia", "both"):
            files["sdf_kornia"][idx] = _to_numpy(signed_distance_kornia(foreground_mask))

        if idx == 0 and template.shape != first_sample["template"].shape:
            raise RuntimeError("Template shape changed across samples unexpectedly.")

    for mmap in files.values():
        mmap.flush()
    bad_indices = write_bad_indices(output_dir, check_wall_integrity=False)
    print(f"Saved {len(bad_indices)} invalid sample indices to {output_dir / 'bad_indices.csv'}")
    save_manifest(output_dir, args)


def create_deformed(args) -> None:
    sample_specs = _resolve_rigid_sample_specs(args.rigid_mode)
    output_dir = Path(args.output_dir)
    dataset = ArteryGeneratorDeformed(
        num_samples=args.num_samples,
        fixed_seed=args.seed,
        sample_specs=sample_specs,
        magnitude=7.0,
        integrations=2,
        scales=14,
        fractal_mode="blur",
        speckle=0.2,
    )

    first_sample = dataset[0]
    img_shape = tuple(first_sample["img"].shape)
    mask_shape = tuple(first_sample["mask"].shape)
    sdf_shape = tuple(foreground_channels(first_sample["mask"]).shape)
    field_shape = tuple(first_sample["field"].shape)

    category_shapes = {
        "img": img_shape,
        "mask": mask_shape,
        "transform": field_shape,
    }
    if args.sdf_type in ("scipy", "both"):
        category_shapes["sdf_scipy"] = sdf_shape
    if args.sdf_type in ("kornia", "both"):
        category_shapes["sdf_kornia"] = sdf_shape

    files = _prepare_files(output_dir, args.num_samples, category_shapes)
    _save_template(output_dir, first_sample["template"])

    for idx in tqdm(range(len(dataset)), desc="Generating deformed dataset"):
        sample = dataset[idx]
        img, mask, template, field = (
            sample["img"],
            sample["mask"],
            sample["template"],
            sample["field"],  # (2d field) deformation field
        )

        files["img"][idx] = _to_numpy(img)
        files["mask"][idx] = _to_numpy(mask)
        files["transform"][idx] = _to_numpy(field)
        foreground_mask = foreground_channels(mask)

        if args.sdf_type in ("scipy", "both"):
            files["sdf_scipy"][idx] = _to_numpy(signed_distance_scipy(foreground_mask))
        if args.sdf_type in ("kornia", "both"):
            files["sdf_kornia"][idx] = _to_numpy(signed_distance_kornia(foreground_mask))

        if idx == 0 and template.shape != first_sample["template"].shape:
            raise RuntimeError("Template shape changed across samples unexpectedly.")

    for mmap in files.values():
        mmap.flush()
    bad_indices = write_bad_indices(output_dir, check_wall_integrity=False)
    print(f"Saved {len(bad_indices)} invalid sample indices to {output_dir / 'bad_indices.csv'}")
    save_manifest(output_dir, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("num_samples", type=int, help="Number of samples to generate")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory e.g. data/artificial/rigid",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--sdf_type",
        type=str,
        choices=["scipy", "kornia", "none", "both"],
        default="both",
        help="Type of SDF computation to use",
    )

    parser.add_argument(
        "--generator_type",
        type=str,
        choices=["rigid", "deformed", "both"],
        default="rigid",
        help="Type of generator to use",
    )
    parser.add_argument(
        "--rigid_mode",
        type=str,
        choices=["none", "rot", "small", "large"],
        help="Mode for rigid transformation before optional deformation",
        default="large",
    )

    args = parser.parse_args()
    if args.num_samples < 0:
        raise ValueError("num_samples must be a positive integer.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.generator_type == "rigid":
        create_rigid(args)
    elif args.generator_type == "deformed":
        create_deformed(args)
    elif args.generator_type == "both":
        rigid_args = deepcopy(args)
        rigid_args.generator_type = "rigid"
        rigid_args.output_dir = str(Path(args.output_dir) / "rigid")
        Path(rigid_args.output_dir).mkdir(parents=True, exist_ok=True)
        create_rigid(rigid_args)

        deformed_args = deepcopy(args)
        deformed_args.generator_type = "deformed"
        deformed_args.output_dir = str(Path(args.output_dir) / "deformed")
        Path(deformed_args.output_dir).mkdir(parents=True, exist_ok=True)
        create_deformed(deformed_args)
    else:
        raise ValueError(f"Unknown generator type: {args.generator_type}")
