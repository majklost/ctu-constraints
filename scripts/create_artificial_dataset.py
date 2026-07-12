"""
Create a dataset with distance functions and masks for training - arficial ellipse dataset.
"""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from constraints.generators.generators import (
    ArteryGeneratorAffine,
    ArteryGeneratorDeformed,
    AffineSampleBound,ROT_ONLY,SMALL
)
from constraints.utils import (
    save_manifest,
    signed_distance_kornia,
    signed_distance_scipy,
)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


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


def create_affine(args) -> None:
    affine_mode = args.affine_mode
    if affine_mode == "rot":
        sample_specs = ROT_ONLY
    elif affine_mode == "small":
        sample_specs = SMALL
    else:
        sample_specs = None



    output_dir = Path(args.output_dir)
    dataset = ArteryGeneratorAffine(
        fixed_seed=args.seed,
        num_samples=args.num_samples,
        speckle=0.2,
        sample_specs=sample_specs,
    )

    first_sample = dataset[0]
    img_shape = tuple(first_sample["img"].shape)
    mask_shape = tuple(first_sample["mask"].shape)
    affine_shape = tuple(first_sample["affine"].shape)

    category_shapes = {
        "img": img_shape,
        "mask": mask_shape,
        "transform": affine_shape,
    }
    if args.sdf_type in ("scipy", "both"):
        category_shapes["sdf_scipy"] = mask_shape
    if args.sdf_type in ("kornia", "both"):
        category_shapes["sdf_kornia"] = mask_shape

    files = _prepare_files(output_dir, args.num_samples, category_shapes)
    _save_template(output_dir, first_sample["template"])

    for idx in tqdm(range(len(dataset)), desc="Generating affine dataset"):
        sample = dataset[idx]
        img, mask, template, affine = (
            sample["img"],
            sample["mask"],
            sample["template"],
            sample["affine"],  # (2,3) affine matrix
        )

        files["img"][idx] = _to_numpy(img)
        files["mask"][idx] = _to_numpy(mask)
        files["transform"][idx] = _to_numpy(affine)

        if args.sdf_type in ("scipy", "both"):
            files["sdf_scipy"][idx] = _to_numpy(signed_distance_scipy(mask))
        if args.sdf_type in ("kornia", "both"):
            files["sdf_kornia"][idx] = _to_numpy(signed_distance_kornia(mask))

        if idx == 0 and template.shape != first_sample["template"].shape:
            raise RuntimeError("Template shape changed across samples unexpectedly.")

    for mmap in files.values():
        mmap.flush()
    save_manifest(output_dir, args)


def create_deformed(args) -> None:
    output_dir = Path(args.output_dir)
    dataset = ArteryGeneratorDeformed(
        num_samples=args.num_samples,
        fixed_seed=args.seed,
        magnitude=7.0,
        integrations=2,
        scales=14,
        fractal_mode="blur",
        speckle=0.2,
    )

    first_sample = dataset[0]
    img_shape = tuple(first_sample["img"].shape)
    mask_shape = tuple(first_sample["mask"].shape)
    field_shape = tuple(first_sample["field"].shape)

    category_shapes = {
        "img": img_shape,
        "mask": mask_shape,
        "transform": field_shape,
    }
    if args.sdf_type in ("scipy", "both"):
        category_shapes["sdf_scipy"] = mask_shape
    if args.sdf_type in ("kornia", "both"):
        category_shapes["sdf_kornia"] = mask_shape

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

        if args.sdf_type in ("scipy", "both"):
            files["sdf_scipy"][idx] = _to_numpy(signed_distance_scipy(mask))
        if args.sdf_type in ("kornia", "both"):
            files["sdf_kornia"][idx] = _to_numpy(signed_distance_kornia(mask))

        if idx == 0 and template.shape != first_sample["template"].shape:
            raise RuntimeError("Template shape changed across samples unexpectedly.")

    for mmap in files.values():
        mmap.flush()
    save_manifest(output_dir, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("num_samples", type=int, help="Number of samples to generate")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory e.g. data/artificial/affine",
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
        choices=["affine", "deformed", "both"],
        default="affine",
        help="Type of generator to use",
    )
    parser.add_argument(
        "--affine_mode",
        type=str,
        choices=["rot","small","large"],
        help="Mode for affine transformation",
        default="large",
    )

    args = parser.parse_args()
    if args.num_samples < 0:
        raise ValueError("num_samples must be a positive integer.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.generator_type == "affine":
        create_affine(args)
    elif args.generator_type == "deformed":
        create_deformed(args)
    elif args.generator_type == "both":
        affine_args = deepcopy(args)
        affine_args.generator_type = "affine"
        affine_args.output_dir = str(Path(args.output_dir) / "affine")
        Path(affine_args.output_dir).mkdir(parents=True, exist_ok=True)
        create_affine(affine_args)

        deformed_args = deepcopy(args)
        deformed_args.generator_type = "deformed"
        deformed_args.output_dir = str(Path(args.output_dir) / "deformed")
        Path(deformed_args.output_dir).mkdir(parents=True, exist_ok=True)
        create_deformed(deformed_args)
    else:
        raise ValueError(f"Unknown generator type: {args.generator_type}")
