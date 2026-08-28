"""Benchmark the composed artificial-data pipeline on its training machine.

Example:
    .venv/bin/python scripts/benchmark_composed_pipeline.py \
        data/artificial/benchmark --prepare-demo --output benchmark.json
"""

import json
import platform
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any

import kornia
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Subset

from constraints.datatools.datasets import ComposedArtificialDataset
from constraints.devices import resolve_compute_device
from constraints.generators.deformation import sample_valid_deformation
from constraints.generators.factories import (
    create_deformation_collection,
    create_layer_collection,
    create_rigid_collection,
    get_source_config,
)
from constraints.generators.layer_generators import (
    PowerPlaqueSamplingRanges,
    SavedLayer,
    power_layer_backup,
)
from constraints.generators.rigid import apply_rigid
from constraints.generators.source import create_source
from constraints.generators.types import (
    DeformationConfig,
    DeformationRejectionConfig,
    FloatRange,
    RigidConfig,
    SourceConfig,
)
from constraints.utils import signed_distance_kornia, signed_distance_scipy


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "kornia": kornia.__version__,
        "scipy": scipy.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
    }


def _prepare_demo_dataset(
    root: Path,
    num_samples: int,
    device: torch.device,
) -> None:
    """Create or reuse the self-contained collection used by this benchmark."""
    required = (
        root / "source_config.json",
        root / "layers/2blobs/labels.npy",
        root / "layers/2blobs/image.npy",
        root / "deformations/validated-default/fields.npy",
        root / "deformations/validated-default/rigid/small.npy",
    )
    if root.exists():
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"cannot prepare demo in incomplete existing folder {root}; "
                f"missing: {missing}"
            )
        print(f"Reusing prepared benchmark dataset: {root}", flush=True)
        return

    print(
        f"Preparing {num_samples} benchmark samples in {root} on {device}",
        flush=True,
    )
    source_config = SourceConfig(num_elements=num_samples)
    create_source(root, source_config)
    default_plaque = PowerPlaqueSamplingRanges()
    create_layer_collection(
        root,
        "2blobs",
        power_layer_backup((default_plaque, default_plaque), seed=42),
    )
    create_deformation_collection(
        root,
        "validated-default",
        DeformationConfig(fractal_mode="blur"),
        seed=42,
        device=device,
    )
    create_rigid_collection(
        root,
        "small",
        RigidConfig(
            angle=FloatRange(-0.1, 0.1),
            dx=FloatRange(-5.0, 5.0),
            dy=FloatRange(-5.0, 5.0),
        ),
        deformation="validated-default",
        seed=52,
    )
    print(f"Prepared benchmark dataset: {root}", flush=True)


def _consume_loader(
    loader: DataLoader,
    transfer_device: torch.device | None,
) -> int:
    count = 0
    for batch in loader:
        count += len(batch["sample_id"])
        if transfer_device is not None:
            batch["image"].to(transfer_device, non_blocking=True)
            batch["target_labels"].to(transfer_device, non_blocking=True)
    if transfer_device is not None:
        _synchronize(transfer_device)
    return count


def _benchmark_loader(
    dataset: ComposedArtificialDataset,
    args: Namespace,
    cuda_device: torch.device | None,
) -> list[dict[str, Any]]:
    repeated = Subset(
        dataset,
        [index % len(dataset) for index in range(args.loader_samples)],
    )
    results = []
    for workers in args.workers:
        transfer_device = cuda_device if args.include_cuda_transfer else None
        loader = DataLoader(
            repeated,
            batch_size=args.batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=transfer_device is not None,
        )
        _consume_loader(loader, transfer_device)
        started = perf_counter()
        count = _consume_loader(loader, transfer_device)
        elapsed = perf_counter() - started
        results.append(
            {
                "workers": workers,
                "samples": count,
                "seconds": elapsed,
                "samples_per_second": count / elapsed,
                "milliseconds_per_batch": (
                    1000 * elapsed / np.ceil(count / args.batch_size)
                ),
                "includes_cuda_transfer": transfer_device is not None,
            }
        )
    return results


def _foreground(
    dataset: ComposedArtificialDataset,
    labels: torch.Tensor,
) -> torch.Tensor:
    return dataset.label_schema.label_map_to_foreground_one_hot(labels).float()


def _time_sdf(
    function,
    mask: torch.Tensor,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    function(mask)
    _synchronize(device)
    started = perf_counter()
    for _ in range(repeats):
        function(mask)
    _synchronize(device)
    elapsed = perf_counter() - started
    sample_count = repeats * mask.shape[0]
    return {
        "device": str(device),
        "batch_size": mask.shape[0],
        "repeats": repeats,
        "seconds": elapsed,
        "milliseconds_per_sample": 1000 * elapsed / sample_count,
        "samples_per_second": sample_count / elapsed,
    }


def _benchmark_sdf(
    dataset: ComposedArtificialDataset,
    labels: torch.Tensor,
    args: Namespace,
    cuda_device: torch.device | None,
) -> dict[str, Any]:
    foreground = _foreground(dataset, labels)
    batch = foreground.unsqueeze(0).repeat(args.sdf_batch_size, 1, 1, 1)
    results = {
        "scipy_cpu": _time_sdf(
            signed_distance_scipy,
            batch,
            args.sdf_repeats,
            torch.device("cpu"),
        ),
        "kornia_cpu": _time_sdf(
            lambda mask: signed_distance_kornia(mask, device="cpu"),
            batch,
            args.sdf_repeats,
            torch.device("cpu"),
        ),
    }
    if cuda_device is not None:
        results["kornia_cuda"] = _time_sdf(
            lambda mask: signed_distance_kornia(mask, device=cuda_device),
            batch.to(cuda_device),
            args.sdf_repeats,
            cuda_device,
        )
    return results


def _compute_sdf(
    foreground: torch.Tensor,
    implementation: str,
    cuda_device: torch.device | None,
) -> np.ndarray:
    if implementation == "scipy":
        result = signed_distance_scipy(foreground)
    elif cuda_device is None:
        result = signed_distance_kornia(foreground, device="cpu")
    else:
        result = signed_distance_kornia(
            foreground.to(cuda_device), device=cuda_device
        ).cpu()
    return result.numpy()


def _rigid_sdf_metrics(
    dataset: ComposedArtificialDataset,
    labels: torch.Tensor,
    parameters: tuple[float, float, float],
    implementation: str,
    cuda_device: torch.device | None,
    band_width: float,
) -> dict[str, Any]:
    foreground = _foreground(dataset, labels)
    sdf_before = _compute_sdf(foreground, implementation, cuda_device)
    rigid_labels = np.rint(
        apply_rigid(labels.numpy(), *parameters, method="nearest")
    ).astype(np.int64)
    rigid_foreground = _foreground(dataset, torch.from_numpy(rigid_labels))
    strict = _compute_sdf(rigid_foreground, implementation, cuda_device)

    padding_results = {}
    for padding_mode in ("zeros", "border"):
        transformed = np.stack(
            [
                apply_rigid(
                    channel,
                    *parameters,
                    method="linear",
                    padding_mode=padding_mode,
                )
                for channel in sdf_before
            ]
        )
        absolute_error = np.abs(transformed - strict)
        band = np.abs(strict) <= band_width
        padding_results[padding_mode] = {
            "mean_absolute_error": float(absolute_error.mean()),
            "band_mean_absolute_error": float(absolute_error[band].mean()),
            "band_maximum_absolute_error": float(absolute_error[band].max()),
            "band_sign_disagreement_fraction": float(
                np.mean((transformed[band] < 0) != (strict[band] < 0))
            ),
        }
    return {
        "implementation": implementation,
        "parameters": {
            "angle_rad": parameters[0],
            "dx_px": parameters[1],
            "dy_px": parameters[2],
        },
        "band_width_px": band_width,
        "padding": padding_results,
    }


def _load_deformation_settings(
    root: Path,
    deformation_name: str | None,
) -> tuple[DeformationConfig, DeformationRejectionConfig]:
    if deformation_name is None:
        return DeformationConfig(fractal_mode="blur"), DeformationRejectionConfig()
    path = root / "deformations" / deformation_name / "config.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return (
        DeformationConfig.from_dict(metadata["config"]),
        DeformationRejectionConfig(**metadata["rejection"]),
    )


def _benchmark_deformation(
    root: Path,
    deformation_name: str | None,
    args: Namespace,
    cuda_device: torch.device | None,
) -> list[dict[str, Any]]:
    source_labels = np.load(root / "empty_artery.npy", mmap_mode="r")
    config, rejection = _load_deformation_settings(root, deformation_name)
    devices = [torch.device("cpu")]
    if cuda_device is not None:
        devices.append(cuda_device)

    results = []
    for device in devices:
        started = perf_counter()
        attempts = 0
        for sample_index in range(args.deformation_samples):
            sample = sample_valid_deformation(
                source_labels,
                config,
                rejection,
                seed=args.deformation_seed,
                sample_index=sample_index,
                device=device,
            )
            attempts += sample.attempts
        _synchronize(device)
        elapsed = perf_counter() - started
        results.append(
            {
                "device": str(device),
                "samples": args.deformation_samples,
                "attempts": attempts,
                "seconds": elapsed,
                "milliseconds_per_sample": (1000 * elapsed / args.deformation_samples),
                "config": config.to_dict(),
            }
        )
    return results


def _print_results(results: dict[str, Any]) -> None:
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--prepare-demo",
        action="store_true",
        help="create a reusable 2blobs + blur deformation + small rigid source",
    )
    parser.add_argument("--prepare-samples", type=int, default=20)
    parser.add_argument(
        "--prepare-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--layer", action="append", default=[])
    parser.add_argument("--deformation")
    parser.add_argument("--rigid")
    parser.add_argument("--loader-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--include-cuda-transfer", action="store_true")
    parser.add_argument("--sdf-batch-size", type=int, default=4)
    parser.add_argument("--sdf-repeats", type=int, default=2)
    parser.add_argument("--sdf-band-width", type=float, default=10.0)
    parser.add_argument("--calibration-angle", type=float, default=0.15)
    parser.add_argument("--calibration-dx", type=float, default=3.0)
    parser.add_argument("--calibration-dy", type=float, default=-2.0)
    parser.add_argument("--deformation-samples", type=int, default=5)
    parser.add_argument("--deformation-seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.prepare_samples <= 0:
        parser.error("prepared demo sample count must be positive")
    if args.loader_samples <= 0 or args.batch_size <= 0:
        parser.error("loader sample and batch counts must be positive")
    if args.sdf_batch_size <= 0 or args.sdf_repeats <= 0:
        parser.error("SDF batch and repeat counts must be positive")
    if args.deformation_samples <= 0:
        parser.error("deformation sample count must be positive")
    if any(workers < 0 for workers in args.workers):
        parser.error("worker counts must be non-negative")

    cuda_device = torch.device("cuda") if torch.cuda.is_available() else None
    if args.prepare_demo:
        if args.layer and args.layer != ["2blobs"]:
            parser.error("--prepare-demo only provides --layer 2blobs")
        if args.deformation not in {None, "validated-default"}:
            parser.error("--prepare-demo only provides --deformation validated-default")
        if args.rigid not in {None, "small"}:
            parser.error("--prepare-demo only provides --rigid small")
        try:
            prepare_device = resolve_compute_device(args.prepare_device)
        except RuntimeError as error:
            parser.error(str(error))
        _prepare_demo_dataset(
            args.source_root,
            args.prepare_samples,
            prepare_device,
        )
        args.layer = ["2blobs"]
        args.deformation = "validated-default"
        args.rigid = "small"
    elif not (args.source_root / "source_config.json").is_file():
        parser.error(
            f"{args.source_root} is not a prepared source dataset; "
            "pass --prepare-demo to create one"
        )

    saved_layers = tuple(SavedLayer(name) for name in args.layer)
    geometry_dataset = ComposedArtificialDataset(
        args.source_root,
        layers=saved_layers,
        deformation=args.deformation,
    )
    dataset = ComposedArtificialDataset(
        args.source_root,
        layers=saved_layers,
        deformation=args.deformation,
        rigid=args.rigid,
    )
    geometry_sample = geometry_dataset[0]
    if args.rigid is None:
        rigid_parameters = (
            args.calibration_angle,
            args.calibration_dx,
            args.calibration_dy,
        )
    else:
        rigid_parameters = tuple(float(value) for value in dataset[0]["rigid"])

    results = {
        "environment": _environment(),
        "recipe": {
            "source_root": str(args.source_root),
            "layers": args.layer,
            "deformation": args.deformation,
            "rigid": args.rigid,
        },
        "loader": _benchmark_loader(dataset, args, cuda_device),
        "sdf": _benchmark_sdf(
            geometry_dataset,
            geometry_sample["target_labels"],
            args,
            cuda_device,
        ),
        "rigid_sdf": [
            _rigid_sdf_metrics(
                geometry_dataset,
                geometry_sample["target_labels"],
                rigid_parameters,
                implementation,
                cuda_device,
                args.sdf_band_width,
            )
            for implementation in ("scipy", "kornia")
        ],
        "deformation": _benchmark_deformation(
            args.source_root,
            args.deformation,
            args,
            cuda_device,
        ),
    }
    _print_results(results)
    if args.output is not None:
        args.output.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
