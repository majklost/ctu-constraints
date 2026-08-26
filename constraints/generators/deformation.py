"""Generation of independent, named deformation-field collections."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from constraints.voxelmorph.utils import random_disp, spatial_transform

from .storage import write_json
from .types import DeformationConfig, DeformationRejectionConfig, SourceConfig
from .validation import DeformationValidationResult, validate_deformation


@dataclass(frozen=True)
class DeformationSample:
    """One accepted deformation field and its rejection diagnostics."""

    field: NDArray[np.float32]
    attempts: int
    validation: DeformationValidationResult


def load_deformation_fields(
    folder: Path,
    name: str,
    source_config: SourceConfig,
) -> np.ndarray:
    """Load one named deformation preset."""
    folder = Path(folder)
    _validate_collection_name(name)
    preset_folder = folder / name
    if not preset_folder.is_dir():
        raise FileNotFoundError(f"deformation preset does not exist: {name}")
    fields = np.load(preset_folder / "fields.npy", mmap_mode="r")
    expected_shape = (source_config.num_elements, 2, *source_config.image_size)
    if fields.shape != expected_shape or fields.dtype != np.float32:
        raise ValueError(
            f"invalid deformation {name!r}: expected float32 "
            f"{expected_shape}, got {fields.shape} {fields.dtype}"
        )
    return fields


def apply_deformation(
    values: np.ndarray,
    field: np.ndarray,
    *,
    method: Literal["nearest", "linear"] = "nearest",
) -> NDArray[np.float32]:
    """Apply one stored backward-sampling field to one ``[H, W]`` array."""
    values = np.asarray(values)
    field = np.asarray(field)
    if values.ndim != 2:
        raise ValueError("values must have shape [H, W]")
    if field.shape != (2, *values.shape):
        raise ValueError("field must have shape [2, H, W] matching values")
    if not np.isfinite(field).all():
        raise ValueError("field contains non-finite values")

    values_tensor = torch.from_numpy(
        np.array(values, dtype=np.float32, copy=True)
    )[None, None]
    field_tensor = torch.from_numpy(np.array(field, dtype=np.float32, copy=True))
    with torch.no_grad():
        warped = spatial_transform(
            values_tensor,
            field_tensor,
            method=method,
        )
    return warped[0, 0].numpy().copy()


def sample_valid_deformation(
    source_config: SourceConfig,
    source_labels: np.ndarray,
    config: DeformationConfig,
    rejection: DeformationRejectionConfig | None = None,
    *,
    seed: int,
    sample_index: int = 0,
    device: torch.device | str = "cpu",
) -> DeformationSample:
    """Generate and validate one deformation without writing it.

    The seed scheme is identical to persisted collections. With the default
    CPU device, a preview for a given ``sample_index`` therefore produces the
    field later stored at the same collection index, independently of length.
    CUDA uses a different random-number backend and preserves the distribution,
    but is not expected to be bit-identical to CPU generation.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if rejection is None:
        rejection = DeformationRejectionConfig()

    source_labels = np.asarray(source_labels)
    if source_labels.shape != source_config.image_size:
        raise ValueError("source_labels must match source_config.image_size")
    device = torch.device(device)
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices.append(
            torch.cuda.current_device() if device.index is None else device.index
        )

    height, width = source_config.image_size
    for attempt_index in range(rejection.max_attempts):
        attempt_seed = _attempt_seed(seed, sample_index, attempt_index)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(attempt_seed)
            field = random_disp(
                shape=(1, 1, height, width),
                scales=config.scales,
                magnitude=config.magnitude,
                integrations=config.integrations,
                voxsize=config.voxsize,
                device=device,
                fractal_mode=config.fractal_mode,
            )[0]
        field_array = (
            field.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        diagnostics = validate_deformation(
            field_array,
            source_labels,
            rejection,
        )
        if diagnostics.accepted:
            return DeformationSample(
                field=field_array,
                attempts=attempt_index + 1,
                validation=diagnostics,
            )

    raise RuntimeError(
        f"failed to sample valid deformation for sample {sample_index} "
        f"after {rejection.max_attempts} attempts"
    )


def generate_deformation_fields(
    folder: Path,
    name: str,
    source_config: SourceConfig,
    source_labels: np.ndarray,
    config: DeformationConfig,
    rejection: DeformationRejectionConfig | None = None,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[Path, Path]:
    """Generate one mmap-friendly ``[N, 2, H, W]`` deformation collection."""
    folder = Path(folder)
    _validate_collection_name(name)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if rejection is None:
        rejection = DeformationRejectionConfig()
    device = torch.device(device)

    folder.mkdir(parents=True, exist_ok=True)
    preset_folder = folder / name
    fields_path = preset_folder / "fields.npy"
    config_path = preset_folder / "config.json"
    if preset_folder.exists():
        raise FileExistsError(f"deformation collection already exists: {name}")

    preset_folder.mkdir()
    (preset_folder / "rigid").mkdir()
    temporary_fields = preset_folder / ".fields.npy.tmp"
    height, width = source_config.image_size
    fields = np.lib.format.open_memmap(
        temporary_fields,
        mode="w+",
        dtype=np.float32,
        shape=(source_config.num_elements, 2, height, width),
    )
    source_labels = np.asarray(source_labels)
    attempts_per_sample: list[int] = []
    accepted_diagnostics: list[DeformationValidationResult] = []
    rejected_candidate_count = 0
    try:
        for sample_index in range(source_config.num_elements):
            sample = sample_valid_deformation(
                source_config,
                source_labels,
                config,
                rejection,
                seed=seed,
                sample_index=sample_index,
                device=device,
            )
            fields[sample_index] = sample.field
            attempts_per_sample.append(sample.attempts)
            accepted_diagnostics.append(sample.validation)
            rejected_candidate_count += sample.attempts - 1
        fields.flush()
        temporary_fields.replace(fields_path)
        write_json(
            config_path,
            {
                "name": name,
                "seed": seed,
                "generation_device": str(device),
                "sample_seed_scheme": "numpy-seed-sequence-sample-attempt-v1",
                "config": config.to_dict(),
                "rejection": rejection.to_dict(),
                "diagnostics": {
                    "rejected_candidate_count": rejected_candidate_count,
                    "attempts_per_sample": attempts_per_sample,
                    "accepted_minimum_jacobians": [
                        item.minimum_jacobian for item in accepted_diagnostics
                    ],
                    "accepted_foreground_margins_px": [
                        item.foreground_margin_px for item in accepted_diagnostics
                    ],
                },
                "array": {
                    "relative_path": "fields.npy",
                    "shape": list(fields.shape),
                    "dtype": str(fields.dtype),
                    "layout": "NCHW",
                    "channel_order": ["dy", "dx"],
                    "units": "pixels",
                    "sampling_convention": (
                        "output(x) = input(x + displacement(x))"
                    ),
                },
            },
        )
    except BaseException:
        temporary_fields.unlink(missing_ok=True)
        fields_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
        (preset_folder / "rigid").rmdir()
        preset_folder.rmdir()
        raise
    finally:
        del fields
    return fields_path, config_path


def _attempt_seed(
    collection_seed: int,
    sample_index: int,
    attempt_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [collection_seed, sample_index, attempt_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _validate_collection_name(name: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("name must be a non-empty filename component")
