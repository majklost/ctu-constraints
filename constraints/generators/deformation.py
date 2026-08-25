"""Generation of independent, named deformation-field collections."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from constraints.voxelmorph.utils import random_disp, spatial_transform

from .storage import write_json
from .types import DeformationConfig, DeformationRejectionConfig, SourceConfig
from .validation import DeformationValidationResult, validate_deformation


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


def generate_deformation_fields(
    folder: Path,
    name: str,
    source_config: SourceConfig,
    source_labels: np.ndarray,
    config: DeformationConfig,
    rejection: DeformationRejectionConfig | None = None,
    *,
    seed: int,
) -> tuple[Path, Path]:
    """Generate one mmap-friendly ``[N, 2, H, W]`` deformation collection."""
    folder = Path(folder)
    _validate_collection_name(name)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if rejection is None:
        rejection = DeformationRejectionConfig()

    folder.mkdir(parents=True, exist_ok=True)
    fields_path = folder / f"{name}.npy"
    config_path = folder / f"{name}.json"
    if fields_path.exists() or config_path.exists():
        raise FileExistsError(f"deformation collection already exists: {name}")

    temporary_fields = folder / f".{name}.npy.tmp"
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
            for attempt_index in range(rejection.max_attempts):
                attempt_seed = _attempt_seed(seed, sample_index, attempt_index)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(attempt_seed)
                    field = random_disp(
                        shape=(1, 1, height, width),
                        scales=config.scales,
                        magnitude=config.magnitude,
                        integrations=config.integrations,
                        voxsize=config.voxsize,
                        device=torch.device("cpu"),
                        fractal_mode=config.fractal_mode,
                    )[0]
                field_array = field.detach().numpy().astype(np.float32, copy=False)
                diagnostics = validate_deformation(
                    field_array,
                    source_labels,
                    rejection,
                )
                if diagnostics.accepted:
                    fields[sample_index] = field_array
                    attempts_per_sample.append(attempt_index + 1)
                    accepted_diagnostics.append(diagnostics)
                    break
                rejected_candidate_count += 1
            else:
                raise RuntimeError(
                    f"failed to sample valid deformation for sample "
                    f"{sample_index} after {rejection.max_attempts} attempts"
                )
        fields.flush()
        temporary_fields.replace(fields_path)
        write_json(
            config_path,
            {
                "name": name,
                "seed": seed,
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
                    "relative_path": f"{name}.npy",
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
