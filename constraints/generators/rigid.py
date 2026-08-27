"""Pixel-coordinate rigid transforms and parent-scoped presets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from .deformation import apply_deformation
from .storage import write_json
from .types import RigidConfig, RigidRejectionConfig, SourceConfig
from .validation import foreground_margin


@dataclass(frozen=True)
class RigidSample:
    """One accepted rigid transform and its rejection diagnostics."""

    parameters: NDArray[np.float32]
    attempts: int
    foreground_margin_px: int


def load_rigid_parameters(
    parent_folder: Path,
    name: str,
    source_config: SourceConfig,
) -> np.ndarray:
    """Mmap one rigid preset stored under its source or deformation parent."""
    _validate_name(name)
    path = Path(parent_folder) / "rigid" / f"{name}.npy"
    parameters = np.load(path, mmap_mode="r")
    expected_shape = (source_config.num_elements, 3)
    if parameters.shape != expected_shape or parameters.dtype != np.float32:
        raise ValueError(
            f"invalid rigid preset {name!r}: expected float32 "
            f"{expected_shape}, got {parameters.shape} {parameters.dtype}"
        )
    return parameters


def apply_rigid(
    values: np.ndarray,
    angle_rad: float,
    dx_px: float,
    dy_px: float,
    *,
    method: Literal["nearest", "linear"] = "nearest",
    padding_mode: Literal["zeros", "border"] = "zeros",
) -> NDArray[np.float32]:
    """Apply an intuitive forward-content rigid transform to ``[H, W]`` data.

    Positive ``dx_px`` moves content right, positive ``dy_px`` moves it down,
    and positive angles rotate content counter-clockwise around the image center.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must have shape [H, W]")
    parameters = np.asarray([angle_rad, dx_px, dy_px], dtype=np.float64)
    if not np.isfinite(parameters).all():
        raise ValueError("rigid parameters must be finite")

    height, width = values.shape
    y_out, x_out = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    x = x_out - (width - 1) / 2 - dx_px
    y = y_out - (height - 1) / 2 - dy_px
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    x_in = cosine * x - sine * y + (width - 1) / 2
    y_in = sine * x + cosine * y + (height - 1) / 2
    x_normalized = 2 * x_in / (width - 1) - 1 if width > 1 else torch.zeros_like(x_in)
    y_normalized = 2 * y_in / (height - 1) - 1 if height > 1 else torch.zeros_like(y_in)
    grid = torch.stack((x_normalized, y_normalized), dim=-1)[None]
    image = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))[None, None]
    with torch.no_grad():
        warped = torch.nn.functional.grid_sample(
            image,
            grid,
            mode="nearest" if method == "nearest" else "bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )
    return warped[0, 0].numpy().copy()


def sample_valid_rigid(
    source_labels: np.ndarray,
    config: RigidConfig,
    rejection: RigidRejectionConfig | None = None,
    *,
    seed: int,
    sample_index: int = 0,
) -> RigidSample:
    """Sample and validate one rigid transform without writing it.

    The seed scheme is identical to persisted rigid collections, so a preview
    for a given ``sample_index`` produces the parameters later stored at that
    collection index.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if rejection is None:
        rejection = RigidRejectionConfig()

    source_labels = np.asarray(source_labels)
    if source_labels.ndim != 2:
        raise ValueError("source_labels must have shape [H, W]")

    for attempt_index in range(rejection.max_attempts):
        rng = np.random.default_rng(_attempt_seed(seed, sample_index, attempt_index))
        candidate = config.sample(rng)
        rigid_labels = apply_rigid(
            source_labels,
            *candidate,
            method="nearest",
        )
        margin = foreground_margin(rigid_labels)
        if margin >= rejection.minimum_foreground_margin_px:
            return RigidSample(
                parameters=np.asarray(candidate, dtype=np.float32),
                attempts=attempt_index + 1,
                foreground_margin_px=margin,
            )

    raise RuntimeError(
        f"failed to sample valid rigid parameters for sample {sample_index} "
        f"after {rejection.max_attempts} attempts"
    )


def generate_rigid_parameters(
    parent_folder: Path,
    parent_deformation: str | None,
    name: str,
    source_config: SourceConfig,
    source_labels: np.ndarray,
    deformation_fields: np.ndarray | None,
    config: RigidConfig,
    rejection: RigidRejectionConfig | None = None,
    *,
    seed: int,
) -> tuple[Path, Path]:
    """Generate a named ``[N,3]`` rigid preset under its validation parent."""
    parent_folder = Path(parent_folder)
    _validate_name(name)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if rejection is None:
        rejection = RigidRejectionConfig()

    rigid_folder = parent_folder / "rigid"
    if not rigid_folder.is_dir():
        raise FileNotFoundError(f"missing rigid folder in {parent_folder}")
    parameters_path = rigid_folder / f"{name}.npy"
    config_path = rigid_folder / f"{name}.json"
    if parameters_path.exists() or config_path.exists():
        raise FileExistsError(f"rigid preset already exists: {name}")

    temporary_parameters = rigid_folder / f".{name}.npy.tmp"
    parameters = np.lib.format.open_memmap(
        temporary_parameters,
        mode="w+",
        dtype=np.float32,
        shape=(source_config.num_elements, 3),
    )
    source_labels = np.asarray(source_labels)
    attempts_per_sample: list[int] = []
    accepted_margins: list[int] = []
    rejected_candidate_count = 0
    try:
        for sample_index in range(source_config.num_elements):
            deformed_labels = source_labels
            if deformation_fields is not None:
                deformed_labels = np.rint(
                    apply_deformation(
                        source_labels,
                        deformation_fields[sample_index],
                        method="nearest",
                    )
                ).astype(np.uint8)

            sample = sample_valid_rigid(
                deformed_labels,
                config,
                rejection,
                seed=seed,
                sample_index=sample_index,
            )
            parameters[sample_index] = sample.parameters
            attempts_per_sample.append(sample.attempts)
            accepted_margins.append(sample.foreground_margin_px)
            rejected_candidate_count += sample.attempts - 1

        parameters.flush()
        temporary_parameters.replace(parameters_path)
        write_json(
            config_path,
            {
                "name": name,
                "parent_deformation": parent_deformation,
                "seed": seed,
                "sample_seed_scheme": "numpy-seed-sequence-sample-attempt-v1",
                "config": config.to_dict(),
                "rejection": rejection.to_dict(),
                "diagnostics": {
                    "rejected_candidate_count": rejected_candidate_count,
                    "attempts_per_sample": attempts_per_sample,
                    "accepted_foreground_margins_px": accepted_margins,
                },
                "array": {
                    "relative_path": f"{name}.npy",
                    "shape": list(parameters.shape),
                    "dtype": str(parameters.dtype),
                    "columns": ["angle_rad", "dx_px", "dy_px"],
                },
                "convention": {
                    "angle": "positive counter-clockwise around image center",
                    "dx": "positive moves content right",
                    "dy": "positive moves content down",
                    "units": {"angle": "radians", "translation": "pixels"},
                },
            },
        )
    except BaseException:
        temporary_parameters.unlink(missing_ok=True)
        parameters_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
        raise
    finally:
        del parameters
    return parameters_path, config_path


def _attempt_seed(seed: int, sample_index: int, attempt_index: int) -> int:
    sequence = np.random.SeedSequence([seed, sample_index, attempt_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _validate_name(name: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("name must be a non-empty filename component")
