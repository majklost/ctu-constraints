"""Creation of a source dataset and independent plaque-mask collections."""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .parametrization.plaque_generators import (
    create_empty_artery,
    create_plaque_mask,
    create_power_plaque,
)
from .parametrization.plaque_samplers import sample_power_plaque_parameter_batch
from .storage import FORMAT_NAME, FORMAT_VERSION, write_json
from .types import (
    ArteryClass,
    PowerPlaqueParameters,
    PowerPlaqueSamplingRanges,
    SourceConfig,
)


@dataclass(frozen=True)
class PowerPlaqueSample:
    """One generated plaque mask and the parameters that produced it."""

    mask: np.ndarray
    parameters: tuple[PowerPlaqueParameters, ...]
    sample_seed: int


def create_source(root: Path, config: SourceConfig) -> None:
    """Initialize a new source root without generating optional child artifacts.

    Existing paths are rejected so a dataset cannot be partially overwritten or
    accidentally acquire a new identity.
    """
    root = Path(root)
    empty_artery = create_empty_artery(config.empty_artery, config.image_size)
    provenance = _git_provenance()

    root.mkdir(parents=True)
    (root / "plaques").mkdir()
    (root / "deformations").mkdir()
    (root / "rigid").mkdir()
    np.save(root / "empty_artery.npy", empty_artery, allow_pickle=False)
    write_json(root / "source_config.json", config.to_dict())

    manifest = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "dataset_id": str(uuid4()),
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "classes": {member.name.lower(): int(member) for member in ArteryClass},
        "artifacts": {
            "source_config": {"relative_path": "source_config.json"},
            "empty_artery": {
                "relative_path": "empty_artery.npy",
                "shape": list(empty_artery.shape),
                "dtype": str(empty_artery.dtype),
            }
        },
        **provenance,
    }
    write_json(root / "manifest.json", manifest)


def load_source_config(root: Path) -> SourceConfig:
    """Load the canonical configuration required by source child artifacts."""
    path = Path(root) / "source_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise ValueError("source_config.json must contain a JSON object")
    return SourceConfig.from_dict(value)


def sample_power_plaque_mask(
    config: SourceConfig,
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
    sample_index: int = 0,
) -> PowerPlaqueSample:
    """Generate one reproducible power-plaque mask without writing it.

    ``seed`` and ``sample_index`` use the same scheme as persisted plaque
    collections, so a preview can be reproduced exactly during generation.
    """
    if config.plaque_generation_method != "power":
        raise ValueError("source config does not select power plaque generation")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if ranges is None:
        ranges = PowerPlaqueSamplingRanges()
    if isinstance(ranges, tuple) and not ranges:
        raise ValueError("at least one plaque range is required")

    sample_seed = _sample_seed(seed, sample_index)
    rng = np.random.default_rng(sample_seed)
    plaque_count = len(ranges) if isinstance(ranges, tuple) else 1
    parameters = sample_power_plaque_parameter_batch(
        ranges,
        plaque_count,
        config.empty_artery,
        rng,
    )
    plaques = tuple(
        create_power_plaque(
            item,
            lumen_radius_px=config.empty_artery.lumen_radius_px,
        )
        for item in parameters
    )
    return PowerPlaqueSample(
        mask=create_plaque_mask(plaques, config),
        parameters=parameters,
        sample_seed=sample_seed,
    )


def generate_plaque_masks_power(
    folder: Path,
    name: str,
    config: SourceConfig,
    ranges: PowerPlaqueSamplingRanges
    | tuple[PowerPlaqueSamplingRanges, ...]
    | None = None,
    *,
    seed: int,
) -> None:
    """Generate a named, reproducible collection of power-plaque union masks.

    A single range creates one plaque in every mask. A tuple creates one plaque
    per range and stores their union while retaining every resolved parameter in
    the paired JSONL record.
    """
    folder = Path(folder)
    _validate_artifact_name(name)
    if config.plaque_generation_method != "power":
        raise ValueError("source config does not select power plaque generation")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if ranges is None:
        ranges = PowerPlaqueSamplingRanges()
    if isinstance(ranges, tuple) and not ranges:
        raise ValueError("at least one plaque range is required")

    folder.mkdir(parents=True, exist_ok=True)
    masks_path = folder / f"{name}.npy"
    parameters_path = folder / f"{name}.jsonl"
    if masks_path.exists() or parameters_path.exists():
        raise FileExistsError(f"plaque collection already exists: {name}")

    temporary_masks = folder / f".{name}.npy.tmp"
    temporary_parameters = folder / f".{name}.jsonl.tmp"
    masks = np.lib.format.open_memmap(
        temporary_masks,
        mode="w+",
        dtype=np.bool_,
        shape=(config.num_elements, *config.image_size),
    )
    try:
        with temporary_parameters.open("w", encoding="utf-8") as stream:
            for sample_index in range(config.num_elements):
                sample = sample_power_plaque_mask(
                    config,
                    ranges,
                    seed=seed,
                    sample_index=sample_index,
                )
                masks[sample_index] = sample.mask
                record = {
                    "sample_index": sample_index,
                    "sample_seed": sample.sample_seed,
                    "plaques": [
                        {"type": "power", "parameters": asdict(item)}
                        for item in sample.parameters
                    ],
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        masks.flush()
        temporary_masks.replace(masks_path)
        temporary_parameters.replace(parameters_path)
    except BaseException:
        temporary_masks.unlink(missing_ok=True)
        temporary_parameters.unlink(missing_ok=True)
        raise
    finally:
        del masks


def _sample_seed(collection_seed: int, sample_index: int) -> int:
    sequence = np.random.SeedSequence([collection_seed, sample_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _validate_artifact_name(name: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("name must be a non-empty filename component")


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}
