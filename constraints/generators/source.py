"""Creation of a source dataset and low-level anatomy sampling."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .layer_generators import create_empty_artery
from .storage import FORMAT_NAME, FORMAT_VERSION, write_json
from .types import ArteryClass, SourceConfig


def create_source(root: Path, config: SourceConfig) -> None:
    """Initialize a new source root without generating optional child artifacts.

    Existing paths are rejected so a dataset cannot be partially overwritten or
    accidentally acquire a new identity.
    """
    root = Path(root)
    empty_artery = create_empty_artery(config.empty_artery)
    provenance = _git_provenance()

    root.mkdir(parents=True)
    (root / "layers").mkdir()
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
            },
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
