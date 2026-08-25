"""Validated, single-writer storage primitives for format-v1 artifacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_NAME = "composable-artificial-artery"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relative_path: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def open_artifact(root: Path, spec: ArtifactSpec, mode: str = "w+") -> np.memmap:
    path = root / spec.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path, mode=mode, dtype=np.dtype(spec.dtype), shape=spec.shape
    )


def validate_artifact(root: Path, spec: ArtifactSpec) -> np.ndarray:
    array = np.load(root / spec.relative_path, mmap_mode="r")
    if array.shape != spec.shape or array.dtype != np.dtype(spec.dtype):
        raise ValueError(
            f"invalid artifact {spec.relative_path}: {array.shape} {array.dtype}"
        )
    return array


def manifest_artifacts(specs: list[ArtifactSpec]) -> dict[str, dict[str, Any]]:
    return {spec.name: asdict(spec) for spec in specs}


def read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("format_name") != FORMAT_NAME
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported artificial dataset format")
    if manifest.get("status") != "complete":
        raise ValueError("dataset is incomplete")
    return manifest
