"""Small metadata contract shared by generated recipe artifacts."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .storage import write_json


def write_artifact_metadata(
    path: Path,
    *,
    kind: str,
    name: str,
    definition: dict[str, Any],
    status: str,
) -> None:
    write_json(
        path,
        {
            "format_name": f"composed-artificial-{kind}",
            "format_version": 1,
            "status": status,
            "name": name,
            "definition": definition,
        },
    )


def read_artifact_definition(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid artifact metadata: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("format_name") != f"composed-artificial-{kind}"
        or value.get("format_version") != 1
        or value.get("status") != "complete"
        or not isinstance(value.get("definition"), dict)
    ):
        raise ValueError(f"incomplete or invalid artifact metadata: {path}")
    return value["definition"]


def definition_differences(stored: Any, requested: Any, path: str = "") -> list[str]:
    """Return readable leaf differences between typed definitions."""
    if stored == requested:
        return []
    if type(stored) is not type(requested):
        return [f"{path or 'definition'}: stored={stored!r}, requested={requested!r}"]
    if is_dataclass(stored):
        differences: list[str] = []
        for item in fields(stored):
            child = f"{path}.{item.name}" if path else item.name
            differences.extend(
                definition_differences(
                    getattr(stored, item.name), getattr(requested, item.name), child
                )
            )
        return differences
    if isinstance(stored, Mapping):
        differences = []
        for key in sorted(stored.keys() | requested.keys(), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in stored:
                differences.append(f"{child}: missing in stored definition")
            elif key not in requested:
                differences.append(f"{child}: missing in requested definition")
            else:
                differences.extend(
                    definition_differences(stored[key], requested[key], child)
                )
        return differences
    if isinstance(stored, Sequence) and not isinstance(stored, (str, bytes)):
        differences = []
        if len(stored) != len(requested):
            differences.append(
                f"{path}: stored length={len(stored)}, "
                f"requested length={len(requested)}"
            )
        for index, (left, right) in enumerate(zip(stored, requested, strict=False)):
            differences.extend(definition_differences(left, right, f"{path}[{index}]"))
        return differences
    return [f"{path}: stored={stored!r}, requested={requested!r}"]
