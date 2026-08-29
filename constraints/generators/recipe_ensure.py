"""Two-phase materialization of all stored artifacts required by a recipe."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constraints.datatools.datasets.composed_artificial_dataset import (
    ComposedArtificialDataset,
)
from constraints.devices import DeviceSelection

from .artifact_metadata import definition_differences, read_artifact_definition
from .deformation import load_deformation_fields
from .factories import (
    create_deformation_collection,
    create_layer_collection,
    create_rigid_collection,
    get_source_config,
)
from .layer_generators import load_layer_collection, resolve_layer_patch
from .recipe_backups import DeformationBackup, LayerBackup, RigidBackup
from .recipes import Recipe
from .rigid import load_rigid_parameters
from .sdf_cache import create_sdf_cache


@dataclass(frozen=True)
class EnsureReport:
    created: tuple[str, ...]
    replaced: tuple[str, ...]
    reused: tuple[str, ...]
    sdf_cache: Path | None = None


@dataclass(frozen=True)
class _Action:
    operation: str
    kind: str
    name: str
    backup: Any | None
    paths: tuple[Path, ...]
    metadata_path: Path

    @property
    def label(self) -> str:
        return f"{self.kind} {self.name!r}"


def ensure_recipe(
    recipe: Recipe,
    source_root: Path | None = None,
    *,
    overwrite: bool = False,
    device: DeviceSelection = "auto",
    sdf_batch_size: int = 16,
    progress: bool = False,
) -> EnsureReport:
    """Preflight the complete recipe, then execute the resulting action plan."""
    root = recipe.resolve_source_root(source_root)
    recipe.require_resolved()
    source_config = get_source_config(root)
    actions, errors = _preflight(recipe, root, source_config, overwrite)
    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(
            f"recipe cannot be ensured; no changes were made:\n{details}"
        )

    geometry_changes = any(
        item.operation in {"create", "replace"}
        and item.kind in {"layer", "deformation"}
        for item in actions
    )
    created: list[str] = []
    replaced: list[str] = []
    reused: list[str] = []
    for action in actions:
        if action.operation == "reuse":
            reused.append(action.label)
            continue
        if action.operation == "replace":
            _remove_action(action)
            replaced.append(action.label)
        else:
            created.append(action.label)
        _create_action(action, recipe, root, device, progress)

    if geometry_changes:
        _invalidate_sdf_caches(root)
    sdf_path = _ensure_sdf(recipe, root, sdf_batch_size, device, progress)
    return EnsureReport(
        created=tuple(created),
        replaced=tuple(replaced),
        reused=tuple(reused),
        sdf_cache=sdf_path,
    )


def _preflight(recipe, root, source_config, overwrite):
    actions: list[_Action] = []
    errors: list[str] = []
    for layer in recipe.layers:
        name = layer.name
        assert name is not None
        folder = root / "layers"
        if layer.backup is not None:
            try:
                resolve_layer_patch(layer.backup, root, source_config, 0)
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(f"layer {name!r} has an invalid backup: {error}")
        actions.append(
            _assess(
                kind="layer",
                name=name,
                backup=layer.backup,
                paths=(folder / name / "labels.npy", folder / name / "image.npy"),
                metadata_path=folder / f"{name}.manifest.json",
                parse=LayerBackup.from_dict,
                validate=lambda name=name: load_layer_collection(
                    root, name, source_config
                ),
                overwrite=overwrite,
                errors=errors,
            )
        )

    deformation_replaced = False
    if recipe.deformation is not None:
        name = recipe.deformation.name
        assert name is not None
        folder = root / "deformations"
        action = _assess(
            kind="deformation",
            name=name,
            backup=recipe.deformation.backup,
            paths=(folder / name / "fields.npy", folder / name / "config.json"),
            metadata_path=folder / f"{name}.manifest.json",
            parse=DeformationBackup.from_dict,
            validate=lambda: load_deformation_fields(folder, name, source_config),
            overwrite=overwrite,
            errors=errors,
        )
        actions.append(action)
        deformation_replaced = action.operation in {"create", "replace"}

    if recipe.rigid is not None:
        name = recipe.rigid.name
        assert name is not None
        parent = (
            root
            if recipe.deformation_name is None
            else root / "deformations" / recipe.deformation_name
        )
        folder = parent / "rigid"
        if deformation_replaced:
            if recipe.rigid.backup is None:
                errors.append(
                    f"rigid {name!r} will be invalidated with its deformation "
                    "and has no backup"
                )
            action = _Action(
                "create",
                "rigid",
                name,
                recipe.rigid.backup,
                (folder / f"{name}.npy", folder / f"{name}.json"),
                folder / f"{name}.manifest.json",
            )
        else:
            action = _assess(
                kind="rigid",
                name=name,
                backup=recipe.rigid.backup,
                paths=(folder / f"{name}.npy", folder / f"{name}.json"),
                metadata_path=folder / f"{name}.manifest.json",
                parse=RigidBackup.from_dict,
                validate=lambda: load_rigid_parameters(parent, name, source_config),
                overwrite=overwrite,
                errors=errors,
            )
        actions.append(action)
    return actions, errors


def _assess(
    *,
    kind: str,
    name: str,
    backup: Any | None,
    paths: tuple[Path, ...],
    metadata_path: Path,
    parse: Callable[[dict[str, Any]], Any],
    validate: Callable[[], Any],
    overwrite: bool,
    errors: list[str],
) -> _Action:
    existence = tuple(path.exists() for path in paths)
    any_exists = any(existence) or metadata_path.exists()
    complete_files = all(existence)
    if not any_exists:
        if backup is None:
            errors.append(f"{kind} {name!r} is missing and has no backup")
        return _Action("create", kind, name, backup, paths, metadata_path)
    if not complete_files:
        if not overwrite or backup is None:
            errors.append(
                f"{kind} {name!r} is partial; pass overwrite=True with a backup"
            )
        return _Action("replace", kind, name, backup, paths, metadata_path)
    try:
        validate()
    except (OSError, ValueError) as error:
        if not overwrite or backup is None:
            errors.append(f"{kind} {name!r} is invalid: {error}")
        return _Action("replace", kind, name, backup, paths, metadata_path)
    if backup is None:
        return _Action("reuse", kind, name, None, paths, metadata_path)
    try:
        stored = parse(
            read_artifact_definition(metadata_path, kind=f"{kind}-collection")
        )
    except (OSError, ValueError) as error:
        if not overwrite:
            errors.append(f"{kind} {name!r} has unverifiable metadata: {error}")
        return _Action("replace", kind, name, backup, paths, metadata_path)
    differences = definition_differences(stored, backup)
    if not differences:
        return _Action("reuse", kind, name, backup, paths, metadata_path)
    if not overwrite:
        formatted = "; ".join(differences)
        errors.append(f"{kind} {name!r} has a different definition: {formatted}")
    return _Action("replace", kind, name, backup, paths, metadata_path)


def _remove_action(action: _Action) -> None:
    if action.kind in {"layer", "deformation"}:
        folder = action.paths[0].parent
        if folder.exists():
            shutil.rmtree(folder)
    else:
        for path in action.paths:
            path.unlink(missing_ok=True)
    action.metadata_path.unlink(missing_ok=True)


def _create_action(action, recipe, root, device, progress) -> None:
    if action.backup is None:
        return
    if action.kind == "layer":
        create_layer_collection(
            root, action.name, action.backup, progress=progress
        )
    elif action.kind == "deformation":
        create_deformation_collection(
            root,
            action.name,
            action.backup.config,
            action.backup.rejection,
            seed=action.backup.seed,
            device=device,
            progress=progress,
        )
    else:
        create_rigid_collection(
            root,
            action.name,
            action.backup.config,
            action.backup.rejection,
            deformation=recipe.deformation_name,
            seed=action.backup.seed,
            progress=progress,
        )


def _invalidate_sdf_caches(root: Path) -> None:
    derived = root / "derived"
    if not derived.is_dir():
        return
    for path in derived.glob("sdf-v*"):
        if path.is_dir():
            shutil.rmtree(path)


def _ensure_sdf(recipe, root, batch_size, device, progress) -> Path | None:
    if recipe.sdf_cache is None:
        return None
    geometry_recipe = Recipe(
        source=recipe.source,
        layers=recipe.layers,
        deformation=recipe.deformation,
    )
    dataset = ComposedArtificialDataset.from_recipe(root, geometry_recipe)
    identity = dataset.sdf_cache_identity(recipe.sdf_cache)
    folder = identity.cache_directory(root)
    if folder.exists():
        manifest = json.loads((folder / "manifest.json").read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("cache_key") != identity.digest
        ):
            raise RuntimeError(f"incomplete or incompatible SDF cache: {folder}")
        return folder / "sdf.npy"
    array_path, _ = create_sdf_cache(
        dataset,
        recipe.sdf_cache,
        batch_size=batch_size,
        device=device,
        progress=progress,
    )
    return array_path
