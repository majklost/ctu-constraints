"""Serializable selections for lazily composed artificial datasets."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

from .rendering import DEFAULT_CLASS_INTENSITIES
from .storage import write_json
from .types import AppearanceKind, ArteryClass, SavedPlaque

RECIPE_FORMAT_NAME = "composed-artificial-recipe"
RECIPE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Recipe:
    """Complete reproducible selection used to compose a dataset."""

    plaques: tuple[SavedPlaque, ...] = ()
    deformation: str | None = None
    rigid: str | None = None
    class_intensities: Mapping[AppearanceKind, float] = field(
        default_factory=lambda: DEFAULT_CLASS_INTENSITIES
    )

    format_version: ClassVar[int] = RECIPE_FORMAT_VERSION

    def __post_init__(self) -> None:
        plaques = tuple(self.plaques)
        if not all(isinstance(plaque, SavedPlaque) for plaque in plaques):
            raise TypeError("recipe plaques must contain SavedPlaque instances")
        names = [plaque.name for plaque in plaques]
        if len(names) != len(set(names)):
            raise ValueError("a recipe cannot contain a plaque collection twice")
        object.__setattr__(self, "plaques", plaques)

        for field_name in ("deformation", "rigid"):
            name = getattr(self, field_name)
            if name is not None and (
                not isinstance(name, str) or not name or Path(name).name != name
            ):
                raise ValueError(f"{field_name} name must be a filename component")

        try:
            intensities = {
                AppearanceKind(kind): float(intensity)
                for kind, intensity in self.class_intensities.items()
            }
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "class_intensities must map appearance kinds to numbers"
            ) from error
        if not all(isfinite(intensity) for intensity in intensities.values()):
            raise ValueError("class intensities must be finite")

        required_appearances = {
            AppearanceKind.BACKGROUND,
            AppearanceKind.BOUNDARY,
            AppearanceKind.LUMEN,
        }
        required_appearances.update(
            plaque.appearance
            if plaque.appearance is not None
            else AppearanceKind(plaque.target_class.value)
            for plaque in plaques
        )
        missing = required_appearances - intensities.keys()
        if missing:
            names = ", ".join(sorted(kind.name for kind in missing))
            raise ValueError(f"missing class intensities for: {names}")
        ordered = dict(sorted(intensities.items(), key=lambda item: item[0].value))
        object.__setattr__(self, "class_intensities", MappingProxyType(ordered))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, human-readable JSON representation."""
        return {
            "format_name": RECIPE_FORMAT_NAME,
            "format_version": self.format_version,
            "plaques": [
                {
                    "name": plaque.name,
                    "target_class": plaque.target_class.name.lower(),
                    "appearance": (
                        None
                        if plaque.appearance is None
                        else plaque.appearance.name.lower()
                    ),
                }
                for plaque in self.plaques
            ],
            "deformation": self.deformation,
            "rigid": self.rigid,
            "class_intensities": {
                kind.name.lower(): intensity
                for kind, intensity in self.class_intensities.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Decode and strictly validate a recipe JSON object."""
        expected = {
            "format_name",
            "format_version",
            "plaques",
            "deformation",
            "rigid",
            "class_intensities",
        }
        if not isinstance(value, dict) or value.keys() != expected:
            raise ValueError("invalid Recipe fields")
        if value["format_name"] != RECIPE_FORMAT_NAME:
            raise ValueError("unsupported recipe format")
        if value["format_version"] != RECIPE_FORMAT_VERSION:
            raise ValueError("unsupported recipe format version")
        plaques_value = value["plaques"]
        intensities_value = value["class_intensities"]
        plaque_fields = {"name", "target_class", "appearance"}
        if not isinstance(plaques_value, list) or any(
            not isinstance(plaque, dict) or plaque.keys() != plaque_fields
            for plaque in plaques_value
        ):
            raise ValueError("invalid Recipe plaques")
        if not isinstance(intensities_value, dict):
            raise ValueError("invalid Recipe class_intensities")
        try:
            plaques = tuple(
                SavedPlaque(
                    name=plaque["name"],
                    target_class=ArteryClass[plaque["target_class"].upper()],
                    appearance=(
                        None
                        if plaque["appearance"] is None
                        else AppearanceKind[plaque["appearance"].upper()]
                    ),
                )
                for plaque in plaques_value
            )
            intensities = {
                AppearanceKind[name.upper()]: intensity
                for name, intensity in intensities_value.items()
            }
            return cls(
                plaques=plaques,
                deformation=value["deformation"],
                rigid=value["rigid"],
                class_intensities=intensities,
            )
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("invalid Recipe value") from error

    def save_json(self, path: Path) -> None:
        """Atomically save this recipe as formatted JSON."""
        write_json(Path(path), self.to_dict())

    @classmethod
    def load_json(cls, path: Path) -> Self:
        """Load and validate a recipe JSON file."""
        try:
            value = json.loads(Path(path).read_text())
        except json.JSONDecodeError as error:
            raise ValueError("invalid Recipe JSON") from error
        return cls.from_dict(value)
