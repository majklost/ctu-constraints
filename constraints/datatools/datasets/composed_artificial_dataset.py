"""Lazy composition of independently stored artificial source layers."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from constraints.generators.deformation import (
    apply_deformation,
    load_deformation_fields,
)
from constraints.generators.factories import (
    ComposedSampleArrays,
    compose_artificial_sample,
    get_source_config,
)
from constraints.generators.rendering import DEFAULT_CLASS_INTENSITIES
from constraints.generators.rigid import apply_rigid, load_rigid_parameters
from constraints.generators.types import ArteryClass, PlaqueLayer

from ..label_schema import LabelSchema
from .base_dataset import PerSampleDataset
from .types import Sample

_LABEL_SCHEMA = LabelSchema.from_lists(
    names=["background", "boundary", "lumen", "plaque"],
    colors=[
        (0.0, 0.0, 0.0),
        (0.9, 0.1, 0.1),
        (0.1, 0.7, 0.1),
        (0.1, 0.35, 0.95),
    ],
)


class ComposedArtificialDataset(PerSampleDataset):
    """Compose selected plaque collections lazily for each source sample.

    ``plaques`` contains names of real-plaque collections. ``fake_plaques``
    maps collection names to the anatomical class they imitate in the target.
    """

    def __init__(
        self,
        root: Path,
        *,
        plaques: Sequence[str] = (),
        fake_plaques: Mapping[str, ArteryClass] | None = None,
        deformation: str | None = None,
        rigid: str | None = None,
        class_intensities: Mapping[ArteryClass, float] = DEFAULT_CLASS_INTENSITIES,
    ) -> None:
        self.root = Path(root)
        self.config = get_source_config(self.root)
        self._empty_artery = np.load(self.root / "empty_artery.npy", mmap_mode="r")
        self._class_intensities = dict(class_intensities)
        self._deformation_name = deformation
        self._deformation_fields = (
            None
            if deformation is None
            else load_deformation_fields(
                self.root / "deformations",
                deformation,
                self.config,
            )
        )
        self._rigid_parameters = (
            None
            if rigid is None
            else load_rigid_parameters(
                self.root
                if deformation is None
                else self.root / "deformations" / deformation,
                rigid,
                self.config,
            )
        )

        fake_plaques = {} if fake_plaques is None else dict(fake_plaques)
        duplicate_names = set(plaques) & fake_plaques.keys()
        if duplicate_names:
            raise ValueError(
                f"collections cannot be both real and fake: {sorted(duplicate_names)}"
            )

        self._real_masks = {
            name: self._load_plaque_collection(name) for name in plaques
        }
        self._fake_masks: dict[str, tuple[np.ndarray, ArteryClass]] = {}
        for name, target_class in fake_plaques.items():
            target_class = ArteryClass(target_class)
            if target_class not in {ArteryClass.BOUNDARY, ArteryClass.LUMEN}:
                raise ValueError("fake plaques must resolve to boundary or lumen")
            self._fake_masks[name] = (
                self._load_plaque_collection(name),
                target_class,
            )

    def __len__(self) -> int:
        return self.config.num_elements

    def __getitem__(self, index: int) -> Sample:
        index = self._normalize_index(index)
        layers = [
            PlaqueLayer(masks[index], target_class)
            for masks, target_class in self._fake_masks.values()
        ]
        layers.extend(
            PlaqueLayer(masks[index], ArteryClass.PLAQUE)
            for masks in self._real_masks.values()
        )
        empty_artery = self._empty_artery
        field = None
        if self._deformation_fields is not None:
            field = self._deformation_fields[index]
            empty_artery = np.rint(
                apply_deformation(empty_artery, field, method="nearest")
            ).astype(np.uint8)
            layers = [
                PlaqueLayer(
                    apply_deformation(layer.mask, field, method="nearest") > 0.5,
                    layer.target_class,
                )
                for layer in layers
            ]
        arrays = compose_artificial_sample(
            empty_artery,
            layers,
            self._class_intensities,
        )
        rigid_parameters = None
        if self._rigid_parameters is not None:
            rigid_parameters = self._rigid_parameters[index]
            arrays = ComposedSampleArrays(
                image=apply_rigid(
                    arrays.image,
                    *rigid_parameters,
                    method="linear",
                ),
                target_labels=np.rint(
                    apply_rigid(
                        arrays.target_labels,
                        *rigid_parameters,
                        method="nearest",
                    )
                ).astype(np.uint8),
            )
        sample = Sample(
            image=torch.from_numpy(arrays.image[None]),
            target_labels=torch.from_numpy(arrays.target_labels).long(),
            sample_id=str(index),
        )
        if field is not None:
            sample["transform"] = torch.from_numpy(np.array(field, copy=True))
        if rigid_parameters is not None:
            sample["rigid"] = torch.from_numpy(np.array(rigid_parameters, copy=True))
        return sample

    @property
    def label_schema(self) -> LabelSchema:
        return _LABEL_SCHEMA

    def _load_plaque_collection(self, name: str) -> np.ndarray:
        if not name or Path(name).name != name:
            raise ValueError("plaque collection name must be a filename component")
        path = self.root / "plaques" / f"{name}.npy"
        masks = np.load(path, mmap_mode="r")
        expected_shape = (
            self.config.num_elements,
            *self.config.empty_artery.image_size,
        )
        if masks.shape != expected_shape or masks.dtype != np.bool_:
            raise ValueError(
                f"invalid plaque collection {name!r}: expected Boolean "
                f"{expected_shape}, got {masks.shape} {masks.dtype}"
            )
        return masks

    def _normalize_index(self, index: int) -> int:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index
