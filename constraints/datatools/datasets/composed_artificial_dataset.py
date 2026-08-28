"""Lazy composition of independently stored artificial source layers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from constraints.datatools.datasets.artery_common_types import (
    ArtificialMaskColor,
    ArtificialMaskLabel,
)
from constraints.generators.deformation import load_deformation_fields
from constraints.generators.factories import (
    compose_artificial_sample,
    get_source_config,
)
from constraints.generators.layer_generators import (
    LayerPatch,
    SavedLayer,
    load_layer_collection,
)
from constraints.generators.recipes import Recipe
from constraints.generators.rigid import load_rigid_parameters
from constraints.generators.storage import read_manifest
from constraints.generators.types import NoiseConfig

from ..label_schema import LabelSchema
from .base_dataset import PerSampleDataset
from .types import Sample

if TYPE_CHECKING:
    from constraints.generators.sdf_cache import SDFCacheConfig, SDFCacheIdentity

# _LABEL_SCHEMA = LabelSchema.from_lists(
#     names=["background", "boundary", "lumen", "plaque"],
#     colors=[
#         (0.0, 0.0, 0.0),
#         (0.9, 0.1, 0.1),
#         (0.1, 0.7, 0.1),
#         (0.1, 0.35, 0.95),
#     ],
# )


class ComposedArtificialDataset(PerSampleDataset):
    """Compose an ordered selection of stored artifacts lazily."""

    def __init__(
        self,
        root: Path,
        *,
        layers: Sequence[SavedLayer] = (),
        deformation: str | None = None,
        rigid: str | None = None,
        noise: NoiseConfig | None = None,
        sample_list: list[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.recipe = Recipe(
            layers=tuple(layers),
            deformation=deformation,
            rigid=rigid,
            noise=noise,
        )
        self.recipe.require_resolved()
        self.config = get_source_config(self.root)
        self._empty_artery = np.load(self.root / "empty_artery.npy", mmap_mode="r")
        self._deformation_fields = (
            None
            if self.recipe.deformation_name is None
            else load_deformation_fields(
                self.root / "deformations",
                self.recipe.deformation_name,
                self.config,
            )
        )
        self._rigid_parameters = (
            None
            if self.recipe.rigid_name is None
            else load_rigid_parameters(
                self.root
                if self.recipe.deformation_name is None
                else self.root / "deformations" / self.recipe.deformation_name,
                self.recipe.rigid_name,
                self.config,
            )
        )
        self._layer_collections = tuple(
            (layer, load_layer_collection(self.root, layer.name, self.config))
            for layer in self.recipe.layers
        )
        self._sample_list = (
            sample_list
            if sample_list is not None
            else list(range(self.config.num_elements))
        )

    @classmethod
    def from_recipe(
        cls, root: Path, recipe: Recipe, sample_list: list[int] | None = None
    ) -> ComposedArtificialDataset:
        """Load a dataset from an explicit artifact-selection recipe."""
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe instance")
        dataset = cls(
            root,
            layers=recipe.layers,
            deformation=recipe.deformation,
            rigid=recipe.rigid,
            noise=recipe.noise,
            sample_list=sample_list,
        )
        dataset.recipe = recipe
        return dataset

    def __len__(self) -> int:
        return len(self._sample_list)

    def __getitem__(self, index: int) -> Sample:
        index = self._normalize_index(index)
        layers = [
            LayerPatch(collection.labels[index], collection.image[index])
            for _, collection in self._layer_collections
        ]
        field = (
            None
            if self._deformation_fields is None
            else self._deformation_fields[index]
        )
        rigid_parameters = (
            None if self._rigid_parameters is None else self._rigid_parameters[index]
        )
        arrays = compose_artificial_sample(
            self._empty_artery,
            layers,
            deformation_field=field,
            rigid_parameters=rigid_parameters,
            noise_config=self.recipe.noise,
            sample_index=index,
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
        return LabelSchema.from_lists(
            names=ArtificialMaskLabel, colors=ArtificialMaskColor
        )

    def sdf_cache_identity(
        self,
        config: SDFCacheConfig | None = None,
    ) -> SDFCacheIdentity:
        """Describe the pre-rigid SDF cache shared by compatible recipes."""
        from constraints.generators.sdf_cache import SDFCacheIdentity

        manifest = read_manifest(self.root)
        dataset_id = manifest.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("source manifest has no dataset_id")
        return SDFCacheIdentity.from_recipe(dataset_id, self.recipe, config)

    def _normalize_index(self, index: int) -> int:
        return self._sample_list[index]
