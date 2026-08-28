"""Dataset-independent descriptions of semantic label channels."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import torch
import torch.nn.functional

from constraints.datatools.datasets.artery_common_types import (
    ArtificialMaskColor,
    ArtificialMaskLabel,
)


@dataclass(frozen=True)
class LabelSchema:
    """Immutable semantic description shared by models and data components.

    Class IDs are the values stored in a class-index target.  For channel
    tensors, channels are expected to be ordered by these IDs.  The channel
    dimension may be either ``0`` (``[C, ...]``) or ``1`` (``[B, C, ...]``).
    """

    names: Mapping[int, str]
    colors: Mapping[int, tuple[float, float, float]]
    background_id: int = 0
    ignore_index: int | None = None

    @classmethod
    def from_lists(
        cls,
        names: list[str],
        colors: list[tuple[float, float, float]],
        background_id: int = 0,
        ignore_index: int | None = None,
    ) -> Self:
        return cls(
            names=dict(enumerate(names)),
            colors=dict(enumerate(colors)),
            background_id=background_id,
            ignore_index=ignore_index,
        )

    def __post_init__(self) -> None:
        names = dict(self.names)
        colors = dict(self.colors)
        ids = set(names)

        if not ids:
            raise ValueError("LabelSchema must define at least one class.")
        if ids != set(range(len(ids))):
            raise ValueError("Label IDs must be contiguous and start at zero.")
        if self.background_id not in ids:
            raise ValueError("background_id must refer to a defined class.")
        if set(colors) != ids:
            raise ValueError("names and colors must define the same class IDs.")
        if self.ignore_index is not None and self.ignore_index in ids:
            raise ValueError("ignore_index must not overlap a class ID.")
        if any(
            len(color) != 3 or any(not 0.0 <= value <= 1.0 for value in color)
            for color in colors.values()
        ):
            raise ValueError("Colors must be RGB triples with values in [0.0, 1.0].")

        object.__setattr__(self, "names", MappingProxyType(names))
        object.__setattr__(self, "colors", MappingProxyType(colors))

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def foreground_ids(self) -> tuple[int, ...]:
        return tuple(
            class_id for class_id in self.names.keys() if class_id != self.background_id
        )

    def foreground_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return all semantic channels except the background channel."""
        if tensor.ndim < 1:
            raise ValueError("Expected a tensor with a channel dimension.")
        if tensor.ndim == 3:
            channel_dim = 0
        elif tensor.ndim >= 4:
            channel_dim = 1
        else:
            raise ValueError("Expected [C, ...] or [B, C, ...] tensor.")
        if tensor.shape[channel_dim] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} channels, got {tensor.shape[channel_dim]}"
            )
        indices = torch.tensor(self.foreground_ids, device=tensor.device)
        return tensor.index_select(channel_dim, indices)

    def label_map_to_one_hot(self, label_map: torch.Tensor) -> torch.Tensor:
        """Convert ``[H, W]`` or ``[B, H, W]`` class IDs to ``[..., C, H, W]``.

        Pixels with ``ignore_index`` are represented by an all-zero vector.
        The returned tensor has ``torch.long`` dtype, matching
        :func:`torch.nn.functional.one_hot`.
        """
        if label_map.ndim not in (2, 3):
            raise ValueError(
                f"Expected [H, W] or [B, H, W] label map, got {tuple(label_map.shape)}"
            )
        if label_map.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"Expected an integer label map, got {label_map.dtype}")

        valid = (
            label_map != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(label_map, dtype=torch.bool)
        )
        invalid = valid & ((label_map < 0) | (label_map >= self.num_classes))
        if invalid.any():
            raise ValueError("Label map contains an undefined class ID.")

        safe_labels = label_map.masked_fill(~valid, 0)
        one_hot = torch.nn.functional.one_hot(
            safe_labels.long(), num_classes=self.num_classes
        )
        one_hot = one_hot.masked_fill(~valid.unsqueeze(-1), 0)
        return one_hot.movedim(-1, -3)

    def label_map_to_foreground_one_hot(self, label_map: torch.Tensor) -> torch.Tensor:
        """Convert a label map to one-hot channels excluding the background."""
        return self.foreground_channels(self.label_map_to_one_hot(label_map))

    def one_hot_to_label_map(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert full ``[C, H, W]`` or ``[B, C, H, W]`` channels to IDs."""
        if one_hot.ndim not in (3, 4):
            raise ValueError(
                f"Expected [C, H, W] or [B, C, H, W], got {tuple(one_hot.shape)}"
            )
        if one_hot.shape[-3] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} channels, got {one_hot.shape[-3]}"
            )
        return one_hot.argmax(dim=-3).long()

    def foreground_one_hot_to_label_map(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert foreground-only channels to IDs, including background."""
        if one_hot.ndim not in (3, 4):
            raise ValueError(
                f"Expected [C, H, W] or [B, C, H, W], got {tuple(one_hot.shape)}"
            )
        if one_hot.shape[-3] != len(self.foreground_ids):
            raise ValueError(
                f"Expected {len(self.foreground_ids)} foreground channels, "
                f"got {one_hot.shape[-3]}"
            )

        foreground_index = one_hot.argmax(dim=-3)
        foreground_id_tensor = torch.tensor(
            self.foreground_ids, device=one_hot.device, dtype=torch.long
        )
        labels = foreground_id_tensor[foreground_index]
        has_foreground = one_hot.amax(dim=-3) > 0
        return torch.where(
            has_foreground,
            labels,
            torch.full_like(labels, self.background_id),
        )

    @staticmethod
    def as_artery():
        return LabelSchema.from_lists(
            names=ArtificialMaskLabel, colors=ArtificialMaskColor
        )


@dataclass(frozen=True)
class DataSpec:
    """Immutable experiment-level data metadata."""

    label_schema: LabelSchema
