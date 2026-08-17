from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..datatools.datasets.types import Batch
from ..datatools.label_schema import LabelSchema


@dataclass
class GtStrategyResult:
    gt: torch.Tensor | None
    detach_seg: bool


@runtime_checkable
class GtStrategy(Protocol):
    def decide(self, batch: Batch, stage: str, epoch: int) -> GtStrategyResult: ...


class NoGt:
    def __init__(self, detach_seg: bool = False) -> None:
        self.detach_seg = detach_seg

    def decide(self, batch: Batch, stage: str, epoch: int) -> GtStrategyResult:
        return GtStrategyResult(gt=None, detach_seg=self.detach_seg)


class AlwaysGt:
    def __init__(self, label_schema: LabelSchema) -> None:
        self.label_schema = label_schema

    def decide(self, batch: Batch, stage: str, epoch: int) -> GtStrategyResult:
        return GtStrategyResult(
            gt=self.label_schema.label_map_to_one_hot(batch["target_labels"]),
            detach_seg=False,
        )


class WarmupGt:
    """GT for the first `n_epochs` of training, then none."""

    def __init__(
        self, n_epochs: int, label_schema: LabelSchema, detach_seg: bool = False
    ) -> None:
        self.n_epochs = n_epochs
        self.detach_seg = detach_seg
        self.label_schema = label_schema

    def decide(self, batch: Batch, stage: str, epoch: int) -> GtStrategyResult:
        if stage == "train" and epoch < self.n_epochs:
            return GtStrategyResult(
                gt=self.label_schema.label_map_to_one_hot(batch["target_labels"]),
                detach_seg=False,
            )
        return GtStrategyResult(gt=None, detach_seg=self.detach_seg)
