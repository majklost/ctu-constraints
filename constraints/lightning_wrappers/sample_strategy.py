from typing import Protocol, runtime_checkable
from ..datatools.datasets import Sample
import torch
from dataclasses import dataclass


@dataclass
class GtStrategyResult:
    gt: torch.Tensor | None
    detach_seg: bool



@runtime_checkable
class GtStrategy(Protocol):
    def decide(self, batch: Sample, stage: str, epoch: int) -> GtStrategyResult: ...


class NoGt:
    def __init__(self,detach_seg: bool = False) -> None:
        self.detach_seg = detach_seg

    def decide(self, batch: Sample, stage: str, epoch: int) -> GtStrategyResult:
        return GtStrategyResult(gt=None, detach_seg=self.detach_seg)


class AlwaysGt:
    def decide(self, batch: Sample, stage: str, epoch: int) -> GtStrategyResult:
        return GtStrategyResult(gt=batch['mask'], detach_seg=False)


class WarmupGt:
    """GT for the first `n_epochs` of training, then none."""
    def __init__(self, n_epochs: int,detach_seg: bool = False) -> None:
        self.n_epochs = n_epochs
        self.detach_seg = detach_seg

    def decide(self, batch: Sample, stage: str, epoch: int) -> GtStrategyResult:
        if stage == "train" and epoch < self.n_epochs:
            return GtStrategyResult(gt=batch['mask'], detach_seg=False)
        return GtStrategyResult(gt=None, detach_seg=self.detach_seg)
