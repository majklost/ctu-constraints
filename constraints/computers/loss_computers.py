from abc import ABC, abstractmethod
import torch
from torch import nn

from ..types import LossInput, LossResult



class LossComputer(nn.Module, ABC):
    """Base class for configurable loss computation.

        Subclass this for ablations. Implement `compute()` and return `LossResult`.

        Usage convention:
        - `compute()` is the canonical API for train/val/test steps because it
            returns the scalar loss and optional components/logs in one call.
        - `forward()` is a convenience wrapper that returns only
            `compute(loss_input).total` for scalar-only use cases.

        This avoids double computation when both optimization and logging values are
        needed.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute(self, loss_input: LossInput) -> LossResult:
        """Compute one structured loss result.

        Returns:
            LossResult containing:
            - `total`: scalar tensor used for `backward()`.
            - `components`: optional named loss terms for diagnostics.
            - `logs`: optional values ready for logger integration.
        """

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        """Convenience scalar-loss interface.

        Prefer calling `compute()` in training loops if you also need logs or
        loss components. `forward()` is best when only the scalar objective is
        required.
        """
        result = self.compute(loss_input)
        if result.total.ndim != 0:
            raise ValueError(
                f"LossResult.total must be a scalar tensor, got shape {tuple(result.total.shape)}"
            )
        return result.total


class ProjectLossComputer(LossComputer):
    """
    Project-specific loss computer used by ProjectLightning.

    Concrete subclasses should implement `compute()` using the shared
    `LossInput` contract.
    """

    @abstractmethod
    def compute(self, loss_input: LossInput) -> LossResult:
        """Implement project-specific loss from the typed `LossInput` contract."""
