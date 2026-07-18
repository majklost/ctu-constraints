from abc import ABC, abstractmethod
import torch
from torch import nn

from ..types import LossInput, LossResult
from torchmetrics.functional.classification import multiclass_jaccard_index    
from ..losses import OneSideSDFSquare


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




class CrossEntrAndOneSide(ProjectLossComputer):
    def __init__(self, num_classes=3, weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self._one_sided = OneSideSDFSquare()
        self._cross_entropy = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _to_labels(x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H, W], already-labeled -> long
        if x.ndim == 4:
            return x.argmax(dim=1)
        return x.long()

    def compute(self, loss_input: LossInput) -> LossResult:
        gt_sdf = loss_input.gt_mask_sdf
        gt_mask = loss_input.gt_mask
        pred_mask_logits = loss_input.segmentation_logits
        warped_template = loss_input.warped_template

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"

        loss_seg = self._cross_entropy(pred_mask_logits, gt_mask)
        loss_sdf = self._one_sided(warped_template, gt_sdf)
        loss = 20 *loss_seg + loss_sdf

        pred_labels = self._to_labels(pred_mask_logits)
        warped_labels = self._to_labels(warped_template)
        gt_labels = self._to_labels(gt_mask)

        iou_pred_vs_gt = multiclass_jaccard_index(
            preds=pred_labels,
            target=gt_labels,
            num_classes=self.num_classes,
            average="macro",
        )
        iou_warped_vs_gt = multiclass_jaccard_index(
            preds=warped_labels,
            target=gt_labels,
            num_classes=self.num_classes,
            average="macro",
        )

        components = {
            "loss_seg": loss_seg,
            "loss_sdf": loss_sdf,
        }
        logs = {
            "iou/pred_vs_gt": iou_pred_vs_gt,
            "iou/warped_vs_gt": iou_warped_vs_gt,
        }

        return LossResult(total=loss, components=components, logs=logs)
