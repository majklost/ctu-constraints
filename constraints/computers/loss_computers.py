from abc import ABC, abstractmethod
import torch
from torch import nn

from ..types import LossInput, LossResult
from ..losses import OneSideSDFSquare


def _compute_grad_interaction_logs(
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    grad_refs: list[torch.Tensor | None],
    eps: float = 1e-12,
) -> dict[str, float | torch.Tensor] | None:
    """Compute gradient-balance diagnostics for two scalar losses.

    Gradients are taken w.r.t. provided model-output tensors so metrics are
    comparable across different model variants.
    """
    if not torch.is_grad_enabled():
        return None

    refs = [tensor for tensor in grad_refs if tensor is not None and tensor.requires_grad]
    if not refs:
        return None

    grads1 = torch.autograd.grad(
        loss1,
        refs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grads2 = torch.autograd.grad(
        loss2,
        refs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    def _flatten(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
        chunks = []
        for grad, ref in zip(grads, refs):
            if grad is None:
                chunks.append(torch.zeros_like(ref).reshape(-1))
            else:
                chunks.append(grad.reshape(-1))
        if not chunks:
            return torch.zeros(1, device=loss1.device, dtype=loss1.dtype)
        return torch.cat(chunks)

    grad_vec1 = _flatten(grads1)
    grad_vec2 = _flatten(grads2)

    grad_norm1 = torch.linalg.vector_norm(grad_vec1)
    grad_norm2 = torch.linalg.vector_norm(grad_vec2)
    grad_ratio = grad_norm1 / (grad_norm2 + eps)
    grad_cosine = torch.dot(grad_vec1, grad_vec2) / (grad_norm1 * grad_norm2 + eps)
    grad_share = grad_norm1 / (grad_norm1 + grad_norm2 + eps)

    return {
        "grad_norm/loss1": grad_norm1,
        "grad_norm/loss2": grad_norm2,
        "grad_ratio": grad_ratio,
        "grad_cosine": grad_cosine,
        "grad_share/loss1": grad_share,
    }


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
    def __init__(self, num_classes=3, seg_loss_weight=20.0, sdf_loss_weight=1.0):
        super().__init__()
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight
        self._one_sided = OneSideSDFSquare()
        self._cross_entropy = torch.nn.CrossEntropyLoss()

    def compute(self, loss_input: LossInput) -> LossResult:
        gt_sdf = loss_input.gt_mask_sdf
        gt_mask = loss_input.gt_mask
        pred_mask_logits = loss_input.segmentation_logits
        warped_template = loss_input.warped_template

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"

        loss_seg = self.seg_loss_weight * self._cross_entropy(pred_mask_logits, gt_mask)
        loss_sdf = self.sdf_loss_weight * self._one_sided(warped_template, gt_sdf)
        loss = loss_seg + loss_sdf

        components = {
            "loss_seg": loss_seg,
            "loss_sdf": loss_sdf,
        }

        logs = _compute_grad_interaction_logs(
            loss1=loss_seg,
            loss2=loss_sdf,
            grad_refs=[pred_mask_logits],
        )

        return LossResult(total=loss, components=components, logs=logs)


class CrossEntrOnly(ProjectLossComputer):
    """
    both losses (warped template vs binary mask) and (segmentation logits vs binary mask) are computed using cross entropy loss
    """

    def __init__(self, num_classes=3, seg_loss_weight=1.0, template_loss_weight=1.0):
        super().__init__()
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.template_loss_weight = template_loss_weight
        self._cross_entropy = torch.nn.CrossEntropyLoss()

    def compute(self, loss_input: LossInput) -> LossResult:
        gt_mask = loss_input.gt_mask
        pred_mask_logits = loss_input.segmentation_logits
        warped_template = loss_input.warped_template

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"

        # Convert probabilities to log-space logits while staying numerically stable.
        warped_template_logits = torch.log(warped_template.clamp_min(1e-8))

        loss_seg = self.seg_loss_weight * self._cross_entropy(pred_mask_logits, gt_mask)
        loss_template = self.template_loss_weight * self._cross_entropy(warped_template_logits, gt_mask)
        loss = loss_seg + loss_template

        components = {
            "loss_seg": loss_seg,
            "loss_template": loss_template,
        }

        logs = _compute_grad_interaction_logs(
            loss1=loss_seg,
            loss2=loss_template,
            grad_refs=[pred_mask_logits],
        )

        return LossResult(total=loss, components=components, logs=logs)


class OneSideOnly(ProjectLossComputer):
    """
    both losses (warped template vs binary mask in SDF representation) and (segmentation logits vs binary mask in SDF representation) are computed using one-sided sdf loss
    """

    def __init__(self, num_classes=3, seg_loss_weight=1.0, sdf_loss_weight=1.0):
        super().__init__()
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight
        self._one_sided = OneSideSDFSquare()

    def compute(self, loss_input: LossInput) -> LossResult:
        gt_sdf = loss_input.gt_mask_sdf
        pred_mask_logits = loss_input.segmentation_logits
        warped_template = loss_input.warped_template

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_sdf is not None, "gt_mask_sdf is required for loss computation"

        pred_mask_probs = torch.softmax(pred_mask_logits, dim=1)

        loss_seg = self.seg_loss_weight * self._one_sided(pred_mask_probs, gt_sdf)
        loss_sdf = self.sdf_loss_weight * self._one_sided(warped_template, gt_sdf)
        loss = loss_seg + loss_sdf

        components = {
            "loss_seg": loss_seg,
            "loss_sdf": loss_sdf,
        }

        logs = _compute_grad_interaction_logs(
            loss1=loss_seg,
            loss2=loss_sdf,
            grad_refs=[pred_mask_logits],
        )

        return LossResult(total=loss, components=components, logs=logs)
