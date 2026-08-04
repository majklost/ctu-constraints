from abc import ABC, abstractmethod

import torch
from torch import nn

from ..datatools.datasets import ARTIFICIAL_MASK_NUM_CLASSES
from ..types import LossInput, LossResult
from .loss_terms import (
    LossTerm,
    RegistrationBlurredMSETerm,
    RegistrationCentroidTerm,
    RegistrationCrossEntropyTerm,
    RegistrationDSDFMSETerm,
    RegistrationMSE_SDFTEMPLATETerm,
    RegistrationOneside_SDFTEMPLATETerm,
    RegistrationOneSideSDFSquareTerm,
    RegistrationOneSideSDFTerm,
    SegmentationCrossEntropyTerm,
    SegmentationOneSideSDFSquareTerm,
    SegmentationOneSideSDFTerm,
)


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

    refs = [
        tensor for tensor in grad_refs if tensor is not None and tensor.requires_grad
    ]
    if not refs:
        return None

    def _grads_or_unused(loss: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        if not loss.requires_grad:
            return tuple(None for _ in refs)
        return torch.autograd.grad(
            loss,
            refs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    grads1 = _grads_or_unused(loss1)
    grads2 = _grads_or_unused(loss2)

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
        "coupling/segmentation_grad_norm": grad_norm1,
        "coupling/registration_grad_norm": grad_norm2,
        "coupling/grad_ratio_segmentation_to_registration": grad_ratio,
        "coupling/grad_cosine": grad_cosine,
        "coupling/segmentation_grad_share": grad_share,
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


class CompositeLossComputer(ProjectLossComputer):
    """Compose weighted loss terms while preserving the ProjectLossComputer API."""

    def __init__(
        self,
        terms: list[tuple[float, LossTerm]],
        *,
        grad_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompositeLossComputer requires at least one loss term")

        self.weights = [float(weight) for weight, _ in terms]
        self.terms = nn.ModuleList(term for _, term in terms)
        self.grad_diagnostics = grad_diagnostics

    def compute(self, loss_input: LossInput) -> LossResult:
        components: dict[str, torch.Tensor] = {}
        weighted_losses: list[torch.Tensor] = []
        logs: dict[str, float | torch.Tensor] = {}

        for weight, term_module in zip(self.weights, self.terms, strict=True):
            assert isinstance(term_module, LossTerm)
            if term_module.name in components:
                raise ValueError(f"Duplicate loss component name: {term_module.name}")
            weighted_loss = weight * term_module(loss_input)
            components[term_module.name] = weighted_loss
            weighted_losses.append(weighted_loss)

            if self.grad_diagnostics:
                term_logs = term_module.logs(loss_input, weighted_loss)
                if term_logs:
                    logs.update(term_logs)

        total = weighted_losses[0]
        for weighted_loss in weighted_losses[1:]:
            total = total + weighted_loss

        if self.grad_diagnostics and len(weighted_losses) >= 2:
            interaction_logs = _compute_grad_interaction_logs(
                loss1=weighted_losses[0],
                loss2=weighted_losses[1],
                grad_refs=[loss_input.segmentation_logits],
            )
            if interaction_logs:
                logs.update(interaction_logs)

        return LossResult(total=total, components=components, logs=logs or None)


# BCE0segmentation metrics


class SBCE_ROneSideSDFSquared(CompositeLossComputer):
    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=20.0,
        sdf_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationCrossEntropyTerm()),
                (sdf_loss_weight, RegistrationOneSideSDFSquareTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight


class SBCE_ROneSideSDF(CompositeLossComputer):
    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=20.0,
        sdf_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationCrossEntropyTerm()),
                (sdf_loss_weight, RegistrationOneSideSDFTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight


class SBCE_RBCE(CompositeLossComputer):
    """
    both losses (warped template vs binary mask) and (segmentation logits vs binary mask) are computed using cross entropy loss
    """

    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=1.0,
        template_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationCrossEntropyTerm()),
                (template_loss_weight, RegistrationCrossEntropyTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.template_loss_weight = template_loss_weight


class SBCE_RCentroid(CompositeLossComputer):
    """
    Compute the centroid of the warped template and compare it to the centroid of the ground truth mask.
    """

    def __init__(self, centroid_loss_weight=1.0, grad_diagnostics=False):
        super().__init__(
            terms=[
                (1.0, SegmentationCrossEntropyTerm()),
                (centroid_loss_weight, RegistrationCentroidTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self._seg_loss_weight = 1.0
        self.centroid_loss_weight = centroid_loss_weight


class SBCE_RDSDF_MSE(CompositeLossComputer):
    """
    differeniable sign distance function computer
    """

    def __init__(
        self, reg_loss_weight=1.0, sdf_clip: float | None = None, grad_diagnostics=False
    ):
        super().__init__(
            terms=[
                (1.0, SegmentationCrossEntropyTerm()),
                (reg_loss_weight, RegistrationDSDFMSETerm(sdf_clip=sdf_clip)),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self.seg_loss_weight = 1.0
        self.reg_loss_weight = reg_loss_weight
        self.sdf_clip = sdf_clip


class SBCE_RBlurredMSE(CompositeLossComputer):
    def __init__(self, blur_sigma=1.0, reduction="mean", grad_diagnostics=False):
        super().__init__(
            terms=[
                (1.0, SegmentationCrossEntropyTerm()),
                (
                    1.0,
                    RegistrationBlurredMSETerm(
                        blur_sigma=blur_sigma, reduction=reduction
                    ),
                ),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self.blur_sigma = blur_sigma
        self.reduction = reduction
        self.seg_loss_weight = 1.0
        self.reg_loss_weight = 1.0


class SBCE_RMSE_SDFTEMPLATE(CompositeLossComputer):
    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=1.0,
        template_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationCrossEntropyTerm()),
                (template_loss_weight, RegistrationMSE_SDFTEMPLATETerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.template_loss_weight = template_loss_weight


class SBCE_ROneSideSDF_SDFTEMPLATE(CompositeLossComputer):
    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=1.0,
        template_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationCrossEntropyTerm()),
                (template_loss_weight, RegistrationOneside_SDFTEMPLATETerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.template_loss_weight = template_loss_weight


# Misc


class SOneSideSDFSquared_ROneSideSDFSquared(CompositeLossComputer):
    """
    both losses (warped template vs binary mask in SDF representation) and (segmentation logits vs binary mask in SDF representation) are computed using one-sided sdf loss
    """

    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=1.0,
        sdf_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationOneSideSDFSquareTerm()),
                (sdf_loss_weight, RegistrationOneSideSDFSquareTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        """
        both losses (warped template vs binary mask in SDF representation) and (segmentation logits vs binary mask in SDF representation) are computed using one-sided sdf loss
        """
        # Kept for constructor backward compatibility; IoU now lives in metric computers.
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight


class SOneSideSDFPlain_ROneSideSDFPlain(CompositeLossComputer):
    """
    both losses (warped template vs binary mask in SDF representation) and (segmentation logits vs binary mask in SDF representation) are computed using one-sided sdf loss
    """

    def __init__(
        self,
        num_classes=ARTIFICIAL_MASK_NUM_CLASSES,
        seg_loss_weight=1.0,
        sdf_loss_weight=1.0,
        grad_diagnostics=False,
    ):
        super().__init__(
            terms=[
                (seg_loss_weight, SegmentationOneSideSDFTerm()),
                (sdf_loss_weight, RegistrationOneSideSDFTerm()),
            ],
            grad_diagnostics=grad_diagnostics,
        )
        self.num_classes = num_classes
        self.seg_loss_weight = seg_loss_weight
        self.sdf_loss_weight = sdf_loss_weight
