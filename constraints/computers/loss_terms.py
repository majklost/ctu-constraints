from abc import ABC, abstractmethod

import torch
from torch import nn

from ..datatools.datasets import (
    ARTIFICIAL_MASK_NUM_CLASSES,
    ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS,
)
from ..losses_metrics import BlurredMSELoss, CentroidLoss, OneSideSDFSquare, OneSideSDF,RawMaskCrossEntropyLoss
from ..types import LossInput
from ..utils import signed_distance_kornia_differentiable


def _require_channels(tensor: torch.Tensor, expected_channels: int, name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[1] != expected_channels:
        raise ValueError(
            f"{name} must have {expected_channels} channels, got {tensor.shape[1]} "
            f"with shape {tuple(tensor.shape)}"
        )


def _compute_grad_norm_logs(
    loss: torch.Tensor,
    grad_refs: list[torch.Tensor | None],
    prefix: str,
) -> dict[str, float | torch.Tensor] | None:
    if not torch.is_grad_enabled() or not loss.requires_grad:
        return None

    refs = [
        tensor for tensor in grad_refs if tensor is not None and tensor.requires_grad
    ]
    if not refs:
        return None

    grads = torch.autograd.grad(
        loss,
        refs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    chunks = []
    for grad, ref in zip(grads, refs, strict=True):
        if grad is None:
            chunks.append(torch.zeros_like(ref).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))

    grad_vec = torch.cat(chunks)
    return {
        f"{prefix}/grad_norm": torch.linalg.vector_norm(grad_vec),
        f"{prefix}/grad_abs_mean": grad_vec.abs().mean(),
        f"{prefix}/grad_nonzero_share": (grad_vec != 0).float().mean(),
    }


class LossTerm(nn.Module, ABC):
    """Single weighted component used inside a composite loss computer."""

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        """Compute one unweighted scalar loss term."""

    def logs(
        self,
        loss_input: LossInput,
        weighted_loss: torch.Tensor,
    ) -> dict[str, float | torch.Tensor] | None:
        del loss_input, weighted_loss
        return None


class SegmentationCrossEntropyTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("segmentation/cross_entropy")
        self._cross_entropy = RawMaskCrossEntropyLoss()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        pred_mask_logits = loss_input.segmentation_logits
        gt_mask = loss_input.gt_mask

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(pred_mask_logits, ARTIFICIAL_MASK_NUM_CLASSES, "segmentation_logits")
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        return self._cross_entropy(pred_mask_logits, gt_mask)


class SegmentationOneSideSDFSquareTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("segmentation/one_side_sdf")
        self._one_sided = OneSideSDFSquare()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        pred_mask_logits = loss_input.segmentation_logits
        gt_sdf = loss_input.gt_mask_sdf

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert gt_sdf is not None, "gt_mask_sdf is required for loss computation"
        _require_channels(pred_mask_logits, ARTIFICIAL_MASK_NUM_CLASSES, "segmentation_logits")
        _require_channels(gt_sdf, ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, "gt_mask_sdf")

        pred_mask_probs = torch.softmax(pred_mask_logits, dim=1)
        pred_mask_probs = pred_mask_probs[:, 1:]
        return self._one_sided(pred_mask_probs, gt_sdf)


class SegmentationOneSideSDFTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("segmentation/one_side_sdf_plain")
        self._one_sided = OneSideSDF()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        pred_mask_logits = loss_input.segmentation_logits
        gt_sdf = loss_input.gt_mask_sdf

        assert pred_mask_logits is not None, "segmentation_logits is required for loss computation"
        assert gt_sdf is not None, "gt_mask_sdf is required for loss computation"
        _require_channels(pred_mask_logits, ARTIFICIAL_MASK_NUM_CLASSES, "segmentation_logits")
        _require_channels(gt_sdf, ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, "gt_mask_sdf")

        pred_mask_probs = torch.softmax(pred_mask_logits, dim=1)
        pred_mask_probs = pred_mask_probs[:, 1:]
        return self._one_sided(pred_mask_probs, gt_sdf)
    

class RegistrationOneSideSDFTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("registration/one_side_sdf_plain")
        self._one_sided = OneSideSDF()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_sdf = loss_input.gt_mask_sdf

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_sdf is not None, "gt_mask_sdf is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_sdf, ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, "gt_mask_sdf")

        warped_template = warped_template[:, 1:]
        return self._one_sided(warped_template, gt_sdf)

class RegistrationOneSideSDFSquareTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("registration/one_side_sdf")
        self._one_sided = OneSideSDFSquare()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_sdf = loss_input.gt_mask_sdf

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_sdf is not None, "gt_mask_sdf is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_sdf, ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, "gt_mask_sdf")

        warped_template = warped_template[:, 1:]
        return self._one_sided(warped_template, gt_sdf)




class RegistrationCrossEntropyTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("registration/cross_entropy")
        self._cross_entropy = RawMaskCrossEntropyLoss()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_mask = loss_input.gt_mask

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        warped_template_logits = torch.log(warped_template.clamp_min(1e-8))
        return self._cross_entropy(warped_template_logits, gt_mask)


class RegistrationCentroidTerm(LossTerm):
    def __init__(self) -> None:
        super().__init__("registration/centroid")
        self._centroid = CentroidLoss()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_mask = loss_input.gt_mask

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        return self._centroid(warped_template, gt_mask)


class RegistrationDSDFMSETerm(LossTerm):
    def __init__(self, sdf_clip: float | None = None) -> None:
        super().__init__("registration/dsdf_mse")
        if sdf_clip is not None and sdf_clip <= 0:
            raise ValueError(f"sdf_clip must be positive or None, got {sdf_clip}")
        self.sdf_clip = sdf_clip
        self._reg_loss = torch.nn.MSELoss()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_sdf = loss_input.gt_mask_sdf

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_sdf is not None, "gt_sdf is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_sdf, ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS, "gt_sdf")

        warped_template_sdf = signed_distance_kornia_differentiable(warped_template[:, 1:])
        if self.sdf_clip is not None:
            warped_template_sdf = warped_template_sdf.clamp(-self.sdf_clip, self.sdf_clip)
            gt_sdf = gt_sdf.clamp(-self.sdf_clip, self.sdf_clip)
        return self._reg_loss(warped_template_sdf, gt_sdf)

    def logs(
        self,
        loss_input: LossInput,
        weighted_loss: torch.Tensor,
    ) -> dict[str, float | torch.Tensor] | None:
        return _compute_grad_norm_logs(
            loss=weighted_loss,
            grad_refs=[loss_input.warped_template],
            prefix="registration/warped_template",
        )


class RegistrationBlurredMSETerm(LossTerm):
    def __init__(self, blur_sigma=1.0, reduction="mean") -> None:
        super().__init__("registration/blurred_mse")
        self.blur_sigma = blur_sigma
        self.reduction = reduction
        self._blurred_mse_loss = BlurredMSELoss(sigma=blur_sigma, reduction=reduction)

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template = loss_input.warped_template
        gt_mask = loss_input.gt_mask

        assert warped_template is not None, "warped_template is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(warped_template, ARTIFICIAL_MASK_NUM_CLASSES, "warped_template")
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        return self._blurred_mse_loss(warped_template, gt_mask)

class RegistrationMSE_SDFTEMPLATETerm(LossTerm):
    """Compare a warped foreground SDF template with the target foreground mask."""

    def __init__(self) -> None:
        super().__init__("registration/sdf_template_mse")
        self._mse = nn.MSELoss()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template_sdf = loss_input.warped_template_sdf
        gt_mask = loss_input.gt_mask

        assert warped_template_sdf is not None, "warped_template_sdf is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(
            warped_template_sdf,
            ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS,
            "warped_template_sdf",
        )
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        return self._mse(warped_template_sdf, gt_mask[:, 1:])

    def logs(
        self,
        loss_input: LossInput,
        weighted_loss: torch.Tensor,
    ) -> dict[str, float | torch.Tensor] | None:
        return _compute_grad_norm_logs(
            loss=weighted_loss,
            grad_refs=[loss_input.warped_template_sdf],
            prefix="registration/warped_template_sdf",
        )


class RegistrationOneside_SDFTEMPLATETerm(LossTerm):
    """Apply one-sided SDF loss to a warped foreground SDF template."""

    def __init__(self) -> None:
        super().__init__("registration/sdf_template_one_side_sdf")
        self._one_sided = OneSideSDFSquare()

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        warped_template_sdf = loss_input.warped_template_sdf
        gt_mask = loss_input.gt_mask

        assert warped_template_sdf is not None, "warped_template_sdf is required for loss computation"
        assert gt_mask is not None, "gt_mask is required for loss computation"
        _require_channels(
            warped_template_sdf,
            ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS,
            "warped_template_sdf",
        )
        _require_channels(gt_mask, ARTIFICIAL_MASK_NUM_CLASSES, "gt_mask")

        return self._one_sided(gt_mask[:, 1:], warped_template_sdf)

    def logs(
        self,
        loss_input: LossInput,
        weighted_loss: torch.Tensor,
    ) -> dict[str, float | torch.Tensor] | None:
        return _compute_grad_norm_logs(
            loss=weighted_loss,
            grad_refs=[loss_input.warped_template_sdf],
            prefix="registration/warped_template_sdf",
        )
