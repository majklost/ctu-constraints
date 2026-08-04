import torch
import torch.nn.functional as functional
from torch import nn

from constraints.computers.loss_computers import (
    SBCE_RMSE_SDFTEMPLATE,
    SBCE_ROneSideSDF_SDFTEMPLATE,
)
from constraints.lightning_wrappers.modules import ProjectLightning
from constraints.transforms.transformers import SpatialTransformer
from constraints.types import LossInput, TransformSpec, WarpResult


class _DummyModel(nn.Module):
    def forward(
        self,
        image: torch.Tensor,
        template: torch.Tensor,
        gt: torch.Tensor | None = None,
        detach_seg: bool = False,
    ) -> tuple[torch.Tensor, TransformSpec]:
        del template, gt, detach_seg
        return image, TransformSpec()


class _IdentityTransformer(SpatialTransformer):
    def forward(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        return WarpResult(warped_template=template, transform_spec=transform_spec)


def _one_hot_target(batch_size: int, height: int, width: int) -> torch.Tensor:
    labels = torch.randint(0, 4, (batch_size, height, width))
    return functional.one_hot(labels, num_classes=4).permute(0, 3, 1, 2).float()


def test_project_lightning_keeps_mask_and_sdf_template_warps_separate():
    batch_size, height, width = 2, 12, 12
    logits = torch.randn(batch_size, 4, height, width)
    template = torch.rand(batch_size, 4, height, width)
    template_sdf = torch.randn(batch_size, 3, height, width)
    module = ProjectLightning(
        _DummyModel(), _IdentityTransformer(), SBCE_RMSE_SDFTEMPLATE()
    )

    predicted_logits, result = module.forward(logits, template, template_sdf=template_sdf)

    assert torch.equal(predicted_logits, logits)
    assert torch.equal(result.warped_template, template)
    assert torch.equal(result.warped_template_sdf, template_sdf)


def test_sdf_template_losses_use_dedicated_sdf_warp():
    batch_size, height, width = 2, 12, 12
    target = _one_hot_target(batch_size, height, width)

    for loss_computer in (SBCE_RMSE_SDFTEMPLATE(), SBCE_ROneSideSDF_SDFTEMPLATE()):
        logits = torch.randn(batch_size, 4, height, width, requires_grad=True)
        warped_template = torch.rand(batch_size, 4, height, width, requires_grad=True)
        warped_template_sdf = torch.randn(
            batch_size, 3, height, width, requires_grad=True
        )

        total = loss_computer.compute(
            LossInput(
                segmentation_logits=logits,
                warped_template=warped_template,
                warped_template_sdf=warped_template_sdf,
                gt_mask=target,
            )
        ).total
        total.backward()

        assert torch.isfinite(total)
        assert logits.grad is not None and logits.grad.abs().sum() > 0
        assert warped_template_sdf.grad is not None
        assert warped_template_sdf.grad.abs().sum() > 0
