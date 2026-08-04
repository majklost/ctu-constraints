from unittest.mock import patch

import torch
import torch.nn.functional as functional
from torch import nn

from constraints.computers.loss_computers import (
    SBCE_RMSE_SDFTEMPLATE,
    SBCE_ROneSideSDF_SDFTEMPLATE,
)
from constraints.computers.loss_terms import (
    RegistrationDSDFMSETerm,
    RegistrationMSE_SDFTEMPLATETerm,
)
from constraints.lightning_wrappers.modules import ProjectLightning
from constraints.transforms.transformers import (
    SequentialTransformer,
    SpatialTransformer,
)
from constraints.types import (
    FieldParams,
    LossInput,
    RigidParams,
    TransformSpec,
    WarpResult,
)
from constraints.utils import signed_distance_kornia_differentiable


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


class _SequentialModel(nn.Module):
    def __init__(self, transform_spec: TransformSpec) -> None:
        super().__init__()
        self.transform_spec = transform_spec

    def forward(
        self,
        image: torch.Tensor,
        template: torch.Tensor,
        gt: torch.Tensor | None = None,
        detach_seg: bool = False,
    ) -> tuple[torch.Tensor, TransformSpec]:
        del template, gt, detach_seg
        return image, self.transform_spec


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

    predicted_logits, result = module.forward(
        logits, template, template_sdf=template_sdf
    )

    assert torch.equal(predicted_logits, logits)
    assert torch.equal(result.warped_template, template)
    assert result.warped_template_sdf is not None
    assert torch.equal(result.warped_template_sdf, template_sdf)


def test_project_lightning_replays_sequential_transforms_for_mask_and_sdf():
    template = torch.arange(12 * 12, dtype=torch.float32).reshape(1, 1, 12, 12)
    template_sdf = template * 0.5
    rigid_spec = TransformSpec(
        rigid=RigidParams(
            angle=torch.tensor([0.1]),
            dx=torch.tensor([0.05]),
            dy=torch.tensor([-0.03]),
        )
    )
    field_spec = TransformSpec(field=FieldParams(field=torch.full((1, 2, 12, 12), 0.1)))
    transform_spec = TransformSpec(steps=(rigid_spec, field_spec))
    transformer = SequentialTransformer()
    module = ProjectLightning(
        _SequentialModel(transform_spec), transformer, SBCE_RMSE_SDFTEMPLATE()
    )

    logits = torch.randn(1, 4, 12, 12)
    _, result = module.forward(logits, template, template_sdf=template_sdf)

    expected_template = transformer(template, transform_spec).warped_template
    expected_template_sdf = transformer(template_sdf, transform_spec).warped_template

    assert torch.allclose(result.warped_template, expected_template)
    assert result.warped_template_sdf is not None
    assert torch.allclose(result.warped_template_sdf, expected_template_sdf)


def test_sdf_template_losses_use_dedicated_sdf_warp():
    batch_size, height, width = 2, 12, 12
    target = _one_hot_target(batch_size, height, width)
    target_sdf = torch.randn(batch_size, 3, height, width)

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
                gt_mask_sdf=target_sdf,
            )
        ).total
        total.backward()

        assert torch.isfinite(total)
        assert logits.grad is not None and logits.grad.abs().sum() > 0
        assert warped_template_sdf.grad is not None
        assert warped_template_sdf.grad.abs().sum() > 0


def test_dsdf_mse_has_finite_gradients_for_binary_foreground_template():
    target = torch.zeros(1, 4, 32, 32)
    target[:, 0] = 1
    target[:, 0, 8:24, 8:24] = 0
    target[:, 1, 8:24, 8:24] = 1
    target_sdf = signed_distance_kornia_differentiable(target[:, 1:])
    warped_template = target.clone().requires_grad_()

    loss = RegistrationDSDFMSETerm()(
        LossInput(
            warped_template=warped_template,
            gt_mask=target,
            gt_mask_sdf=target_sdf,
        )
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss < 1e-6
    assert warped_template.grad is not None
    assert torch.isfinite(warped_template.grad).all()


def test_sdf_template_mse_compares_matching_signed_distances():
    warped_template_sdf = torch.tensor(
        [[[[-10.0, 10.0]], [[-10.0, 10.0]], [[-10.0, 10.0]]]]
    )

    loss = RegistrationMSE_SDFTEMPLATETerm()(
        LossInput(
            warped_template_sdf=warped_template_sdf,
            gt_mask_sdf=warped_template_sdf.clone(),
        )
    )

    assert loss < 1e-8


def test_gradient_diagnostics_are_opt_in():
    batch_size, height, width = 2, 12, 12
    target = _one_hot_target(batch_size, height, width)
    loss_input = LossInput(
        segmentation_logits=torch.randn(
            batch_size, 4, height, width, requires_grad=True
        ),
        warped_template=torch.rand(batch_size, 4, height, width, requires_grad=True),
        warped_template_sdf=torch.randn(
            batch_size, 3, height, width, requires_grad=True
        ),
        gt_mask=target,
        gt_mask_sdf=torch.randn(batch_size, 3, height, width),
    )

    with patch(
        "torch.autograd.grad",
        side_effect=AssertionError("unexpected gradient diagnostic"),
    ):
        result = SBCE_RMSE_SDFTEMPLATE().compute(loss_input)

    assert result.logs is None

    result = SBCE_RMSE_SDFTEMPLATE(grad_diagnostics=True).compute(loss_input)
    assert result.logs is not None
    assert "coupling/segmentation_grad_norm" in result.logs
    assert "registration/warped_template_sdf/grad_norm" in result.logs
