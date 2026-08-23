import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from constraints.computers.loss_computers import CompositeLossComputer
from constraints.computers.loss_terms import (
    DeformationGradientTerm,
    RegistrationBlurredMSETerm,
    RegistrationCentroidTerm,
    RegistrationCrossEntropyTerm,
    RegistrationOneSideSDFSquareTerm,
    RegistrationOneSideSDFTerm,
    SegmentationCrossEntropyTerm,
    SegmentationOneSideSDFSquareTerm,
    SegmentationOneSideSDFTerm,
)
from constraints.datatools.label_schema import LabelSchema
from constraints.types import (
    DiscreteSegmentation,
    FieldParams,
    LossInput,
    TransformSpec,
    WeightedLossTerm,
)

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1)],
)


class _CaptureFirstArgument(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value: torch.Tensor | None = None

    def forward(self, value: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        self.value = value
        return value.sum() * 0


def _gt(labels: torch.Tensor) -> DiscreteSegmentation:
    return DiscreteSegmentation(labels=labels, label_schema=LABEL_SCHEMA)


def test_segmentation_cross_entropy_receives_raw_logits() -> None:
    logits = torch.tensor(
        [[[[2.0, -1.0]], [[-3.0, 4.0]], [[0.5, 0.0]]]], requires_grad=True
    )
    labels = torch.tensor([[[0, 1]]])

    actual = SegmentationCrossEntropyTerm(LABEL_SCHEMA)(
        LossInput(segmentation_logits=logits, gt=_gt(labels))
    )

    assert torch.allclose(actual, functional.cross_entropy(logits, labels))
    assert not torch.allclose(
        actual, functional.cross_entropy(logits.softmax(dim=1), labels)
    )


def test_deformation_gradient_uses_voxelmorph_field_and_preserves_gradients() -> None:
    field = torch.zeros((1, 2, 4, 5), requires_grad=True)
    with torch.no_grad():
        field[:, 0] = torch.arange(4).view(1, 4, 1)

    loss = DeformationGradientTerm(LABEL_SCHEMA)(
        LossInput(transform_spec=TransformSpec(field=FieldParams(field)))
    )

    assert loss.ndim == 0
    assert loss.item() > 0
    loss.backward()
    assert field.grad is not None
    assert torch.count_nonzero(field.grad) > 0


def test_composite_logs_unweighted_and_weighted_deformation_regularization() -> None:
    field = torch.zeros((1, 2, 4, 5), requires_grad=True)
    with torch.no_grad():
        field[:, 0] = torch.arange(4).view(1, 4, 1)
    logits = torch.randn((1, LABEL_SCHEMA.num_classes, 4, 5), requires_grad=True)
    labels = torch.zeros((1, 4, 5), dtype=torch.long)
    computer = CompositeLossComputer(
        terms=[
            WeightedLossTerm(0.0, SegmentationCrossEntropyTerm(LABEL_SCHEMA)),
            WeightedLossTerm(0.25, DeformationGradientTerm(LABEL_SCHEMA)),
        ],
    )

    result = computer.compute(
        LossInput(
            segmentation_logits=logits,
            gt=_gt(labels),
            transform_spec=TransformSpec(field=FieldParams(field)),
        )
    )

    assert result.components is not None
    weighted = result.components["registration/deformation_gradient"]
    assert torch.allclose(result.total, weighted)


@pytest.mark.parametrize(
    "term_type,attribute",
    [
        (SegmentationOneSideSDFTerm, "_one_sided"),
        (SegmentationOneSideSDFSquareTerm, "_one_sided"),
    ],
)
def test_segmentation_soft_mask_losses_apply_softmax(term_type, attribute) -> None:
    logits = torch.randn(2, LABEL_SCHEMA.num_classes, 3, 4)
    capture = _CaptureFirstArgument()
    term = term_type(LABEL_SCHEMA)
    setattr(term, attribute, capture)

    term(LossInput(segmentation_logits=logits, gt_mask_sdf=torch.zeros(2, 2, 3, 4)))

    expected = LABEL_SCHEMA.foreground_channels(logits.softmax(dim=1))
    assert capture.value is not None
    assert torch.allclose(capture.value, expected)


@pytest.mark.parametrize(
    "term_type,attribute",
    [
        (RegistrationOneSideSDFTerm, "_one_sided"),
        (RegistrationOneSideSDFSquareTerm, "_one_sided"),
    ],
)
def test_registration_sdf_losses_use_soft_template_without_softmax(
    term_type, attribute
) -> None:
    warped_template = torch.rand(2, LABEL_SCHEMA.num_classes, 3, 4)
    capture = _CaptureFirstArgument()
    term = term_type(LABEL_SCHEMA)
    setattr(term, attribute, capture)

    term(
        LossInput(
            warped_template=warped_template,
            gt_mask_sdf=torch.zeros(2, 2, 3, 4),
        )
    )

    expected = LABEL_SCHEMA.foreground_channels(warped_template)
    assert capture.value is not None
    assert torch.equal(capture.value, expected)


@pytest.mark.parametrize(
    "term_type,attribute",
    [
        (RegistrationCentroidTerm, "_centroid"),
        (RegistrationBlurredMSETerm, "_blurred_mse_loss"),
    ],
)
def test_registration_mask_losses_use_soft_template_without_softmax(
    term_type, attribute
) -> None:
    warped_template = torch.rand(2, LABEL_SCHEMA.num_classes, 3, 4)
    labels = torch.randint(0, LABEL_SCHEMA.num_classes, (2, 3, 4))
    capture = _CaptureFirstArgument()
    term = term_type(LABEL_SCHEMA)
    setattr(term, attribute, capture)

    term(LossInput(warped_template=warped_template, gt=_gt(labels)))

    assert capture.value is warped_template


def test_registration_cross_entropy_converts_soft_template_to_log_probs() -> None:
    warped_template = torch.tensor(
        [[[[0.7, 0.1]], [[0.2, 0.3]], [[0.1, 0.6]]]], requires_grad=True
    )
    labels = torch.tensor([[[0, 2]]])

    actual = RegistrationCrossEntropyTerm(LABEL_SCHEMA)(
        LossInput(warped_template=warped_template, gt=_gt(labels))
    )
    expected = functional.nll_loss(warped_template.clamp_min(1e-8).log(), labels)

    assert torch.allclose(actual, expected)
