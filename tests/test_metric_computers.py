import torch
import torch.nn.functional as functional

from constraints.datatools.label_schema import LabelSchema
from constraints.factories.metrics import (
    create_default_staged_metrics,
    create_segmentation_staged_metrics,
)
from constraints.types import (
    DiscreteSegmentation,
    FieldParams,
    MetricInput,
    StepContext,
    TransformSpec,
)

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen", "plaque"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.35, 0.95)],
)


def _valid_vessel_labels(size: int = 32) -> torch.Tensor:
    labels = torch.zeros((size, size), dtype=torch.long)
    labels[6:26, 6:26] = 1
    labels[9:23, 9:23] = 2
    return labels


def _input(predicted: torch.Tensor, target: torch.Tensor) -> MetricInput:
    return MetricInput(
        image=torch.zeros((predicted.shape[0], 1, *predicted.shape[1:])),
        segmentation_logits=functional.one_hot(
            predicted, LABEL_SCHEMA.num_classes
        ).movedim(-1, 1).float(),
        warped_template=functional.one_hot(
            target, LABEL_SCHEMA.num_classes
        ).movedim(-1, 1).float(),
        gt=DiscreteSegmentation(target, LABEL_SCHEMA),
    )


def _context(stage: str) -> StepContext:
    return StepContext(stage=stage, batch_idx=0, current_epoch=0, global_step=0)


def test_default_staged_metrics_use_independent_validation_state() -> None:
    valid = _valid_vessel_labels()
    invalid = torch.zeros_like(valid)
    predicted = torch.stack((valid, invalid))
    target = torch.stack((valid, valid))
    metrics = create_default_staged_metrics(LABEL_SCHEMA)

    metric_input = _input(predicted, target)
    metrics.update(_context("val"), metric_input)
    metrics.update(_context("val_extra"), metric_input)

    for stage in ("val", "val_extra"):
        result = metrics.compute(_context(stage)).scalars
        assert "segmentation/iou/pred_vs_gt" in result
        assert "registration/iou/warped_vs_gt" in result
        assert torch.isclose(
            result["segmentation/constraint/violation_rate"], torch.tensor(0.5)
        )
        assert torch.isclose(
            result["registration/constraint/violation_rate"], torch.tensor(0.0)
        )
        metrics.reset(_context(stage))
        assert metrics.compute(_context(stage)).scalars == {}


def test_default_staged_metrics_emit_no_constraint_rate_for_train() -> None:
    valid = _valid_vessel_labels()
    metrics = create_default_staged_metrics(LABEL_SCHEMA)
    context = _context("train")

    metrics.update(context, _input(torch.stack((valid,)), torch.stack((valid,))))

    result = metrics.compute(context).scalars
    assert "segmentation/iou/pred_vs_gt" in result
    assert "registration/iou/warped_vs_gt" in result
    assert "segmentation/constraint/violation_rate" not in result


def test_default_staged_metrics_do_not_share_state_between_validation_stages() -> None:
    valid = _valid_vessel_labels()
    metrics = create_default_staged_metrics(LABEL_SCHEMA)
    metric_input = _input(torch.stack((valid,)), torch.stack((valid,)))

    metrics.update(_context("val"), metric_input)

    assert metrics.compute(_context("val")).scalars
    assert metrics.compute(_context("val_extra")).scalars == {}


def test_segmentation_staged_metrics_support_binary_non_vessel_schema() -> None:
    label_schema = LabelSchema.from_lists(
        ["background", "myocardium"],
        [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1)],
    )
    target = torch.zeros((1, 32, 32), dtype=torch.long)
    target[:, 4:28, 4:28] = 1
    target[:, 10:22, 10:22] = 0
    metric_input = MetricInput(
        image=torch.zeros((1, 1, 32, 32)),
        segmentation_logits=functional.one_hot(target, 2).movedim(-1, 1).float(),
        gt=DiscreteSegmentation(target, label_schema),
    )
    metrics = create_segmentation_staged_metrics(label_schema)

    metrics.update(_context("val"), metric_input)

    result = metrics.compute(_context("val")).scalars
    assert torch.isclose(
        result["segmentation/iou/pred_vs_gt"],
        torch.tensor(1.0),
    )
    assert torch.isclose(
        result["segmentation/constraint/violation_rate"],
        torch.tensor(0.0),
    )
    assert "registration/constraint/violation_rate" not in result


def test_deformation_jacobian_diagnostics_report_identity_and_folding() -> None:
    valid = _valid_vessel_labels()
    metrics = create_default_staged_metrics(LABEL_SCHEMA)
    context = _context("val")
    inputs = _input(torch.stack((valid, valid)), torch.stack((valid, valid)))
    # Identity for the first sample; the second maps y -> -y and folds with
    # determinant -1 everywhere.
    field = torch.zeros((2, 2, 8, 8))
    field[1, 0] = -2 * torch.arange(8).view(8, 1)
    inputs.transform_spec = TransformSpec(field=FieldParams(field))

    metrics.update(context, inputs)
    result = metrics.compute(context).scalars

    assert torch.isclose(
        torch.as_tensor(result["registration/jacobian/mean_nonpositive_pixel_fraction"]),
        torch.tensor(0.5),
    )
    assert torch.isclose(
        torch.as_tensor(result["registration/jacobian/samples_with_nonpositive_fraction"]),
        torch.tensor(0.5),
    )
    assert torch.isclose(
        result["registration/jacobian/mean_sample_minimum"], torch.tensor(0.0)
    )
    assert torch.isclose(
        result["registration/jacobian/p01_sample_minimum"], torch.tensor(-0.98)
    )
