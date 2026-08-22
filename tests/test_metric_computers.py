import torch
import torch.nn.functional as functional

from constraints.datatools.label_schema import LabelSchema
from constraints.factories.metrics import create_default_staged_metrics
from constraints.types import DiscreteSegmentation, MetricInput, StepContext


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
