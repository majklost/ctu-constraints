import torch
import torch.nn.functional as functional

from constraints.computers.metric_computers import DefaultSegmentationMetricComputer
from constraints.datatools.datasets import ARTIFICIAL_MASK_NUM_CLASSES
from constraints.types import MetricInput


def _valid_vessel_labels(size: int = 32) -> torch.Tensor:
    labels = torch.zeros((size, size), dtype=torch.long)
    labels[6:26, 6:26] = 1
    labels[9:23, 9:23] = 2
    return labels


def _one_hot_masks(labels: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(labels, ARTIFICIAL_MASK_NUM_CLASSES).permute(0, 3, 1, 2)


def test_default_metrics_log_per_class_iou_and_validation_violation_counts():
    valid_labels = _valid_vessel_labels()
    invalid_labels = torch.zeros_like(valid_labels)
    predicted_labels = torch.stack([valid_labels, invalid_labels])
    target_labels = torch.stack([valid_labels, valid_labels])
    logits = _one_hot_masks(predicted_labels).float()
    target_masks = _one_hot_masks(target_labels).float()

    result = DefaultSegmentationMetricComputer().compute(
        MetricInput(
            stage="val",
            batch_idx=0,
            current_epoch=0,
            segmentation_logits=logits,
            warped_template=target_masks,
            gt_mask=target_masks,
        )
    )

    assert result.logs is not None
    assert "segmentation/iou/pred_vs_gt" in result.logs
    assert "segmentation/iou/background_vs_gt" in result.logs
    assert "segmentation/iou/boundary_vs_gt" in result.logs
    assert "segmentation/iou/lumen_vs_gt" in result.logs
    assert "segmentation/iou/plaque_vs_gt" in result.logs
    assert torch.isclose(
        result.logs["segmentation/constraint/violation_rate"], torch.tensor(0.5)
    )
    assert torch.isclose(
        result.logs["registration/constraint/violation_rate"], torch.tensor(0.0)
    )

    assert result.sum_logs is not None
    assert result.sum_logs["segmentation/constraint/violating_samples"].item() == 1
    assert result.sum_logs["segmentation/constraint/total_samples"].item() == 2
    assert result.sum_logs["registration/constraint/violating_samples"].item() == 0
    assert result.sum_logs["registration/constraint/total_samples"].item() == 2
    assert "registration/iou/warped_vs_gt" in result.logs
    assert "registration/iou/background_vs_gt" in result.logs
    assert "registration/iou/boundary_vs_gt" in result.logs
    assert "registration/iou/lumen_vs_gt" in result.logs
    assert "registration/iou/plaque_vs_gt" in result.logs
