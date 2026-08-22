from types import SimpleNamespace

import torch

from constraints.lightning_wrappers.callbacks import (
    SegmentationRegistrationEarlyStopping,
)


def _trainer(metrics: dict[str, torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(
        callback_metrics=metrics,
        sanity_checking=False,
        should_stop=False,
    )


def test_early_stopping_waits_for_registration_iou_to_plateau():
    callback = SegmentationRegistrationEarlyStopping(
        patience=2,
        segmentation_min_delta=1e-3,
        registration_min_delta=1e-3,
    )
    trainer = _trainer(
        {
            "val/epoch/segmentation/iou/pred_vs_gt": torch.tensor(0.5),
            "val/epoch/registration/iou/warped_vs_gt": torch.tensor(0.5),
        }
    )

    callback.on_validation_end(trainer, None)
    assert not trainer.should_stop

    trainer.callback_metrics = {
        "val/epoch/segmentation/iou/pred_vs_gt": torch.tensor(0.5),
        "val/epoch/registration/iou/warped_vs_gt": torch.tensor(0.6),
    }
    callback.on_validation_end(trainer, None)
    assert callback.wait_count == 0

    callback.on_validation_end(trainer, None)
    assert callback.wait_count == 1
    callback.on_validation_end(trainer, None)
    assert trainer.should_stop


def test_early_stopping_uses_only_iou_without_registration_metrics():
    callback = SegmentationRegistrationEarlyStopping(
        patience=1,
        segmentation_min_delta=1e-3,
    )
    trainer = _trainer({"val/epoch/segmentation/iou/pred_vs_gt": torch.tensor(0.5)})

    callback.on_validation_end(trainer, None)
    assert not trainer.should_stop
    callback.on_validation_end(trainer, None)
    assert trainer.should_stop
