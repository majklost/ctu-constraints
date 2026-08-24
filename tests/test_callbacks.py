import json
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from constraints.lightning_wrappers.callbacks import (
    InferenceWeightsCheckpoint,
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


def test_inference_weights_checkpoint_round_trips_and_prefers_registration(
    tmp_path, monkeypatch
):
    import constraints.lightning_wrappers.callbacks as callbacks_module

    class _Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, inputs):
            return self.linear(inputs)

        def training_step(self, batch, batch_idx):
            inputs, targets = batch
            return torch.nn.functional.mse_loss(self(inputs), targets)

        def validation_step(self, batch, batch_idx):
            self.log("val/epoch/segmentation/iou/pred_vs_gt", 0.8)
            self.log("val/epoch/registration/iou/warped_vs_gt", 0.9)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    monkeypatch.setattr(
        callbacks_module, "get_weights_folder", lambda *args: tmp_path
    )
    callback = InferenceWeightsCheckpoint(
        experiment="ex4", filename="example", run_id="run-123"
    )
    model = _Model()
    data = TensorDataset(torch.ones(2, 2), torch.zeros(2, 1))
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )
    trainer.fit(model, DataLoader(data), DataLoader(data))

    reloaded = _Model()
    checkpoint = torch.load(tmp_path / "weights.ckpt", weights_only=True)
    reloaded.load_state_dict(checkpoint["state_dict"])
    fixed_input = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(reloaded(fixed_input), model(fixed_input))
    assert callback.monitor == "val/epoch/registration/iou/warped_vs_gt"
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    selection = metadata["checkpoint_selection"]
    assert selection["metric"] == "val/epoch/registration/iou/warped_vs_gt"
    assert metadata["weight_file_size_bytes"] > 0


def test_inference_weights_checkpoint_falls_back_to_segmentation():
    selected = InferenceWeightsCheckpoint._select_monitor(
        {"val/epoch/segmentation/iou/pred_vs_gt": torch.tensor(0.8)}
    )
    assert selected == "val/epoch/segmentation/iou/pred_vs_gt"
