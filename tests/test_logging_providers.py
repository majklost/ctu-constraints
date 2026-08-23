from dataclasses import dataclass, field

import torch

from constraints.logging.logging_providers import LightningWandbLoggingProvider
from constraints.types import LossResult, MetricResult, OverlayResult, StepContext


def _context(stage: str = "val_extra") -> StepContext:
    return StepContext(stage=stage, batch_idx=3, current_epoch=7, global_step=11)


def test_provider_names_batch_and_epoch_scalars_from_the_shared_context() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    provider = LightningWandbLoggingProvider(
        scalar_log=lambda *args, **kwargs: calls.append((args, kwargs))
    )

    provider.log_batch(_context(), MetricResult(scalars={"segmentation/iou": 0.5}))
    provider.log_epoch(_context(), MetricResult(scalars={"segmentation/iou": 0.7}))

    assert calls == [
        (("val_extra/batch/segmentation/iou", 0.5), {"on_step": True, "on_epoch": False}),
        (("val_extra/epoch/segmentation/iou", 0.7), {"on_step": False, "on_epoch": True}),
    ]


def test_provider_logs_training_loss_per_step_and_validation_loss_per_epoch() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    provider = LightningWandbLoggingProvider(
        scalar_log=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    result = LossResult(
        total=torch.tensor(1.0),
        components={"segmentation/cross_entropy": torch.tensor(0.8)},
        unweighted_components={"segmentation/cross_entropy": torch.tensor(0.4)},
        logs={"coupling/grad_norm": torch.tensor(0.2)},
    )

    provider.log_loss(_context("train"), result, prog_bar=True)
    provider.log_loss(_context("val"), result, prog_bar=False)

    assert calls[0][0][0] == "train/loss"
    assert calls[0][1] == {"on_step": True, "on_epoch": True, "prog_bar": True}
    assert calls[1][0][0] == "train/loss/segmentation/cross_entropy"
    assert calls[2][0][0] == "train/loss/segmentation/cross_entropy/unweighted"
    assert calls[3][0][0] == "train/info/coupling/grad_norm"
    assert calls[4][0][0] == "val/loss"
    assert calls[4][1] == {"on_step": False, "on_epoch": True, "prog_bar": False}


@dataclass
class _FakeRun:
    defined_metrics: list[tuple[str, str]] = field(default_factory=list)
    payloads: list[dict[str, object]] = field(default_factory=list)

    def define_metric(self, key: str, step_metric: str) -> None:
        self.defined_metrics.append((key, step_metric))

    def log(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)


def test_provider_logs_generic_overlay_without_rescaling_it() -> None:
    run = _FakeRun()
    image_calls: list[tuple[object, dict[str, object], str | None]] = []
    provider = LightningWandbLoggingProvider(
        scalar_log=lambda *args, **kwargs: None,
        wandb_run_getter=lambda: run,
        wandb_image_factory=lambda image, **kwargs: image_calls.append(
            (image, kwargs["masks"], kwargs["caption"])
        )
        or "wandb-image",
    )
    image = torch.tensor([[2.0, -1.0]])
    provider.log_overlay(
        _context(),
        "labels",
        OverlayResult(
            image=image,
            masks={"prediction": torch.tensor([[1, 2]])},
            class_labels={1: "lumen"},
            caption="sample-x",
        ),
    )

    assert run.defined_metrics == [("val_extra/artifacts/labels", "val_extra/epoch")]
    assert run.payloads == [
        {"val_extra/artifacts/labels": ["wandb-image"], "val_extra/epoch": 7}
    ]
    assert image_calls[0][0].tolist() == [[2.0, -1.0]]
    assert image_calls[0][1]["prediction"]["mask_data"].tolist() == [[1, 2]]


def test_provider_skips_overlays_without_a_run_or_on_nonprimary_process() -> None:
    provider = LightningWandbLoggingProvider(
        scalar_log=lambda *args, **kwargs: None,
        wandb_run_getter=lambda: None,
        is_global_zero=lambda: False,
    )

    provider.log_overlay(
        _context(), "labels", OverlayResult(image=torch.zeros((2, 2)), masks={})
    )
