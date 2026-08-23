"""Adapters from generic metric/overlay results to Lightning and W&B."""

from collections.abc import Callable
from typing import Any

import wandb

from ..types import (
    LoggingProvider,
    LossResult,
    MetricResult,
    OverlayResult,
    StepContext,
)

ScalarLog = Callable[..., None]
WandbRunGetter = Callable[[], Any | None]
WandbImageFactory = Callable[..., Any]
IsGlobalZero = Callable[[], bool]


def _current_wandb_run() -> Any | None:
    return wandb.run


def _primary_process() -> bool:
    return True


class LightningWandbLoggingProvider(LoggingProvider):
    """Log scalar metrics through Lightning and generic overlays through W&B.

    The provider receives capabilities rather than a LightningModule. In
    particular, ``scalar_log`` is normally the module's bound ``self.log``.
    W&B lookup is deliberately lazy because its run is initialized after the
    module constructor has completed.
    """

    def __init__(
        self,
        scalar_log: ScalarLog,
        *,
        wandb_run_getter: WandbRunGetter = _current_wandb_run,
        is_global_zero: IsGlobalZero = _primary_process,
        wandb_image_factory: WandbImageFactory = wandb.Image,
    ) -> None:
        self._scalar_log = scalar_log
        self._wandb_run_getter = wandb_run_getter
        self._is_global_zero = is_global_zero
        self._wandb_image_factory = wandb_image_factory

    def log_batch(self, context: StepContext, result: MetricResult) -> None:
        self._log_scalars(context, "batch", result, on_step=True, on_epoch=False)

    def log_loss(
        self, context: StepContext, result: LossResult, *, prog_bar: bool
    ) -> None:
        # Batch-level training loss is useful for the progress bar and spotting
        # unstable optimization. Validation loss is reported only per epoch.
        on_step = context.stage == "train"
        self._scalar_log(
            f"{context.stage}/loss",
            result.total,
            on_step=on_step,
            on_epoch=True,
            prog_bar=prog_bar,
        )
        for name, value in (result.components or {}).items():
            self._scalar_log(
                f"{context.stage}/loss/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
            )
        for name, value in (result.unweighted_components or {}).items():
            self._scalar_log(
                f"{context.stage}/loss/{name}/unweighted",
                value,
                on_step=on_step,
                on_epoch=True,
            )
        for name, value in (result.logs or {}).items():
            self._scalar_log(
                f"{context.stage}/info/{name}",
                value,
                on_step=on_step,
                on_epoch=True,
            )

    def log_epoch(self, context: StepContext, result: MetricResult) -> None:
        self._log_scalars(context, "epoch", result, on_step=False, on_epoch=True)

    def log_overlay(
        self, context: StepContext, name: str, result: OverlayResult
    ) -> None:
        if not self._is_global_zero():
            return
        run = self._wandb_run_getter()
        if run is None:
            return

        key = self._key(context, "artifacts", name)
        run.define_metric(key, step_metric=f"{context.stage}/epoch")
        run.log(
            {
                key: [self._to_wandb_image(result)],
                f"{context.stage}/epoch": context.current_epoch,
            }
        )

    def _log_scalars(
        self,
        context: StepContext,
        scope: str,
        result: MetricResult,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in result.scalars.items():
            self._scalar_log(
                self._key(context, scope, name),
                value,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    @staticmethod
    def _key(context: StepContext, scope: str, name: str) -> str:
        normalized_name = name.strip("/")
        if not normalized_name:
            raise ValueError("Metric and overlay names must not be empty")
        return f"{context.stage}/{scope}/{normalized_name}"

    def _to_wandb_image(self, result: OverlayResult) -> Any:
        masks = {
            name: {
                "mask_data": mask.detach().cpu().numpy().astype("int32"),
                "class_labels": result.class_labels or {},
            }
            for name, mask in result.masks.items()
        }
        # Overlay computers own image preparation (channel layout, scale and
        # clamping). The provider only transfers the generic payload to W&B.
        image = result.image.detach().cpu().numpy()
        return self._wandb_image_factory(image, masks=masks, caption=result.caption)
