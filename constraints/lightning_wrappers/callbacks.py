"""
Various callbacks for Lightning Wrappers.
- debugging, inspection
- saving runs, etc...
"""

import math

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class GradientNormLogger(Callback):
    """Logs global gradient norm under `debug/grad_norm`.

    This callback is optional and intended for quick tuning/debug sessions.
    Throttling keeps overhead low in longer runs.
    """

    def __init__(self, every_n_steps: int = 50) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be > 0")
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer  # Callback computes across all module parameters.

        if trainer.global_step % self.every_n_steps != 0:
            return

        sq_norm_sum = 0.0
        has_grad = False
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().data.norm(2)
            sq_norm_sum += float(grad_norm.item()) ** 2
            has_grad = True

        if not has_grad:
            return

        total_norm = math.sqrt(sq_norm_sum)
        pl_module.log(
            "debug/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False
        )


class SegmentationRegistrationEarlyStopping(Callback):
    """Stop only after segmentation and registration IoU both plateau.

    Models without ``val/registration/iou/warped_vs_gt``, such as the plain
    U-Net baseline, are stopped solely based on segmentation IoU.
    """

    def __init__(
        self,
        patience: int,
        segmentation_min_delta: float,
        registration_min_delta: float = 1e-3,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be > 0")
        if segmentation_min_delta < 0:
            raise ValueError("segmentation_min_delta must be >= 0")
        if registration_min_delta < 0:
            raise ValueError("registration_min_delta must be >= 0")

        self.patience = patience
        self.segmentation_min_delta = segmentation_min_delta
        self.registration_min_delta = registration_min_delta
        self.best_segmentation_iou = float("-inf")
        self.best_registration_iou = float("-inf")
        self.wait_count = 0

    @staticmethod
    def _metric_value(metric: torch.Tensor | float) -> float:
        if isinstance(metric, torch.Tensor):
            return float(metric.detach().cpu().item())
        return float(metric)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        segmentation_metric = metrics.get("val/segmentation/iou/pred_vs_gt")
        if segmentation_metric is None:
            raise RuntimeError(
                "SegmentationRegistrationEarlyStopping requires "
                "'val/segmentation/iou/pred_vs_gt'."
            )

        segmentation_iou = self._metric_value(segmentation_metric)
        if not math.isfinite(segmentation_iou):
            trainer.should_stop = True
            return

        segmentation_improved = (
            segmentation_iou > self.best_segmentation_iou + self.segmentation_min_delta
        )
        if segmentation_improved:
            self.best_segmentation_iou = segmentation_iou

        registration_improved = False
        registration_metric = metrics.get("val/registration/iou/warped_vs_gt")
        if registration_metric is not None:
            registration_iou = self._metric_value(registration_metric)
            if not math.isfinite(registration_iou):
                trainer.should_stop = True
                return

            registration_improved = (
                registration_iou
                > self.best_registration_iou + self.registration_min_delta
            )
            if registration_improved:
                self.best_registration_iou = registration_iou

        if segmentation_improved or registration_improved:
            self.wait_count = 0
            return

        self.wait_count += 1
        if self.wait_count >= self.patience:
            trainer.should_stop = True
