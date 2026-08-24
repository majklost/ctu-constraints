"""
Various callbacks for Lightning Wrappers.
- debugging, inspection
- saving runs, etc...
"""

import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from constraints.utils import REPO_ROOT, get_weights_folder


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

    Models without ``val/epoch/registration/iou/warped_vs_gt``, such as the plain
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
        segmentation_metric = metrics.get("val/epoch/segmentation/iou/pred_vs_gt")
        if segmentation_metric is None:
            raise RuntimeError(
                "SegmentationRegistrationEarlyStopping requires "
                "'val/epoch/segmentation/iou/pred_vs_gt'."
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
        registration_metric = metrics.get("val/epoch/registration/iou/warped_vs_gt")
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


class InferenceWeightsCheckpoint(ModelCheckpoint):
    """Save the best weights-only Lightning checkpoint and run metadata.

    The monitored metric is selected on the first real validation pass:
    registration IoU is preferred, with segmentation IoU as the fallback for
    models such as the U-Net baseline.
    """

    _SEGMENTATION_METRIC = "val/epoch/segmentation/iou/pred_vs_gt"
    _REGISTRATION_METRIC = "val/epoch/registration/iou/warped_vs_gt"

    def __init__(
        self,
        *,
        experiment: str | Path,
        filename: str,
        run_id: str,
    ) -> None:
        folder = get_weights_folder(experiment, filename, run_id)
        super().__init__(
            dirpath=folder,
            filename="weights",
            monitor=None,
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self.experiment_filename = filename
        self.run_id = run_id

    @staticmethod
    def _git_metadata() -> dict[str, Any]:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            commit = os.environ.get("CTU_GIT_COMMIT")
            dirty = None
        return {"git_commit": commit, "git_dirty_worktree": dirty}

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return

        if self.monitor is None:
            self.monitor = self._select_monitor(trainer.callback_metrics)
        super().on_validation_end(trainer, pl_module)

    @classmethod
    def _select_monitor(cls, metrics: dict[str, Any]) -> str:
        if cls._REGISTRATION_METRIC in metrics:
            return cls._REGISTRATION_METRIC
        if cls._SEGMENTATION_METRIC in metrics:
            return cls._SEGMENTATION_METRIC
        raise RuntimeError(
            "InferenceWeightsCheckpoint requires either "
            f"'{cls._REGISTRATION_METRIC}' or '{cls._SEGMENTATION_METRIC}'."
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero or not self.best_model_path:
            return

        run = getattr(trainer.logger, "experiment", None)
        weights_path = Path(self.best_model_path)
        run_config = getattr(run, "config", {})

        metadata = {
            "wandb": {
                "name": getattr(run, "name", None),
                "id": self.run_id,
                "url": getattr(run, "url", None),
            },
            "creation_time": datetime.now(UTC).isoformat(),
            "experiment_filename": self.experiment_filename,
            "command_line_arguments": dict(run_config),
            "command": " ".join(sys.argv),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "checkpoint_selection": {
                "metric": self.monitor,
                "value": (
                    float(self.best_model_score.detach().cpu().item())
                    if self.best_model_score is not None
                    else None
                ),
            },
            "model_configuration": {
                "lightning_module_class": (
                    f"{type(pl_module).__module__}.{type(pl_module).__qualname__}"
                ),
                "wandb_config": dict(run_config),
            },
            **self._git_metadata(),
            "weight_file_size_bytes": weights_path.stat().st_size,
        }
        with (weights_path.parent / "metadata.json").open("w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
