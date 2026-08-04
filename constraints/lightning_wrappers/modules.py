from collections.abc import Callable

import pytorch_lightning as pl
import torch

# Whatever configure_optimizers() itself may legally return
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
import torch.nn.functional as F
from ..computers.loss_computers import ProjectLossComputer
from ..computers.metric_computers import (
    DefaultSegmentationMetricComputer,
    NoOpMetricComputer,
    ProjectMetricComputer,
)
from ..datatools.datasets import (
    ARTIFICIAL_MASK_NUM_CLASSES,
    Sample,
    artificial_mask_to_label_map,
)
from ..transforms.transformers import SpatialTransformer
from ..types import LossInput, MetricInput, WandbOverlay
from .sample_strategy import GtStrategy, NoGt

OptimizerFactory = Callable[[nn.Module], OptimizerLRScheduler]

from pytorch_lightning.loggers import WandbLogger

from ..models.segmentator import get_learned_segmentator
import wandb


# TODO: Handling template better save memory, template is same for all samples in the batch


class MetricLoggingMixin(pl.LightningModule):
    def _log_wandb_overlay(
        self, stage: str, image_tag: str, overlay: WandbOverlay
    ) -> None:
        if self.trainer is None:
            return
        if self.trainer.global_rank != 0:
            return

        logger = self.trainer.logger
        if logger is None:
            return
        if WandbLogger is None or not isinstance(logger, WandbLogger):
            return

        experiment = logger.experiment
        key = f"{stage}/{image_tag}"
        experiment.define_metric(key, step_metric=f"{stage}/epoch")

        if wandb is None:
            return

        image = overlay.image.detach().cpu().float()
        image_min = float(image.min().item())
        image_max = float(image.max().item())
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        image_uint8 = (image.clamp(0.0, 1.0).numpy() * 255).astype("uint8")

        masks: dict[str, dict[str, object]] = {}
        for mask_name, mask_tensor in overlay.masks.items():
            masks[mask_name] = {
                "mask_data": mask_tensor.detach().cpu().numpy().astype("int32"),
                "class_labels": overlay.class_labels or {},
            }

        experiment.log(
            {
                key: [wandb.Image(image_uint8, masks=masks, caption=overlay.caption)],
                f"{stage}/epoch": int(self.current_epoch),
            }
        )

    def _log_metric_output(self, stage: str, image: torch.Tensor, metric_output) -> None:
        if metric_output.logs:
            self.log_dict(
                {
                    f"{stage}/{name}": value
                    for name, value in metric_output.logs.items()
                },
                on_step=True,
                on_epoch=True,
                batch_size=image.shape[0],
            )

        if metric_output.sum_logs:
            for name, value in metric_output.sum_logs.items():
                self.log(
                    f"{stage}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx="sum",
                )

        if metric_output.wandb_overlays:
            for image_tag, overlay in metric_output.wandb_overlays.items():
                self._log_wandb_overlay(
                    stage=stage, image_tag=image_tag, overlay=overlay
                )


class ProjectLightning(MetricLoggingMixin):
    """The project architecture implemented in Lightning.

    Args:
        gt_strategy: Chooses whether ground truth is passed to the model during
            training and the primary validation pass. Its results are logged as
            ``train/*`` and ``val/*``.
        validation_strategy: Optionally runs a second validation forward pass
            for every batch. It is intended for a logits-only strategy such as
            ``NoGt()`` and logs independently as ``val_logits/*``. Set it to
            ``None`` to run only the primary validation pass.
    """

    def __init__(
        self,
        model: nn.Module,
        spatial_transform: SpatialTransformer,
        loss_computer: ProjectLossComputer,
        metric_computer: ProjectMetricComputer | None = None,
        optimizer_callback: OptimizerFactory = lambda module: torch.optim.Adam(
            module.parameters(), lr=1e-3
        ),
        gt_strategy: GtStrategy = NoGt(),
        template_as_sdf: bool = False,
        validation_strategy: GtStrategy | None = None,
    ):

        super().__init__()
        self.model = model
        self.spatial_transform = spatial_transform
        self.loss_computer = loss_computer
        self.metric_computer = (
            metric_computer if metric_computer is not None else NoOpMetricComputer()
        )
        self.optimizer_callback = optimizer_callback
        self.template_as_sdf = template_as_sdf
        self.gt_strategy = gt_strategy
        self.validation_strategy = validation_strategy

    def forward(
        self,
        img: torch.Tensor,
        template: torch.Tensor,
        template_sdf: torch.Tensor | None = None,
        gt: torch.Tensor | None = None,
        detach_seg: bool = False,
    ):
        segmentation_logits, transform_spec = self.model(
            img, template, gt=gt, detach_seg=detach_seg
        )
        if template_sdf is None:
            warp_result = self.spatial_transform(template, transform_spec)
        else:
            warp_result = self.spatial_transform(template_sdf, transform_spec)
        return segmentation_logits, warp_result

    def _shared_step(
        self,
        batch: Sample,
        batch_idx,
        stage: str,
        strategy: GtStrategy,
    ):
        template = batch[
            "template"
        ]  # template is same for all samples in the batch, so we can take the first one
        img = batch["image"]
        template_sdf = None
        if self.template_as_sdf and "template_sdf" in batch:
            template_sdf = batch["template_sdf"]

        decision = strategy.decide(batch, stage, int(self.current_epoch))
        segmentation_logits, warp_result = self.forward(
            img, template, gt=decision.gt, detach_seg=decision.detach_seg, template_sdf=template_sdf
        )
        # Plug into loss computer
        loss_input = LossInput(
            segmentation_logits=segmentation_logits,
            warped_template=warp_result.warped_template,
            gt_mask=batch["mask"],
            gt_mask_sdf=batch["sdf"],
            transform_spec=warp_result.transform_spec,
        )
        loss_output = self.loss_computer.compute(loss_input)

        # Logging
        self.log(
            f"{stage}/loss",
            loss_output.total,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage == "train"),
        )

        if loss_output.components:
            for component_name, component_value in loss_output.components.items():
                self.log(
                    f"{stage}/loss/{component_name}",
                    component_value,
                    on_step=True,
                    on_epoch=True,
                )

        if loss_output.logs:
            for log_name, log_value in loss_output.logs.items():
                self.log(
                    f"{stage}/info/{log_name}", log_value, on_step=True, on_epoch=True
                )

        metric_input = MetricInput(
            stage=stage,
            batch_idx=batch_idx,
            current_epoch=int(self.current_epoch),
            global_step=int(self.global_step),
            image=img,
            segmentation_logits=segmentation_logits,
            warped_template=warp_result.warped_template,
            gt_mask=batch["mask"],
            gt_mask_sdf=batch["sdf"],
            transform_spec=warp_result.transform_spec,
        )
        metric_output = self.metric_computer.compute(metric_input)
        self._log_metric_output(stage=stage, image=img, metric_output=metric_output)

        return loss_output.total

    def training_step(self, batch: Sample, batch_idx):
        return self._shared_step(
            batch, batch_idx, stage="train", strategy=self.gt_strategy
        )

    def validation_step(self, batch: Sample, batch_idx):
        loss = self._shared_step(
            batch, batch_idx, stage="val", strategy=self.gt_strategy
        )
        if self.validation_strategy is not None:
            self._shared_step(
                batch,
                batch_idx,
                stage="val_extra",
                strategy=self.validation_strategy,
            )
        return loss

    def configure_optimizers(self):
        return self.optimizer_callback(self)



class UnetLightning(MetricLoggingMixin):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        num_classes: int = ARTIFICIAL_MASK_NUM_CLASSES,
        metric_computer: ProjectMetricComputer | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_computer"])
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.unet = get_learned_segmentator()
        self.metric_computer = (
            metric_computer
            if metric_computer is not None
            else DefaultSegmentationMetricComputer(num_classes=num_classes)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.unet(image.float())

    def _shared_step(self, batch, batch_idx, stage: str):
        image = batch["image"]
        target_labels = artificial_mask_to_label_map(batch["mask"])
        logits = self.forward(image)
        loss = F.cross_entropy(logits, target_labels)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage == "train"),
        )

        metric_output = self.metric_computer.compute(
            MetricInput(
                stage=stage,
                batch_idx=batch_idx,
                current_epoch=int(self.current_epoch),
                global_step=int(self.global_step),
                image=image,
                segmentation_logits=logits,
                gt_mask=batch["mask"],
            )
        )
        self._log_metric_output(
            stage=stage,
            image=image,
            metric_output=metric_output,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
