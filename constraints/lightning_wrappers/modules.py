import torch
import pytorch_lightning as pl
from torch import nn
from ..datatools.datasets import Sample
from ..types import LossInput, MetricInput, WandbOverlay
from ..transforms.transformers import SpatialTransformer
from ..computers.loss_computers import ProjectLossComputer
from ..computers.metric_computers import ProjectMetricComputer, NoOpMetricComputer

try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:  # pragma: no cover - optional dependency
    WandbLogger = None

try:
    import wandb
except Exception:  # pragma: no cover - optional runtime dependency
    wandb = None

#TODO: Handling template better save memory, template is same for all samples in the batch


class ProjectLightning(pl.LightningModule):
    """
    The Project Architecture implemented in Lightning
    """
    def __init__(self, model: nn.Module,
                 spatial_transform: SpatialTransformer,
                 loss_computer: ProjectLossComputer,
                 metric_computer: ProjectMetricComputer | None = None,

                  ):
        super().__init__()
        self.model = model
        self.spatial_transform = spatial_transform
        self.loss_computer = loss_computer
        self.metric_computer = metric_computer if metric_computer is not None else NoOpMetricComputer()

    def forward(self, img:torch.Tensor, template:torch.Tensor):
        segmentation_logits, transform_spec = self.model(img, template)
        warp_result = self.spatial_transform(template, transform_spec)
        return segmentation_logits, warp_result

    def _log_wandb_overlay(self, stage: str, image_tag: str, overlay: WandbOverlay) -> None:
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

        # Decouple this key from global_step's monotonic counter — define once, cheap to call repeatedly
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

    def _shared_step(self,batch:Sample,batch_idx,stage:str):
        template = batch['template'] # template is same for all samples in the batch, so we can take the first one
        img = batch['image']
        segmentation_logits, warp_result = self.forward(img, template)
        # Plug into loss computer
        loss_input = LossInput(segmentation_logits=segmentation_logits,
                               warped_template=warp_result.warped_template,
                               gt_mask=batch['mask'],
                               gt_mask_sdf=batch['sdf'],
                               transform_spec=warp_result.transform_spec,
                               )
        loss_output = self.loss_computer.compute(loss_input)

        # Logging
        self.log(f"{stage}/loss", loss_output.total, on_step=True, on_epoch=True, prog_bar=(stage == "train"))

        if loss_output.components:
            for component_name, component_value in loss_output.components.items():
                self.log(f"{stage}/loss/{component_name}", component_value, on_step=True, on_epoch=True)

        if loss_output.logs:
            for log_name, log_value in loss_output.logs.items():
                self.log(f"{stage}/{log_name}", log_value, on_step=True, on_epoch=True)

        metric_input = MetricInput(
            stage=stage,
            batch_idx=batch_idx,
            current_epoch=int(self.current_epoch),
            global_step=int(self.global_step),
            image=img,
            segmentation_logits=segmentation_logits,
            warped_template=warp_result.warped_template,
            gt_mask=batch['mask'],
            gt_mask_sdf=batch['sdf'],
            transform_spec=warp_result.transform_spec,
        )
        metric_output = self.metric_computer.compute(metric_input)

        if metric_output.logs:
            self.log_dict(
                {f"{stage}/{name}": value for name, value in metric_output.logs.items()},
                on_step=True,
                on_epoch=True,
            )

        if metric_output.wandb_overlays:
            for image_tag, overlay in metric_output.wandb_overlays.items():
                self._log_wandb_overlay(stage=stage, image_tag=image_tag, overlay=overlay)

        return loss_output.total

    def training_step(self, batch:Sample, batch_idx):
        return self._shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch:Sample, batch_idx):
        return self._shared_step(batch, batch_idx, stage='val')
