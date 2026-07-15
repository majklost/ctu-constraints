import torch
import pytorch_lightning as pl
from torch import nn
from ..types import LossResult
from ..datatools.datasets import Sample
from ..types import WarpResult,LossInput
from ..transforms.transformers import SpatialTransformer
from ..computers.loss_computers import ProjectLossComputer

#TODO: Handling template better save memory, template is same for all samples in the batch


class ProjectLightning(pl.LightningModule):
    """
    The Project Architecture implemented in Lightning
    """
    def __init__(self, model: nn.Module,
                 spatial_transform: SpatialTransformer,
                 loss_computer: ProjectLossComputer,

                  ):
        super().__init__()
        self.model = model
        self.spatial_transform = spatial_transform
        self.loss_computer = loss_computer

    def _extra_logging(self,
                       stage: str,
                       batch: Sample,
                       batch_idx: int,
                       loss_output: LossResult,
                       ) -> None:
        """Extension hook for ad-hoc notebook diagnostics.

        Subclasses can override this to log additional experiment-specific values
        without modifying the stable baseline logging in `_shared_step`.
        """
        return None

    def forward(self, img:torch.Tensor, template:torch.Tensor):
        segmentation_logits, transform_spec = self.model(img, template)
        warp_result = self.spatial_transform(template, transform_spec)
        return segmentation_logits, warp_result

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

        self._extra_logging(
            stage=stage,
            batch=batch,
            batch_idx=batch_idx,
            loss_output=loss_output,
        )

        return loss_output.total

    def training_step(self, batch:Sample, batch_idx):
        return self._shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch:Sample, batch_idx):
        return self._shared_step(batch, batch_idx, stage='val')
