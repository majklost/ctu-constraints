import torch
import pytorch_lightning as pl
from torch import nn
from typing import Callable
from ..types import TransformSpec
from ..types import WarpResult
from ..transforms.transformers import SpatialTransformer




class ProjectLightning(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 spatial_transform: SpatialTransformer,
                  loss_seg: nn.Module,
                   loss_reg: nn.Module):
        super().__init__()
        self.model = model
        self.spatial_transform = spatial_transform
        self.loss_seg = loss_seg
        self.loss_reg = loss_reg
