from abc import ABC, abstractmethod
from turtle import forward

import torch
from torch import nn

from .datasets.types import TemplateAssets, TemplateBatch


class TemplateRefiner(nn.Module, ABC):
    @abstractmethod
    def forward(self, template_batch: TemplateBatch) -> TemplateBatch:
        pass


class IdentityTemplateRefiner(TemplateRefiner):
    def forward(self, template_batch: TemplateBatch):
        return template_batch
