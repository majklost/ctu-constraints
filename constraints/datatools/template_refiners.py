from abc import ABC, abstractmethod
from turtle import forward

import torch
from datasets.types import TemplateAssets, TemplateBatch
from torch import nn


class TemplateRefiner(nn.Module, ABC):
    @abstractmethod
    def forward(self, template_batch: TemplateBatch) -> TemplateBatch:
        pass


class IdentityTemplateRefiner(TemplateRefiner):
    def forward(self, template_batch: TemplateBatch):
        return template_batch
