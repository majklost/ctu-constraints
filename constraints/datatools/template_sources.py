from abc import ABC, abstractmethod

import torch
from torch import nn

from .datasets.types import Batch, TemplateAssets, TemplateBatch


class TemplateSource(nn.Module, ABC):
    """
    Provides the template from dataset
    """

    def __init__(self, template_assets: TemplateAssets) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, sample: Batch) -> TemplateBatch:
        raise NotImplementedError("Abstract stub")


class PerSampleTemplateSource(TemplateSource):
    def forward(self, sample: Batch) -> TemplateBatch:
        assert "template" in sample, (
            "PerSampleTemplateSource received a sample withou template"
        )

        tb = TemplateBatch(masks=sample["template"], sdfs=sample.get("template_sdf"))
        return tb


class BankTemplateSource(TemplateSource):
    def __init__(self, template_assets: TemplateAssets) -> None:
        super().__init__(template_assets)
        self._validate()
        # TODO register buffer

    def _validate(self) -> bool:
        return True
