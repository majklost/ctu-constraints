from abc import ABC, abstractmethod

from torch import nn

from .datasets.types import Batch, TemplateAssets, TemplateBatch
from .label_schema import LabelSchema


class TemplateSource(nn.Module, ABC):
    """
    Provides the template from dataset
    """

    def __init__(
        self, template_assets: TemplateAssets, label_schema: LabelSchema
    ) -> None:
        super().__init__()
        self._template_assets = template_assets
        self._label_schema = label_schema

    @abstractmethod
    def forward(self, batch: Batch) -> TemplateBatch:
        raise NotImplementedError("Abstract stub")


class PerSampleTemplateSource(TemplateSource):
    def forward(self, batch: Batch) -> TemplateBatch:
        if "template" not in batch:
            raise KeyError("PerSampleTemplateSource requires a 'template' batch field.")

        return TemplateBatch(
            masks=self._label_schema.label_map_to_one_hot(batch["template"]).float(),
            sdfs=batch.get("template_sdf"),
        )


class BankTemplateSource(TemplateSource):
    def __init__(
        self, template_assets: TemplateAssets, label_schema: LabelSchema
    ) -> None:
        super().__init__(template_assets, label_schema)
        self._validate()
        # TODO register buffer

    def _validate(self) -> bool:
        return True
