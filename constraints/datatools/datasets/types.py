from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import torch


class Sample(TypedDict):
    image: torch.Tensor  # [C, H, W]
    target_labels: torch.Tensor  # [H,W], torch.long
    sample_id: str
    sdf: NotRequired[torch.Tensor]  # [N,H,W]
    transform: NotRequired[torch.Tensor]
    template: NotRequired[torch.Tensor]  # [H, W]
    template_sdf: NotRequired[torch.Tensor]  # [N, H, W]
    template_indexes: NotRequired[list[int]]  # which template from bank are relevant


@dataclass
class TemplateBank:
    templates: torch.Tensor


@dataclass
class TemplateAssets:
    """
    Information object carrying either None
    """

    # mode: Literal["bank", "per_sample"]
    template_bank: TemplateBank | None = None


@dataclass
class TemplateBatch:
    masks: torch.Tensor  # [B,C,H,W]
    sdfs: torch.Tensor | None = None
