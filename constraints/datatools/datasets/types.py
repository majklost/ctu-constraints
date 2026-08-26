from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import torch


class Sample(TypedDict):
    image: torch.Tensor  # [C, H, W]
    target_labels: torch.Tensor  # [H,W], torch.long
    sample_id: str
    sdf: NotRequired[torch.Tensor]  # [N,H,W]
    transform: NotRequired[torch.Tensor]
    rigid: NotRequired[torch.Tensor]  # [3]: angle radians, dx pixels, dy pixels
    template: NotRequired[torch.Tensor]  # [H, W]
    template_sdf: NotRequired[torch.Tensor]  # [N, H, W]
    template_indexes: NotRequired[list[int]]  # which template from bank are relevant


class Batch(TypedDict):
    image: torch.Tensor  # [B, C, H, W]
    target_labels: torch.Tensor  # [B, H, W], torch.long
    sample_id: list[str]  # List of length B
    sdf: NotRequired[torch.Tensor]  # [B, N, H, W]
    transform: NotRequired[torch.Tensor]  # [B, 3, 3] or [B, *dims]
    template: NotRequired[torch.Tensor]  # [B, H, W]
    template_sdf: NotRequired[torch.Tensor]  # [B, N, H, W]
    template_indexes: NotRequired[
        list[torch.Tensor]
    ]  # Length K, each tensor is shape [B]


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
