from typing import NotRequired, TypedDict

import torch


class Sample(TypedDict):
    image: torch.Tensor  # [C, H, W]
    target_labels: torch.Tensor  # [H,W], torch.long
    sample_id: str
    sdf: NotRequired[torch.Tensor]  # [N,H,W]
    transform: NotRequired[torch.Tensor]
    template: NotRequired[torch.Tensor]  # [H, W]
    template_sdf: NotRequired[torch.Tensor]  # [N, H, W]
    # mask: torch.Tensor
    # template: torch.Tensor
    # sdf: torch.Tensor
    # template_sdf: NotRequired[torch.Tensor]  # key may be missing entirely
