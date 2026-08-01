from typing import Literal

import torch
from torch import nn

from ..types import TransformSpec


RegistrationInputMode = Literal["logits", "probabilities"]


class SegmentationRegistrationModel(nn.Module):
    """Compose a segmentation network with a registration network."""

    def __init__(
        self,
        segmentation_net: nn.Module,
        registration_net: nn.Module,
        registration_input_mode: RegistrationInputMode = "logits",
    ) -> None:
        super().__init__()
        if registration_input_mode not in {"logits", "probabilities"}:
            raise ValueError(
                "registration_input_mode must be one of {'logits', 'probabilities'}"
            )
        self.segmentation_net = segmentation_net
        self.registration_net = registration_net
        self.registration_input_mode = registration_input_mode

    def _prepare_registration_input(
        self,
        segmentation_logits: torch.Tensor,
        gt: torch.Tensor | None,
        detach_seg: bool,
    ) -> torch.Tensor:
        if gt is not None:
            return gt

        if self.registration_input_mode == "probabilities":
            registration_input = torch.softmax(segmentation_logits, dim=1)
        else:
            registration_input = segmentation_logits

        if detach_seg:
            registration_input = registration_input.detach()

        return registration_input

    def forward(
        self,
        x: torch.Tensor,
        template: torch.Tensor,
        gt: torch.Tensor | None = None,
        detach_seg: bool = False,
    ) -> tuple[torch.Tensor, TransformSpec]:
        segmentation_logits = self.segmentation_net(x)
        registration_input = self._prepare_registration_input(
            segmentation_logits=segmentation_logits,
            gt=gt,
            detach_seg=detach_seg,
        )
        transform_spec = self.registration_net(registration_input, template)
        return segmentation_logits, transform_spec
