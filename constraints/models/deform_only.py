from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
import torch.nn.functional as nnf

from ..datatools.label_schema import LabelSchema
from ..types import FieldParams, TransformSpec
from ..voxelmorph import modules
from ..voxelmorph.models import VxmPairwise
from .composed import SegmentationRegistrationModel
from .segmentator import get_segmentator

# class TwoBranch(torch.nn.Module):
#     """
#     Shared-encoder, two-decoder model for single-image segmentation and
#     template deformation.

#     Inputs
#     ------
#     x : torch.Tensor
#         Single image tensor, shape (B, C, H, W).

#     Outputs
#     -------
#     segmentation_logits : torch.Tensor
#         Segmentation branch output, shape (B, target_channels, H, W).
#     field : torch.Tensor
#         Velocity or displacement field from deformation branch, shape (B, 2, H, W).
#     warped_template : torch.Tensor (optional)
#         Implicit template warped by displacement field, shape (B, C, H, W).
#     """

#     def __init__(
#         self,
#         source_channels: int = 1,
#         target_channels: int = ARTIFICIAL_MASK_NUM_CLASSES,
#         integration_steps: int = 5,
#         nb_features: Sequence[int] = (16, 16, 16, 16, 16),
#         encoder_name: str = "resnet18",
#         encoder_weights: str | None = "imagenet",
#         flow_initializer: float = 1e-5,
#     ):
#         super().__init__()
#         raise ValueError("TwoBranch is deprecated.")

#         self.target_channels = target_channels

#         self.unet = smp.Unet(
#             encoder_name=encoder_name,
#             encoder_weights=encoder_weights,
#             in_channels=source_channels,
#             classes=target_channels,
#             encoder_depth=len(nb_features),
#             decoder_channels=nb_features,
#         )
#         self.source_channels = source_channels
#         self.integration_steps = integration_steps

#         # Shared encoder + task-specific decoders/heads
#         self.encoder = self.unet.encoder
#         self.segmentation_decoder = self.unet.decoder
#         self.segmentation_head = self.unet.segmentation_head

#         self.deformation_decoder = UnetDecoder(
#             encoder_channels=self.encoder.out_channels,
#             decoder_channels=tuple(nb_features),
#             n_blocks=len(nb_features),
#         )
#         self.flow_head = SegmentationHead(
#             in_channels=nb_features[-1],
#             out_channels=2,  # 2D velocity/displacement components
#             kernel_size=3,
#             activation=None,
#             upsampling=1,
#         )

#         self._init_flow_head(flow_initializer=flow_initializer)

#         if self.integration_steps > 0:
#             self.velocity_field_integrator = modules.IntegrateVelocityField(
#                 steps=self.integration_steps
#             )

#         self.spatial_transformer = modules.SpatialTransformer()

#     def forward(
#         self,
#         x: torch.Tensor,
#         template: torch.Tensor | None = None,
#         return_field_type: Literal["displacement", "velocity", "svf"] = "displacement",
#     ) -> tuple[torch.Tensor, ...]:
#         raise ValueError("TwoBranch is deprecated.")

#         valid_field_types = {"velocity", "svf", "displacement"}
#         if return_field_type not in valid_field_types:
#             raise ValueError(
#                 f"return_field_type must be one of {valid_field_types}, got '{return_field_type}'"
#             )

#         encoded = self.encoder(x)

#         segmentation_logits = self.segmentation_head(self.segmentation_decoder(encoded))
#         deformation_features = self.deformation_decoder(encoded)
#         velocity = self.flow_head(deformation_features)

#         displacement = velocity
#         if self.integration_steps > 0:
#             displacement = self.velocity_field_integrator(velocity)

#         return_field = displacement if return_field_type == "displacement" else velocity

#         if template is None:
#             return segmentation_logits, return_field

#         template_to_warp = template.detach()
#         self._validate_template_shape(template_to_warp)

#         if template_to_warp.shape[2:] != x.shape[2:]:
#             template_to_warp = nnf.interpolate(
#                 template_to_warp, size=x.shape[2:], mode="nearest"
#             )
#         template_to_warp = template_to_warp.expand(x.shape[0], -1, -1, -1)

#         warped_template = self.spatial_transformer(template_to_warp, displacement)
#         return segmentation_logits, return_field, warped_template

# def _init_flow_head(self, flow_initializer: float = 1e-5) -> None:
#     """
#     Initialize first flow-head convolution with small random weights.
#     """
#     if flow_initializer is None:
#         return

#     first_conv = None
#     for module in self.flow_head.modules():
#         if isinstance(module, torch.nn.Conv2d):
#             first_conv = module
#             break

#     if first_conv is None:
#         return

#     with torch.no_grad():
#         torch.nn.init.normal_(first_conv.weight, mean=0.0, std=flow_initializer)
#         if first_conv.bias is not None:
#             first_conv.bias.zero_()

# def _validate_template_shape(self, template: torch.Tensor) -> None:
#     if template.ndim != 4:
#         raise ValueError(
#             f"template must have 4 dims (B,C,H,W), got shape {tuple(template.shape)}"
#         )
#     if template.shape[1] != self.target_channels:
#         raise ValueError(
#             f"template channels ({template.shape[1]}) must match target_channels "
#             f"({self.target_channels})"
#         )


class DeformableRegistrationNet(torch.nn.Module):
    def __init__(self, ls: LabelSchema) -> None:
        super().__init__()
        nb_features: Any = [
            [32, 32, 32, 32],
            [32, 32, 32, 32],
        ]
        self.encoder = VxmPairwise(
            ndim=2,
            source_channels=ls.num_classes,
            target_channels=ls.num_classes,
            nb_features=nb_features,
        )

    def forward(
        self, registration_input: torch.Tensor, template: torch.Tensor
    ) -> TransformSpec:
        deformation_field = self.encoder(registration_input, template)
        field_params = FieldParams(field=deformation_field)
        return TransformSpec(field=field_params)


class ProjectWithTemplateD(SegmentationRegistrationModel):
    """
    Encode->Decode segmentations, then pass the segmentation map (or ground
    truth, if provided) and template into a registration network.
    The registration network processes both and returns a 2D deformation field.
    ---
    Return: segmentation logits, deformation field
    ---
    """

    def __init__(self, ls: LabelSchema) -> None:
        segmentation_net = get_segmentator(num_classes=ls.num_classes)
        super().__init__(
            segmentation_net=segmentation_net,
            registration_net=DeformableRegistrationNet(ls=ls),
            registration_input_mode="probabilities",
        )

    @property
    def unet(self) -> torch.nn.Module:
        return self.segmentation_net

    @property
    def encoder(self) -> torch.nn.Module:
        return cast(torch.nn.Module, self.registration_net.encoder)
