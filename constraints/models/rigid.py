from typing import cast

import timm
import torch

from ..datatools.label_schema import LabelSchema
from ..types import RigidParams, TransformSpec
from .composed import SegmentationRegistrationModel
from .helpers import RigidTransformHead
from .segmentator import get_segmentator

# class TwoBranch(torch.nn.Module):
#     def __init__(self, max_translation=0.5):
#         super().__init__()
#         raise ValueError("TwoBranch is deprecated.")
#         self.unet = smp.Unet(
#             "resnet18",
#             encoder_weights="imagenet",
#             in_channels=1,
#             classes=ARTIFICIAL_MASK_NUM_CLASSES,
#         )
#         # keep it simple - linear layer to each layer of the UNET encoder
#         self.projector = torch.nn.LazyConv2d(64, kernel_size=1)
#         self.output_angle_layer = torch.nn.Linear(64, 2)
#         self.output_translation_layer = torch.nn.Linear(64, 2)
#         self.max_translation = max_translation
#         # angle = torch.nn.Parameter(torch.tensor([[0.0]]))
#         # self.register_parameter("angle", angle)

#     def forward(self, x):
#         raise ValueError("TwoBranch is deprecated.")
#         encoded = self.unet.encoder(x)
#         bottleneck = encoded[-1]
#         decoded = self.unet.segmentation_head(self.unet.decoder(encoded))
#         projected = self.projector(bottleneck).mean(
#             dim=(2, 3)
#         )  # Global average pooling
#         angle_vec = self.output_angle_layer(projected)
#         translation_vec = self.output_translation_layer(projected)
#         angle = torch.atan2(angle_vec[:, 0], angle_vec[:, 1]).view(-1, 1)
#         translation = torch.tanh(translation_vec) * self.max_translation
#         return decoded, angle, translation


class RigidRegistrationNet(torch.nn.Module):
    def __init__(
        self,
        ls: LabelSchema,
        max_translation=0.5,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet34",
            pretrained=True,
            in_chans=2 * ls.num_classes,
            num_classes=0,
            global_pool="avg",
        )
        self.transform_head = RigidTransformHead(max_translation=max_translation)

    def forward(
        self, registration_input: torch.Tensor, template: torch.Tensor
    ) -> TransformSpec:
        concatenated_input = torch.cat([registration_input, template], dim=1)
        features = self.encoder(concatenated_input)
        angle, translation = self.transform_head(features)
        rigid = RigidParams(angle=angle, dx=translation[:, 0:1], dy=translation[:, 1:2])
        return TransformSpec(rigid=rigid)


class ProjectWithTemplateRigid(SegmentationRegistrationModel):
    """
    Encode->Decode segmentations, then pass the segmentation map and template into registration network
    Registration network processes both, return the transformation parameters
    Transformation parameters are applied to spatial transform

    ===
    Return: segmentation map, angle, translation
    ===
    """

    def __init__(self, ls: LabelSchema, max_translation=0.5):
        segmentation_net = get_segmentator(num_classes=ls.num_classes)
        super().__init__(
            segmentation_net=segmentation_net,
            registration_net=RigidRegistrationNet(
                max_translation=max_translation, ls=ls
            ),
            registration_input_mode="probabilities",
        )

    @property
    def unet(self) -> torch.nn.Module:
        return self.segmentation_net

    @property
    def encoder(self) -> torch.nn.Module:
        registration_net = cast(RigidRegistrationNet, self.registration_net)
        return registration_net.encoder

    @property
    def TransformHead(self) -> RigidTransformHead:
        registration_net = cast(RigidRegistrationNet, self.registration_net)
        return registration_net.transform_head
