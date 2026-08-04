from typing import cast

import torch

from ..transforms.transformers import RigidTransformer
from ..types import TransformSpec
from .affine import AffineRegistrationNet
from .composed import SegmentationRegistrationModel
from .deform_only import DeformableRegistrationNet
from .segmentator import get_segmentator


class DeepAffDefRegistrationNet(torch.nn.Module):
    """
    Registration net that predicts first affine parameters and then a deformation field
    """

    def __init__(self, max_translation=0.3):
        super().__init__()
        self.rigid_transformer = (
            RigidTransformer()
        )  # must transform before passing to next component
        self.affine_registration_net = AffineRegistrationNet(
            max_translation=max_translation
        )
        self.deformable_registration_net = DeformableRegistrationNet()

    def forward(
        self, registration_input: torch.Tensor, template: torch.Tensor
    ) -> TransformSpec:
        rigid_transform_spec = self.affine_registration_net(
            registration_input, template
        )
        warped_template = self.rigid_transformer(template, rigid_transform_spec)
        deformable_transform_spec = self.deformable_registration_net(
            registration_input, warped_template.warped_template
        )
        return TransformSpec(steps=(rigid_transform_spec, deformable_transform_spec))


class ProjectWithTemplateBDeepAff(SegmentationRegistrationModel):
    """Segment an image, then register its logits with affine and deformable stages."""

    def __init__(self, max_translation: float = 0.3) -> None:
        super().__init__(
            segmentation_net=get_segmentator(),
            registration_net=DeepAffDefRegistrationNet(max_translation=max_translation),
            registration_input_mode="logits",
        )

    @property
    def unet(self) -> torch.nn.Module:
        return self.segmentation_net

    @property
    def encoder(self) -> torch.nn.Module:
        registration_net = cast(DeepAffDefRegistrationNet, self.registration_net)
        return registration_net.affine_registration_net.encoder


class CalcAffDefRegistrationNet(torch.nn.Module):
    """
    Registration net that uses moment alignment to calculate affine parameters,
    then predicts a deformation field.
    """

    def __init__(self, max_translation=0.3):
        super().__init__()
        pass
