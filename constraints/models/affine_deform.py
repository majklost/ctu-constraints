from typing import cast

import torch

from ..datatools.label_schema import LabelSchema
from ..transforms.transformers import RigidTransformer
from ..types import TransformSpec
from .affine import AffineRegistrationNet
from .composed import SegmentationRegistrationModel
from .deform_only import DeformableRegistrationNet
from .helpers import MomentsAffineAlignment
from .segmentator import get_segmentator


class DeepAffDefRegistrationNet(torch.nn.Module):
    """
    Registration net that predicts first affine parameters and then a deformation field
    """

    def __init__(self, label_schema: LabelSchema, max_translation=0.3):
        super().__init__()
        self.rigid_transformer = (
            RigidTransformer()
        )  # must transform before passing to next component
        self.affine_registration_net = AffineRegistrationNet(
            ls=label_schema, max_translation=max_translation
        )
        self.deformable_registration_net = DeformableRegistrationNet(ls=label_schema)

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


class CalcAffDefRegistrationNet(torch.nn.Module):
    """
    Registration net that uses moment alignment to calculate affine parameters,
    then predicts a deformation field.
    """

    def __init__(self, label_schema: LabelSchema, max_translation=0.3):
        super().__init__()
        self.rigid_transformer = (
            RigidTransformer()
        )  # must transform before passing to next component
        self.deformable_registration_net = DeformableRegistrationNet(ls=label_schema)
        self.moments_affine_alignment = MomentsAffineAlignment()

    def forward(
        self, registration_input: torch.Tensor, template: torch.Tensor
    ) -> TransformSpec:
        rigid_transform_spec = self.moments_affine_alignment(
            registration_input, template
        )
        warped_template = self.rigid_transformer(template, rigid_transform_spec)
        deformable_transform_spec = self.deformable_registration_net(
            registration_input, warped_template.warped_template
        )
        return TransformSpec(steps=(rigid_transform_spec, deformable_transform_spec))


class ProjectWithTemplateBDeepAff(SegmentationRegistrationModel):
    """Segment an image, then register it with predicted affine and deformable
    stages."""

    def __init__(
        self,
        ls: LabelSchema,
        max_translation: float = 0.3,
    ) -> None:
        super().__init__(
            segmentation_net=get_segmentator(ls.num_classes),
            registration_net=DeepAffDefRegistrationNet(
                label_schema=ls, max_translation=max_translation
            ),
            registration_input_mode="probabilities",
        )

    @property
    def unet(self) -> torch.nn.Module:
        return self.segmentation_net

    @property
    def encoder(self) -> torch.nn.Module:
        registration_net = cast(DeepAffDefRegistrationNet, self.registration_net)
        return registration_net.affine_registration_net.encoder


class ProjectWithTemplateBCalcAff(SegmentationRegistrationModel):
    """Segment an image, then register it with a moment-aligned affine stage
    followed by a deformable one."""

    def __init__(self, label_schema: LabelSchema, max_translation: float = 0.3) -> None:
        super().__init__(
            segmentation_net=get_segmentator(label_schema.num_classes),
            registration_net=CalcAffDefRegistrationNet(
                label_schema=label_schema, max_translation=max_translation
            ),
            # Moment alignment is only defined on non-negative input: on raw logits
            # the centroid denominator goes negative and the affine stage collapses
            # to (roughly) the image centre with a flipped sign.
            registration_input_mode="probabilities",
        )

    @property
    def unet(self) -> torch.nn.Module:
        return self.segmentation_net

    @property
    def encoder(self) -> torch.nn.Module:
        registration_net = cast(CalcAffDefRegistrationNet, self.registration_net)
        return registration_net.deformable_registration_net.encoder
