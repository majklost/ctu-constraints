from collections.abc import Sequence
from typing import Literal

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as nnf
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


from ..voxelmorph import modules
from ..voxelmorph.models import VxmPairwise
from ..types import FieldParams, TransformSpec

class TwoBranch(torch.nn.Module):
    """
    Shared-encoder, two-decoder model for single-image segmentation and
    template deformation.

    Inputs
    ------
    x : torch.Tensor
        Single image tensor, shape (B, C, H, W).

    Outputs
    -------
    segmentation_logits : torch.Tensor
        Segmentation branch output, shape (B, target_channels, H, W).
    field : torch.Tensor
        Velocity or displacement field from deformation branch, shape (B, 2, H, W).
    warped_template : torch.Tensor (optional)
        Implicit template warped by displacement field, shape (B, C, H, W).
    """

    def __init__(
        self,
        source_channels: int = 1,
        target_channels: int = 3,
        integration_steps: int = 5,
        nb_features: Sequence[int] = (16, 16, 16, 16, 16),
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        flow_initializer: float = 1e-5,
    ):
        super().__init__()
        raise ValueError("TwoBranch is deprecated.")

        self.target_channels = target_channels

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=source_channels,
            classes=target_channels,
            encoder_depth=len(nb_features),
            decoder_channels=nb_features,
        )
        self.source_channels = source_channels
        self.integration_steps = integration_steps

        # Shared encoder + task-specific decoders/heads
        self.encoder = self.unet.encoder
        self.segmentation_decoder = self.unet.decoder
        self.segmentation_head = self.unet.segmentation_head

        self.deformation_decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=tuple(nb_features),
            n_blocks=len(nb_features),
        )
        self.flow_head = SegmentationHead(
            in_channels=nb_features[-1],
            out_channels=2,  # 2D velocity/displacement components
            kernel_size=3,
            activation=None,
            upsampling=1,
        )

        self._init_flow_head(flow_initializer=flow_initializer)

        if self.integration_steps > 0:
            self.velocity_field_integrator = modules.IntegrateVelocityField(
                steps=self.integration_steps
            )

        self.spatial_transformer = modules.SpatialTransformer()

    def forward(
        self,
        x: torch.Tensor,
        template: torch.Tensor | None = None,
        return_field_type: Literal["displacement", "velocity", "svf"] = "displacement",
    ) -> tuple[torch.Tensor, ...]:
        raise ValueError("TwoBranch is deprecated.")

        valid_field_types = {"velocity", "svf", "displacement"}
        if return_field_type not in valid_field_types:
            raise ValueError(
                f"return_field_type must be one of {valid_field_types}, got '{return_field_type}'"
            )

        encoded = self.encoder(x)

        segmentation_logits = self.segmentation_head(self.segmentation_decoder(encoded))
        deformation_features = self.deformation_decoder(encoded)
        velocity = self.flow_head(deformation_features)

        displacement = velocity
        if self.integration_steps > 0:
            displacement = self.velocity_field_integrator(velocity)

        return_field = displacement if return_field_type == "displacement" else velocity

        if template is None:
            return segmentation_logits, return_field

        template_to_warp = template.detach()
        self._validate_template_shape(template_to_warp)

        if template_to_warp.shape[2:] != x.shape[2:]:
            template_to_warp = nnf.interpolate(
                template_to_warp, size=x.shape[2:], mode="nearest"
            )
        template_to_warp = template_to_warp.expand(x.shape[0], -1, -1, -1)

        warped_template = self.spatial_transformer(template_to_warp, displacement)
        return segmentation_logits, return_field, warped_template

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

    def _validate_template_shape(self, template: torch.Tensor) -> None:
        if template.ndim != 4:
            raise ValueError(
                f"template must have 4 dims (B,C,H,W), got shape {tuple(template.shape)}"
            )
        if template.shape[1] != self.target_channels:
            raise ValueError(
                f"template channels ({template.shape[1]}) must match target_channels "
                f"({self.target_channels})"
            )

class ProjectWithTemplateD(torch.nn.Module):
    """
    Encode->Decode segmentations, then pass the segmentation map (or ground
    truth, if provided) and template into a registration network.
    The registration network processes both and returns a 2D deformation field.
    ---
    Return: segmentation logits, deformation field
    ---
    """

    def __init__(self) -> None:
        super().__init__()
        self.unet = smp.Unet(
            "resnet18", encoder_weights="imagenet", in_channels=1, classes=3
        )

        nb_features = [
            [32, 32, 32, 32],  # encoder features
            [32, 32, 32, 32],  # decoder features
        ]
        self.encoder = VxmPairwise(
            ndim=2,
            source_channels=3,
            target_channels=3,
            nb_features=nb_features,
        )

    def forward(
        self,
        x: torch.Tensor,
        template: torch.Tensor,
        gt: torch.Tensor |None = None,
        detach_seg: bool = False,
    ) -> tuple[torch.Tensor, TransformSpec]:
        """
        Run the UNet to obtain segmentation logits, then feed either those
        logits or a supplied ground-truth mask into the registration network
        together with the template.

        Args:
            x: Input image, shape (B, 1, H, W).
            template: Topologically-correct one-hot template, shape (B, C, H, W).
            gt: Optional ground-truth one-hot segmentation, shape (B, C, H, W).
                If provided, this is passed to the registration network
                instead of the UNet's predicted logits, decoupling the
                registration network's training from the UNet entirely
                (no gradient path between the two).
                If None, the UNet's own logits are used instead.
            detach_seg: Only relevant when gt is None. If True, stop-gradient
                is applied to the segmentation logits before they enter the
                registration network, so the warp loss updates the
                registration network only and does not backpropagate into
                the UNet's weights.

        Returns:
            segmentation_logits: Raw UNet output, shape (B, C, H, W).
            transform_spec: Predicted deformation field wrapped as a
                TransformSpec, to be applied to the template via a spatial
                transform.
        """
        segmentation_logits: torch.Tensor = self.unet(x)  # (B, C, H, W)

        registration_input: torch.Tensor
        if gt is not None:
            registration_input = gt
        else:
            registration_input = segmentation_logits
            if detach_seg:
                registration_input = registration_input.detach()

        deformation_field: torch.Tensor = self.encoder(registration_input, template)
        field_params = FieldParams(field=deformation_field)
        transform_spec = TransformSpec(field=field_params)

        return segmentation_logits, transform_spec
