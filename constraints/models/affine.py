import segmentation_models_pytorch as smp
import torch
import timm
from .helpers import RigidTransformHead

class TwoBranch(torch.nn.Module):
    def __init__(self, max_translation=0.3):
        super().__init__()
        raise ValueError("TwoBranch is deprecated.")
        self.unet = smp.Unet(
            "resnet18", encoder_weights="imagenet", in_channels=1, classes=3
        )
        # keep it simple - linear layer to each layer of the UNET encoder
        self.projector = torch.nn.LazyConv2d(64, kernel_size=1)
        self.output_angle_layer = torch.nn.Linear(64, 2)
        self.output_translation_layer = torch.nn.Linear(64, 2)
        self.max_translation = max_translation
        # angle = torch.nn.Parameter(torch.tensor([[0.0]]))
        # self.register_parameter("angle", angle)

    def forward(self, x):
        raise ValueError("TwoBranch is deprecated.")
        encoded = self.unet.encoder(x)
        bottleneck = encoded[-1]
        decoded = self.unet.segmentation_head(self.unet.decoder(encoded))
        projected = self.projector(bottleneck).mean(
            dim=(2, 3)
        )  # Global average pooling
        angle_vec = self.output_angle_layer(projected)
        translation_vec = self.output_translation_layer(projected)
        angle = torch.atan2(angle_vec[:, 0], angle_vec[:, 1]).view(-1, 1)
        translation = torch.tanh(translation_vec) * self.max_translation
        return decoded, angle, translation


class ProjectWithTemplateA(torch.nn.Module):
    """
    Encode->Decode segmentations, then pass the segmentation map and template into registration network
    Registration network processes both, return the transformation parameters
    Transformation parameters are applied to spatial transform 
    
    ===
    Return: segmentation map, angle, translation
    ===
    """
    def __init__(self, max_translation=0.3):
        super().__init__()
        self.unet = smp.Unet(
            "resnet18", encoder_weights="imagenet", in_channels=1, classes=3
        )
        self.encoder = timm.create_model(
            'resnet34',  
            pretrained=True,
            in_chans=6,  # concatenated channel count
            num_classes=0,  # remove classification head, gives pooled features
            global_pool='avg',
        )
        self.TransformHead = RigidTransformHead(max_translation=max_translation)

    def forward(self, x,template) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        segmentation_logits = self.unet(x) #B,C,H,W
        concatenated_input = torch.cat([segmentation_logits, template], dim=1)  # B,2C,H,W
        features = self.encoder(concatenated_input)  # B, feature_dim
        angle, translation = self.TransformHead(features)  # B,1 and B,2
        return segmentation_logits, angle, translation
