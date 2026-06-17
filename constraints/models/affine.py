import segmentation_models_pytorch as smp
import torch


class TwoBranch(torch.nn.Module):
    def __init__(self, max_translation=0.3):
        super().__init__()
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
