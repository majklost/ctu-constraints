import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from ..datatools.label_schema import LabelSchema

_encoder_weights: str | None = "imagenet"


def set_segmentator_encoder_weights(encoder_weights: str | None) -> None:
    """Set encoder weights used by all segmentators created in this process."""
    global _encoder_weights
    _encoder_weights = encoder_weights


def get_segmentator(num_classes: int):
    """Create a segmentator using the configured encoder weights."""
    return smp.Unet(
        "resnet18",
        encoder_weights=_encoder_weights,
        in_channels=1,
        classes=num_classes,
    )
