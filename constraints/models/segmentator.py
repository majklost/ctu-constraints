import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from ..datatools.datasets import ARTIFICIAL_MASK_NUM_CLASSES


def get_learned_segmentator():
    return smp.Unet(
            "resnet18", encoder_weights="imagenet", in_channels=1, classes=ARTIFICIAL_MASK_NUM_CLASSES
        )
