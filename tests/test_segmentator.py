from unittest.mock import patch

from constraints.models.segmentator import (
    get_segmentator,
    set_segmentator_encoder_weights,
)


def test_get_segmentator_uses_imagenet_weights_by_default():
    set_segmentator_encoder_weights("imagenet")
    with patch("constraints.models.segmentator.smp.Unet") as unet:
        get_segmentator()

    assert unet.call_args.args == ("resnet18",)
    assert unet.call_args.kwargs["encoder_weights"] == "imagenet"


def test_get_segmentator_allows_untrained_encoder():
    try:
        set_segmentator_encoder_weights(None)
        with patch("constraints.models.segmentator.smp.Unet") as unet:
            get_segmentator()

        assert unet.call_args.kwargs["encoder_weights"] is None
    finally:
        set_segmentator_encoder_weights("imagenet")
