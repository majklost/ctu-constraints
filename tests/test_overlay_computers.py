import torch
import torch.nn.functional as functional

from constraints.computers.overlay_computers import SegmentationOverlayComputer
from constraints.datatools.label_schema import LabelSchema
from constraints.types import (
    DiscreteSegmentation,
    MetricInput,
    OverlayPolicy,
    StepContext,
)

SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1)],
)


def _channels(labels: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(labels, SCHEMA.num_classes).movedim(-1, 1).float()


def test_segmentation_overlay_uses_stable_ids_and_generic_payloads() -> None:
    ground_truth = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[2, 1], [0, 2]],
        ]
    )
    predicted = torch.tensor(
        [
            [[0, 2], [2, 0]],
            [[1, 1], [0, 2]],
        ]
    )
    overlay_computer = SegmentationOverlayComputer(
        SCHEMA,
        OverlayPolicy(
            stages=frozenset({"val"}),
            every_n_epochs=2,
            sample_ids=("second", "first"),
        ),
    )
    metric_input = MetricInput(
        image=torch.tensor(
            [
                [[[0.0, 2.0], [1.0, 3.0]]],
                [[[3.0, 1.0], [2.0, 0.0]]],
            ]
        ),
        segmentation_logits=_channels(predicted),
        warped_template=_channels(ground_truth),
        gt=DiscreteSegmentation(ground_truth, SCHEMA),
        sample_ids=("first", "second"),
    )

    overlays = overlay_computer.compute(
        metric_input,
        StepContext(stage="val", batch_idx=4, current_epoch=2, global_step=10),
    )

    assert list(overlays) == ["labels/second", "labels/first"]
    second = overlays["labels/second"]
    assert set(second.masks) == {"ground_truth", "warped", "predicted"}
    assert torch.equal(second.masks["predicted"], predicted[1])
    assert second.image.min() == 0
    assert second.image.max() == 1
    assert second.class_labels == dict(SCHEMA.names)
    assert "sample=second" in (second.caption or "")


def test_segmentation_overlay_skips_unscheduled_context_or_absent_ids() -> None:
    overlay_computer = SegmentationOverlayComputer(
        SCHEMA,
        OverlayPolicy(
            stages=frozenset({"val"}),
            every_n_epochs=2,
            sample_ids=("wanted",),
        ),
    )
    metric_input = MetricInput(
        image=torch.zeros((1, 1, 2, 2)),
        sample_ids=("other",),
    )

    assert overlay_computer.compute(
        metric_input,
        StepContext(stage="train", batch_idx=0, current_epoch=2, global_step=0),
    ) == {}
    assert overlay_computer.compute(
        metric_input,
        StepContext(stage="val", batch_idx=0, current_epoch=2, global_step=0),
    ) == {}


def test_segmentation_overlay_uses_first_samples_from_first_epoch_batch() -> None:
    overlay_computer = SegmentationOverlayComputer(
        SCHEMA,
        OverlayPolicy(
            stages=frozenset({"val"}),
            every_n_epochs=1,
            first_n_samples=2,
        ),
    )
    metric_input = MetricInput(
        image=torch.zeros((3, 1, 2, 2)),
        segmentation_logits=_channels(torch.zeros((3, 2, 2), dtype=torch.long)),
        sample_ids=("first", "second", "third"),
    )

    overlays = overlay_computer.compute(
        metric_input,
        StepContext(stage="val", batch_idx=0, current_epoch=1, global_step=0),
    )

    assert list(overlays) == ["labels/first", "labels/second"]
    assert overlay_computer.compute(
        metric_input,
        StepContext(stage="val", batch_idx=1, current_epoch=1, global_step=1),
    ) == {}
