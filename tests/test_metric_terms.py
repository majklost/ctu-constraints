import pytest
import torch
import torch.nn.functional as functional

from constraints.computers.metric_terms import (
    ACDCRegistrationConstraintViolationTerm,
    ACDCSegmentationConstraintViolationTerm,
    CompositeMetric,
    RegistrationConstraintViolationTerm,
    RegistrationIoUTerm,
    SegmentationConstraintViolationTerm,
    SegmentationIoUTerm,
    StatefulMetric,
)
from constraints.datatools.label_schema import LabelSchema
from constraints.types import DiscreteSegmentation, MetricInput, MetricResult

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1)],
)

CONSTRAINT_LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen", "plaque"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.35, 0.95)],
)

ACDC_LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "myocardium"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1)],
)


def _channels(labels: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(labels, LABEL_SCHEMA.num_classes).movedim(-1, 1).float()


def _metric_input(labels: torch.Tensor) -> MetricInput:
    channels = _channels(labels)
    return MetricInput(
        image=torch.zeros((labels.shape[0], 1, *labels.shape[1:])),
        segmentation_logits=channels,
        warped_template=channels,
        gt=DiscreteSegmentation(labels=labels, label_schema=LABEL_SCHEMA),
    )


def _segmentation_metric_input(
    predictions: torch.Tensor, targets: torch.Tensor
) -> MetricInput:
    return MetricInput(
        image=torch.zeros((targets.shape[0], 1, *targets.shape[1:])),
        segmentation_logits=_channels(predictions),
        gt=DiscreteSegmentation(labels=targets, label_schema=LABEL_SCHEMA),
    )


def test_iou_terms_report_dataset_macro_and_per_class_iou() -> None:
    labels = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[2, 1], [0, 2]],
        ]
    )

    expected_names = {
        "segmentation/iou/pred_vs_gt",
        "segmentation/iou/background_vs_gt",
        "segmentation/iou/boundary_vs_gt",
        "segmentation/iou/lumen_vs_gt",
    }
    terms_and_expected_names = (
        (SegmentationIoUTerm(LABEL_SCHEMA), expected_names),
        (
            RegistrationIoUTerm(LABEL_SCHEMA),
            {name.replace("segmentation", "registration").replace(
                "pred", "warped"
            ) for name in expected_names},
        ),
    )

    for term, names in terms_and_expected_names:
        assert term.update(_metric_input(labels)).scalars == {}
        result = term.compute().scalars

        assert set(result) == names
        assert all(torch.isclose(value, torch.tensor(1.0)) for value in result.values())

        term.reset()


def test_iou_accumulates_over_unequal_batches_as_one_dataset() -> None:
    targets = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [2, 0]],
            [[2, 1], [0, 2]],
        ]
    )
    predictions = targets.clone()
    predictions[1:] = 0
    accumulated = SegmentationIoUTerm(LABEL_SCHEMA)
    combined = SegmentationIoUTerm(LABEL_SCHEMA)

    accumulated.update(_segmentation_metric_input(predictions[:1], targets[:1]))
    accumulated.update(_segmentation_metric_input(predictions[1:], targets[1:]))
    combined.update(_segmentation_metric_input(predictions, targets))

    accumulated_result = accumulated.compute().scalars
    combined_result = combined.compute().scalars
    assert accumulated_result.keys() == combined_result.keys()
    for name in accumulated_result:
        assert torch.isclose(accumulated_result[name], combined_result[name])


class _StaticMetric(StatefulMetric):
    def __init__(self, scalars: dict[str, float]) -> None:
        super().__init__()
        self._scalars = scalars

    def update(self, inputs: MetricInput) -> MetricResult:
        del inputs
        return MetricResult(scalars=dict(self._scalars))

    def compute(self) -> MetricResult:
        return MetricResult(scalars=dict(self._scalars))

    def reset(self) -> None:
        pass


def test_composite_metric_merges_empty_results_and_normalizes_prefix() -> None:
    composite = CompositeMetric(
        [
            _StaticMetric({}),
            _StaticMetric({"segmentation/iou": 0.8}),
            _StaticMetric({"registration/iou": 0.7}),
        ],
        prefix="comparison/primary/",
    )

    assert composite.compute().scalars == {
        "comparison/primary/segmentation/iou": 0.8,
        "comparison/primary/registration/iou": 0.7,
    }


def test_composite_metric_rejects_duplicate_scalar_names() -> None:
    composite = CompositeMetric(
        [
            _StaticMetric({"segmentation/iou": 0.8}),
            _StaticMetric({"segmentation/iou": 0.7}),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate metric scalar name 'segmentation/iou'",
    ):
        composite.compute()


def test_composite_metric_rejects_a_slash_only_prefix() -> None:
    with pytest.raises(ValueError, match="must contain non-slash characters"):
        CompositeMetric([], prefix="///")


def _valid_vessel_labels(size: int = 32) -> torch.Tensor:
    labels = torch.zeros((size, size), dtype=torch.long)
    labels[6:26, 6:26] = 1
    labels[9:23, 9:23] = 2
    return labels


def _constraint_metric_input(
    segmentation_labels: torch.Tensor | None,
    registration_labels: torch.Tensor | None,
    sample_ids: tuple[str, ...] = (),
) -> MetricInput:
    batch_size = (
        segmentation_labels.shape[0]
        if segmentation_labels is not None
        else registration_labels.shape[0]
    )
    return MetricInput(
        image=torch.zeros((batch_size, 1, 32, 32)),
        segmentation_logits=(
            functional.one_hot(
                segmentation_labels, CONSTRAINT_LABEL_SCHEMA.num_classes
            ).movedim(-1, 1).float()
            if segmentation_labels is not None
            else None
        ),
        warped_template=(
            functional.one_hot(
                registration_labels, CONSTRAINT_LABEL_SCHEMA.num_classes
            ).movedim(-1, 1).float()
            if registration_labels is not None
            else None
        ),
        sample_ids=sample_ids,
    )


def test_constraint_violation_terms_accumulate_and_reset_independently() -> None:
    valid = _valid_vessel_labels()
    invalid = torch.zeros_like(valid)
    segmentation = SegmentationConstraintViolationTerm(CONSTRAINT_LABEL_SCHEMA)
    registration = RegistrationConstraintViolationTerm(CONSTRAINT_LABEL_SCHEMA)

    first_batch = _constraint_metric_input(
        torch.stack((valid, invalid)), torch.stack((valid, valid))
    )
    second_batch = _constraint_metric_input(torch.stack((valid,)), None)
    segmentation.update(first_batch)
    registration.update(first_batch)
    segmentation.update(second_batch)
    registration.update(second_batch)

    assert torch.isclose(
        segmentation.compute().scalars["segmentation/constraint/violation_rate"],
        torch.tensor(1 / 3),
    )
    assert torch.isclose(
        registration.compute().scalars["registration/constraint/violation_rate"],
        torch.tensor(0.0),
    )

    segmentation.reset()
    registration.reset()
    assert segmentation.compute().scalars == {}
    assert registration.compute().scalars == {}


def test_constraint_violation_term_can_emit_violating_batch_samples() -> None:
    valid = _valid_vessel_labels()
    invalid = torch.zeros_like(valid)
    term = SegmentationConstraintViolationTerm(
        CONSTRAINT_LABEL_SCHEMA,
        track_violating_samples=True,
    )

    result = term.update(
        _constraint_metric_input(
            torch.stack((valid, invalid)),
            None,
            sample_ids=("valid-sample", "invalid-sample"),
        )
    )

    samples = result.constraint_violation_samples["segmentation/constraint"]
    assert samples.sample_ids == ("invalid-sample",)
    assert samples.details
    assert samples.details[0]


def test_constraint_sample_tracking_requires_stable_sample_ids() -> None:
    term = SegmentationConstraintViolationTerm(
        CONSTRAINT_LABEL_SCHEMA,
        track_violating_samples=True,
    )

    with pytest.raises(ValueError, match="requires MetricInput.sample_ids"):
        term.update(
            _constraint_metric_input(torch.stack((_valid_vessel_labels(),)), None)
        )


def _annular_myocardium_labels() -> torch.Tensor:
    labels = torch.zeros((32, 32), dtype=torch.long)
    labels[4:28, 4:28] = 1
    labels[10:22, 10:22] = 0
    return labels


def _acdc_metric_input(
    segmentation_labels: torch.Tensor | None,
    registration_labels: torch.Tensor | None,
    sample_ids: tuple[str, ...] = (),
) -> MetricInput:
    labels = (
        segmentation_labels
        if segmentation_labels is not None
        else registration_labels
    )
    assert labels is not None

    def channels(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return functional.one_hot(value, 2).movedim(-1, 1).float()

    return MetricInput(
        image=torch.zeros((labels.shape[0], 1, 32, 32)),
        segmentation_logits=channels(segmentation_labels),
        warped_template=channels(registration_labels),
        sample_ids=sample_ids,
    )


def test_acdc_annularity_terms_accumulate_and_reset_independently() -> None:
    annular = _annular_myocardium_labels()
    non_annular = annular.clone()
    non_annular[10:22, 10:22] = 1
    segmentation = ACDCSegmentationConstraintViolationTerm(ACDC_LABEL_SCHEMA)
    registration = ACDCRegistrationConstraintViolationTerm(ACDC_LABEL_SCHEMA)
    metric_input = _acdc_metric_input(
        torch.stack((annular, non_annular)),
        torch.stack((annular, annular)),
    )

    segmentation.update(metric_input)
    registration.update(metric_input)

    assert torch.isclose(
        segmentation.compute().scalars["segmentation/constraint/violation_rate"],
        torch.tensor(0.5),
    )
    assert torch.isclose(
        registration.compute().scalars["registration/constraint/violation_rate"],
        torch.tensor(0.0),
    )

    segmentation.reset()
    registration.reset()
    assert segmentation.compute().scalars == {}
    assert registration.compute().scalars == {}


def test_acdc_annularity_term_forwards_specific_violation_details() -> None:
    empty = torch.zeros((1, 32, 32), dtype=torch.long)
    term = ACDCSegmentationConstraintViolationTerm(
        ACDC_LABEL_SCHEMA,
        track_violating_samples=True,
    )

    result = term.update(
        _acdc_metric_input(empty, None, sample_ids=("empty-myocardium",))
    )

    samples = result.constraint_violation_samples["segmentation/constraint"]
    assert samples.sample_ids == ("empty-myocardium",)
    assert samples.details == (("Myocardium mask is empty.",),)
