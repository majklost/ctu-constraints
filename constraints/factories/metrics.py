"""Reusable metric compositions for Lightning architectures."""

from ..computers.metric_computers import StagedMetricComputer
from ..computers.metric_terms import (
    CompositeMetric,
    DeformationJacobianTerm,
    RegistrationConstraintViolationTerm,
    RegistrationIoUTerm,
    SegmentationConstraintViolationTerm,
    SegmentationIoUTerm,
)
from ..datatools.label_schema import LabelSchema


def _iou_metrics(label_schema: LabelSchema) -> list:
    """Create fresh terms; stateful metrics must never be shared by stage."""
    return [SegmentationIoUTerm(label_schema), RegistrationIoUTerm(label_schema)]


def _validation_metrics(
    label_schema: LabelSchema,
    blob_threshold: int,
    check_wall_integrity: bool,
) -> list:
    return [
        *_iou_metrics(label_schema),
        DeformationJacobianTerm(label_schema),
        SegmentationConstraintViolationTerm(
            label_schema,
            blob_threshold=blob_threshold,
            check_wall_integrity=check_wall_integrity,
        ),
        RegistrationConstraintViolationTerm(
            label_schema,
            blob_threshold=blob_threshold,
            check_wall_integrity=check_wall_integrity,
        ),
    ]


def create_default_staged_metrics(
    label_schema: LabelSchema,
    *,
    blob_threshold: int = 50,
    check_wall_integrity: bool = True,
) -> StagedMetricComputer:
    """Build the former default metrics using independent state per stage.

    IoU is emitted for every supported stage. Constraint rates are emitted for
    validation-style stages, matching the former validation-only behavior.
    """
    return StagedMetricComputer(
        {
            "train": CompositeMetric(_iou_metrics(label_schema)),
            "val": CompositeMetric(
                _validation_metrics(label_schema, blob_threshold, check_wall_integrity)
            ),
            "val_extra": CompositeMetric(
                _validation_metrics(label_schema, blob_threshold, check_wall_integrity)
            ),
            "test": CompositeMetric(
                _validation_metrics(label_schema, blob_threshold, check_wall_integrity)
            ),
        }
    )
