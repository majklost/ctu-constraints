"""Reusable metric compositions for Lightning architectures."""

from ..computers.metric_computers import StagedMetricComputer
from ..computers.metric_terms import (
    ACDCRegistrationConstraintViolationTerm,
    ACDCSegmentationConstraintViolationTerm,
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
    max_ignored_enclosed_background_area: int,
) -> list:
    return [
        *_iou_metrics(label_schema),
        DeformationJacobianTerm(label_schema),
        SegmentationConstraintViolationTerm(
            label_schema,
            blob_threshold=blob_threshold,
            check_wall_integrity=check_wall_integrity,
            max_ignored_enclosed_background_area=(max_ignored_enclosed_background_area),
        ),
        RegistrationConstraintViolationTerm(
            label_schema,
            blob_threshold=blob_threshold,
            check_wall_integrity=check_wall_integrity,
            max_ignored_enclosed_background_area=(max_ignored_enclosed_background_area),
        ),
    ]


def create_default_staged_metrics(
    label_schema: LabelSchema,
    *,
    blob_threshold: int = 50,
    check_wall_integrity: bool = True,
    max_ignored_enclosed_background_area: int = 2,
) -> StagedMetricComputer:
    """Build the former default metrics using independent state per stage.

    IoU is emitted for every supported stage. Constraint rates are emitted for
    validation-style stages, matching the former validation-only behavior.
    """
    return StagedMetricComputer(
        {
            "train": CompositeMetric(_iou_metrics(label_schema)),
            "val": CompositeMetric(
                _validation_metrics(
                    label_schema,
                    blob_threshold,
                    check_wall_integrity,
                    max_ignored_enclosed_background_area,
                )
            ),
            "val_extra": CompositeMetric(
                _validation_metrics(
                    label_schema,
                    blob_threshold,
                    check_wall_integrity,
                    max_ignored_enclosed_background_area,
                )
            ),
            "test": CompositeMetric(
                _validation_metrics(
                    label_schema,
                    blob_threshold,
                    check_wall_integrity,
                    max_ignored_enclosed_background_area,
                )
            ),
        }
    )


def create_segmentation_staged_metrics(
    label_schema: LabelSchema,
    *,
    min_hole_area: int = 10,
    min_component_area: int | None = 5,
) -> StagedMetricComputer:
    """Build ACDC segmentation and annularity metrics with independent state."""
    validation_stages = ("val", "val_extra", "test")
    return StagedMetricComputer(
        {
            "train": CompositeMetric([SegmentationIoUTerm(label_schema)]),
            **{
                stage: CompositeMetric(
                    [
                        SegmentationIoUTerm(label_schema),
                        ACDCSegmentationConstraintViolationTerm(
                            label_schema,
                            min_hole_area=min_hole_area,
                            min_component_area=min_component_area,
                        ),
                        ACDCRegistrationConstraintViolationTerm(
                            label_schema,
                            min_hole_area=min_hole_area,
                            min_component_area=min_component_area,
                        ),
                    ]
                )
                for stage in validation_stages
            },
        }
    )
