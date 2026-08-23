"""Named, configurable loss-objective presets."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

from ..computers.loss_computers import CompositeLossComputer
from ..computers.loss_terms import (
    DeformationGradientTerm,
    LossTerm,
    RegistrationBlurredMSETerm,
    RegistrationCentroidTerm,
    RegistrationCrossEntropyTerm,
    RegistrationDSDFMSETerm,
    RegistrationMSE_SDFTEMPLATETerm,
    RegistrationOneside_SDFTEMPLATETerm,
    RegistrationOneSideSDFSquareTerm,
    RegistrationOneSideSDFTerm,
    SegmentationCrossEntropyTerm,
    SegmentationOneSideSDFSquareTerm,
    SegmentationOneSideSDFTerm,
)
from ..datatools.label_schema import LabelSchema
from ..types import WeightedLossTerm

LossPresetName = Literal[
    "bce_one_side_sdf_squared",
    "bce_one_side_sdf_plain",
    "bce_bce",
    "bce_centroid",
    "bce_blurred_mse",
    "bce_dsdf_mse",
    "bce_sdf_template_mse",
    "bce_sdf_template_one_side_sdf_squared",
    "one_side_sdf_squared_one_side_sdf_squared",
    "one_side_sdf_plain_one_side_sdf_plain",
]


@dataclass(frozen=True)
class _LossPreset:
    segmentation_term: Callable[[LabelSchema], LossTerm]
    registration_term: Callable[[LabelSchema], LossTerm]
    default_segmentation_weight: float
    default_registration_weight: float


_PRESETS: dict[LossPresetName, _LossPreset] = {
    "bce_one_side_sdf_squared": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationOneSideSDFSquareTerm, 20.0, 1.0
    ),
    "bce_one_side_sdf_plain": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationOneSideSDFTerm, 20.0, 1.0
    ),
    "bce_bce": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationCrossEntropyTerm, 1.0, 1.0
    ),
    "bce_centroid": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationCentroidTerm, 1.0, 1.0
    ),
    "bce_blurred_mse": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationBlurredMSETerm, 1.0, 1.0
    ),
    "bce_dsdf_mse": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationDSDFMSETerm, 1.0, 1e-3
    ),
    "bce_sdf_template_mse": _LossPreset(
        SegmentationCrossEntropyTerm, RegistrationMSE_SDFTEMPLATETerm, 1.0, 1.0
    ),
    "bce_sdf_template_one_side_sdf_squared": _LossPreset(
        SegmentationCrossEntropyTerm,
        RegistrationOneside_SDFTEMPLATETerm,
        1.0,
        1.0,
    ),
    "one_side_sdf_squared_one_side_sdf_squared": _LossPreset(
        SegmentationOneSideSDFSquareTerm, RegistrationOneSideSDFSquareTerm, 1.0, 1.0
    ),
    "one_side_sdf_plain_one_side_sdf_plain": _LossPreset(
        SegmentationOneSideSDFTerm, RegistrationOneSideSDFTerm, 1.0, 1.0
    ),
}


def available_loss_presets() -> tuple[LossPresetName, ...]:
    return tuple(_PRESETS)


def create_loss_computer(
    preset: LossPresetName,
    label_schema: LabelSchema,
    *,
    segmentation_weight: float | None = None,
    registration_weight: float | None = None,
    field_regularization_weight: float = 0.0,
    extra_terms: Sequence[WeightedLossTerm] = (),
    grad_diagnostics: bool = False,
) -> CompositeLossComputer:
    """Create a named objective with resolved, explicit weighted terms."""
    if field_regularization_weight < 0:
        raise ValueError("field_regularization_weight must be non-negative")
    definition = _PRESETS[preset]
    terms = [
        WeightedLossTerm(
            definition.default_segmentation_weight
            if segmentation_weight is None
            else segmentation_weight,
            definition.segmentation_term(label_schema),
        ),
        WeightedLossTerm(
            definition.default_registration_weight
            if registration_weight is None
            else registration_weight,
            definition.registration_term(label_schema),
        ),
    ]
    if field_regularization_weight:
        terms.append(
            WeightedLossTerm(
                field_regularization_weight, DeformationGradientTerm(label_schema)
            )
        )
    return CompositeLossComputer(
        terms=[*terms, *extra_terms],
        grad_diagnostics=grad_diagnostics,
        preset_name=preset,
    )
