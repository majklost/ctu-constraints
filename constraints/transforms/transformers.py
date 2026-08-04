"""
Spatial transfomers and its helper types
wraps the transoformers so it can be simply plugged into lightning module
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..types import TransformSpec, WarpResult
from .transforms import differentiable_rigid, field_application


class SpatialTransformer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        pass


class RigidTransformer(SpatialTransformer):
    def __init__(self):
        super().__init__()

    def forward(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        assert transform_spec.rigid is not None, (
            "Rigid parameters must be provided for rigid transform"
        )
        assert transform_spec.field is None, (
            "RigidTransformer does not support field transforms"
        )
        rigid = transform_spec.rigid
        warped_template = differentiable_rigid(
            template, rigid.angle, dx=rigid.dx, dy=rigid.dy
        )
        return WarpResult(
            warped_template=warped_template, transform_spec=transform_spec
        )


class DeformableTransformer(SpatialTransformer):
    def __init__(self):
        super().__init__()

    def forward(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        assert transform_spec.field is not None, (
            "Field parameters must be provided for deformable transform"
        )
        assert transform_spec.rigid is None, (
            "DeformableTransformer does not support rigid transforms"
        )
        field_result = field_application(
            template,
            transform_spec.field.field,
            return_warped_source=True,
        )
        assert field_result.warped_source is not None
        warped_template = field_result.warped_source
        return WarpResult(
            warped_template=warped_template, transform_spec=transform_spec
        )


class SequentialTransformer(SpatialTransformer):
    """Apply an atomic transform or each transform in an ordered sequence."""

    def __init__(self):
        super().__init__()
        self.rigid_transformer = RigidTransformer()
        self.deformable_transformer = DeformableTransformer()

    def forward(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        if transform_spec.steps is None:
            return self._apply_atomic(template, transform_spec)

        if transform_spec.rigid is not None or transform_spec.field is not None:
            raise ValueError(
                "A sequential transform spec cannot also contain a transform"
            )
        if not transform_spec.steps:
            raise ValueError(
                "A sequential transform spec must contain at least one step"
            )

        warped_template = template
        for step in transform_spec.steps:
            warped_template = self._apply_atomic(warped_template, step).warped_template

        return WarpResult(
            warped_template=warped_template, transform_spec=transform_spec
        )

    def _apply_atomic(
        self, template: torch.Tensor, transform_spec: TransformSpec
    ) -> WarpResult:
        if transform_spec.steps is not None:
            raise ValueError("Nested sequential transform specs are not supported")
        if transform_spec.rigid is not None and transform_spec.field is not None:
            raise ValueError(
                "An atomic transform spec cannot contain rigid and field values"
            )
        if transform_spec.rigid is not None:
            return self.rigid_transformer(template, transform_spec)
        if transform_spec.field is not None:
            return self.deformable_transformer(template, transform_spec)
        raise ValueError("An atomic transform spec must contain rigid or field values")
