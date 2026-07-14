"""
Spatial transfomers and its helper types
wraps the transoformers so it can be simply plugged into lightning module
"""

from abc import ABC, abstractmethod
from .transforms import differentiable_rigid, field_application
from ..types import TransformSpec, WarpResult
import torch

class SpatialTransformer(ABC):
    @abstractmethod
    def forward(self, template:torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        pass


class RigidTransformer(SpatialTransformer):
    def forward(self, template: torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        assert transform_spec.rigid is not None, "Rigid parameters must be provided for rigid transform"
        assert transform_spec.field is None, "RigidTransformer does not support field transforms"
        rigid = transform_spec.rigid
        warped_template = differentiable_rigid(template, rigid.angle, dx=rigid.dx, dy=rigid.dy)
        return WarpResult(warped_template=warped_template, transform_spec=transform_spec)

class DeformableTransformer(SpatialTransformer):
    def forward(self, template: torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        assert transform_spec.field is not None, "Field parameters must be provided for deformable transform"
        assert transform_spec.rigid is None, "DeformableTransformer does not support rigid transforms"
        field_result = field_application(
            template,
            transform_spec.field.field,
            return_warped_source=True,
        )
        assert field_result.warped_source is not None
        warped_template = field_result.warped_source
        return WarpResult(warped_template=warped_template, transform_spec=transform_spec)
