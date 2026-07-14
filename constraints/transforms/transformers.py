"""
Spatial transfomers and its helper types
wraps the transoformers so it can be simply plugged into lightning module
"""

from abc import ABC, abstractmethod
from .transforms import differentiable_rigid, field_application
from ..types import TransformSpec, WarpResult
import torch

class Transformer(ABC):
    @abstractmethod
    def forward(self, template:torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        pass


class RigidTransformer(Transformer):
    def forward(self, template: torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        assert transform_spec.kind == "rigid", "RigidTransformer only supports rigid transforms"
        assert transform_spec.angle is not None and transform_spec.translation is not None,


class DeformableTransformer(Transformer):
    def forward(self, template: torch.Tensor, transform_spec: TransformSpec) -> WarpResult:
        assert transform_spec.kind == "field", "DeformableTransformer only supports field transforms"
        assert transform_spec.field is not None, "Field must be provided for deformable transform"
        warped_template = field_application(template, transform_spec.field)
        return WarpResult(warped_source=warped_template, field=transform_spec.field, transform=transform_spec)
