from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class RigidParams:
    angle: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor

@dataclass
class FieldParams:
    field: torch.Tensor


@dataclass
class FieldApplicationResult:
    field: torch.Tensor | None = None
    warped_source: torch.Tensor | None = None
    warped_target: torch.Tensor | None = None



@dataclass
class TransformSpec:
    rigid: RigidParams | None = None
    field: FieldParams | None = None
    meta: dict | None = None


@dataclass
class WarpResult:
    warped_template: torch.Tensor
    transform_spec: TransformSpec
    warped_mask: torch.Tensor | None = None


