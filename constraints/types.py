from dataclasses import dataclass
from typing import Any
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


@dataclass
class LossInput:
    """Canonical model output contract used by loss computers.

    Keep common outputs explicit and route ablation-specific outputs through
    `extras` to avoid changing call signatures during experiments.
    """

    segmentation_logits: torch.Tensor | None = None
    warped_template: torch.Tensor | None = None
    gt_mask: torch.Tensor | None = None
    gt_mask_sdf: torch.Tensor | None = None
    transform_spec: TransformSpec | None = None
    extras: dict[str, torch.Tensor] | None = None



@dataclass
class LossResult:
    """Structured loss output with total scalar for backward()."""

    total: torch.Tensor
    components: dict[str, torch.Tensor] | None = None
    logs: dict[str, float | torch.Tensor] | None = None


