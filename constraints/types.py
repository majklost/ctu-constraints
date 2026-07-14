from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class TransformSpec:
    kind: Literal["rigid", "field"]
    angle: torch.Tensor | None = None
    translation: torch.Tensor | None = None
    field: torch.Tensor | None = None
    meta: dict | None = None


@dataclass
class WarpResult:
    warped_source: torch.Tensor | None = None
    warped_target: torch.Tensor | None = None
    field: torch.Tensor | None = None
    transform: TransformSpec | None = None
