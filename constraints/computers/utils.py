from collections.abc import Iterator

import torch

from ..types import TransformSpec


def iter_deformation_fields(
    transform_spec: TransformSpec | None,
) -> Iterator[torch.Tensor]:
    """Yield atomic VoxelMorph displacement fields from a transform spec."""
    if transform_spec is None:
        return
    if transform_spec.field is not None:
        yield transform_spec.field.field
    if transform_spec.steps is not None:
        for step in transform_spec.steps:
            yield from iter_deformation_fields(step)
