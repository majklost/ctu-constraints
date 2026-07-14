import torch

from ..types import FieldApplicationResult
from ..voxelmorph import modules


def _resolve_broadcast_batch_size(named_sizes):
    batch_size = max(named_sizes.values())
    incompatible = [
        f"{name}={size}"
        for name, size in named_sizes.items()
        if size not in (1, batch_size)
    ]
    if incompatible:
        raise ValueError(
            "Incompatible batch sizes for broadcasting: "
            + ", ".join(incompatible)
            + f" (resolved batch size: {batch_size})"
        )
    return batch_size


def _as_batch_parameter(value, batch_size, image, name):
    value = torch.as_tensor(value, device=image.device, dtype=image.dtype).reshape(-1)
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(
            f"Expected {name} to be scalar or have one value per image in the batch, "
            f"got {value.numel()} values for batch size {batch_size}"
        )
    return value


def differentiable_rigid(image, angle, dx, dy):
    """
    Compute a differentiable rigid transformation (rotation + translation) of an image.

    Image and parameter batches are broadcastable: each input batch dimension can
    be either 1 or N.

    """
    input_was_unbatched = image.ndim == 3
    if input_was_unbatched:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"Expected image shape (N, C, H, W) or (C, H, W), got {tuple(image.shape)}"
        )

    angle = torch.as_tensor(angle, device=image.device, dtype=image.dtype).reshape(-1)
    dx = torch.as_tensor(dx, device=image.device, dtype=image.dtype).reshape(-1)
    dy = torch.as_tensor(dy, device=image.device, dtype=image.dtype).reshape(-1)

    batch_size = _resolve_broadcast_batch_size(
        {
            "image": image.shape[0],
            "angle": angle.numel(),
            "dx": dx.numel(),
            "dy": dy.numel(),
        }
    )

    if image.shape[0] == 1 and batch_size > 1:
        image = image.expand(batch_size, -1, -1, -1)

    angle = _as_batch_parameter(angle, batch_size, image, "angle")
    dx = _as_batch_parameter(dx, batch_size, image, "dx")
    dy = _as_batch_parameter(dy, batch_size, image, "dy")

    # Build theta from tensors so gradients can flow back to parameters.
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    theta = torch.stack(
        [
            cos,
            -sin,
            dx,
            sin,
            cos,
            dy,
        ],
        dim=1,
    ).reshape(-1, 2, 3)  # Shape: (N, 2, 3)

    # print("theta_shape:", theta.shape)
    # print("image_shape:", image.shape)

    grid = torch.nn.functional.affine_grid(theta, image.size(), align_corners=False)
    rotated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
    return rotated_image.squeeze(0) if input_was_unbatched and batch_size == 1 else rotated_image


def differentiable_rotation(image, angle):
    """Rotate a batch of images by one angle per image.

    Args:
        image: Tensor with shape (N, C, H, W), or a single image (C, H, W).
        angle: 1D tensor with shape (N,) containing angles in radians.
            Batch size is broadcasted between image and angle (1 or N).
    """
    return differentiable_rigid(image, angle, dx=0.0, dy=0.0)


def field_application(
    source,
    displacement,
    target=None,
    *,
    return_field=False,
    return_warped_source=True,
    return_warped_target=False,
):
    """
    Apply a deformation field to an image using differentiable sampling.

    Args:
        source: Tensor with shape (N, C, H, W), or a single image (C, H, W).
        displacement: Tensor with shape (N, 2, H, W) containing displacement vectors.
            The source/displacement batch dimensions are broadcastable: each can
            be either 1 or N.
        target: Optional tensor with shape (N, C, H, W) or (C, H, W).
        return_field: If True, include displacement in the output.
        return_warped_source: If True, include source warped by +displacement.
        return_warped_target: If True, include target warped by -displacement.

    Returns:
        FieldApplicationResult with requested members populated.
    """
    return _field_application_impl(
        source,
        displacement,
        target=target,
        return_field=return_field,
        return_warped_source=return_warped_source,
        return_warped_target=return_warped_target,
    )


def _as_batched_image(image, name):
    input_was_unbatched = image.ndim == 3
    if input_was_unbatched:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"Expected {name} shape (N, C, H, W) or (C, H, W), got {tuple(image.shape)}"
        )
    return image, input_was_unbatched


def _as_batched_field(field, spatial_shape, reference):
    input_was_unbatched = field.ndim == 3
    if input_was_unbatched:
        field = field.unsqueeze(0)
    if field.ndim != 4:
        raise ValueError(
            f"Expected field shape (N, 2, H, W) or (2, H, W), got {tuple(field.shape)}"
        )
    if field.shape[1] != 2:
        raise ValueError(
            f"Expected field to have 2 channels (dx, dy), got {field.shape[1]}"
        )
    if tuple(field.shape[2:]) != tuple(spatial_shape):
        raise ValueError(
            f"Field spatial shape {tuple(field.shape[2:])} does not match image "
            f"spatial shape {tuple(spatial_shape)}"
        )

    return field.to(device=reference.device, dtype=reference.dtype), input_was_unbatched


def _field_application_impl(
    source,
    displacement,
    target=None,
    *,
    return_field=False,
    return_warped_source=True,
    return_warped_target=False,
):
    """Apply a displacement field to source/target images.

    This mirrors VoxelMorph conventions:
    - positive displacement warps source -> target
    - negative displacement warps target -> source

    Args:
        source: Tensor with shape (N, C, H, W) or (C, H, W).
        displacement: Tensor with shape (N, 2, H, W) or (2, H, W).
            source/displacement batch sizes are broadcastable (1 or N).
        target: Optional tensor with shape (N, C, H, W) or (C, H, W).
        return_field: If True, include displacement as first returned item.
        return_warped_source: If True, include source warped by +displacement.
        return_warped_target: If True, include target warped by -displacement.

    Returns:
        FieldApplicationResult with requested members populated.
    """
    if not return_field and not return_warped_source and not return_warped_target:
        raise ValueError("At least one of return_field/return_warped_source/return_warped_target must be True")

    source_batched, source_was_unbatched = _as_batched_image(source, "source")
    field_batched, field_was_unbatched = _as_batched_field(
        displacement,
        spatial_shape=source_batched.shape[2:],
        reference=source_batched,
    )

    batch_size = _resolve_broadcast_batch_size(
        {
            "source": source_batched.shape[0],
            "displacement": field_batched.shape[0],
        }
    )

    if source_batched.shape[0] == 1 and batch_size > 1:
        source_batched = source_batched.expand(batch_size, -1, -1, -1)
    if field_batched.shape[0] == 1 and batch_size > 1:
        field_batched = field_batched.expand(batch_size, -1, -1, -1)

    target_batched = None
    target_was_unbatched = False
    if return_warped_target:
        if target is None:
            raise ValueError("target must be provided when return_warped_target=True")
        target_batched, target_was_unbatched = _as_batched_image(target, "target")
        if target_batched.shape[0] == 1 and batch_size > 1:
            target_batched = target_batched.expand(batch_size, -1, -1, -1)
        elif target_batched.shape[0] != batch_size:
            raise ValueError(
                f"Target batch size {target_batched.shape[0]} must be 1 or match resolved batch size {batch_size}"
            )
        if tuple(target_batched.shape[2:]) != tuple(source_batched.shape[2:]):
            raise ValueError(
                f"Target spatial shape {tuple(target_batched.shape[2:])} must match source spatial shape {tuple(source_batched.shape[2:])}"
            )
        target_batched = target_batched.to(
            device=source_batched.device, dtype=source_batched.dtype
        )

    transformer = modules.SpatialTransformer()

    result = FieldApplicationResult()

    if return_field:
        field_out = field_batched
        if source_was_unbatched and field_was_unbatched and batch_size == 1:
            field_out = field_out.squeeze(0)
        result.field = field_out

    if return_warped_source:
        warped_source = transformer(source_batched, field_batched)
        if source_was_unbatched and batch_size == 1:
            warped_source = warped_source.squeeze(0)
        result.warped_source = warped_source

    if return_warped_target:
        warped_target = transformer(target_batched, -field_batched)
        if target_was_unbatched and batch_size == 1:
            warped_target = warped_target.squeeze(0)
        result.warped_target = warped_target

    return result


