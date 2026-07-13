import torch

from .voxelmorph import modules


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

    """
    input_was_unbatched = image.ndim == 3
    if input_was_unbatched:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"Expected image shape (N, C, H, W) or (C, H, W), got {tuple(image.shape)}"
        )

    batch_size = image.shape[0]
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
    return rotated_image.squeeze(0) if input_was_unbatched else rotated_image


def differentiable_rotation(image, angle):
    """Rotate a batch of images by one angle per image.

    Args:
        image: Tensor with shape (N, C, H, W), or a single image (C, H, W).
        angle: 1D tensor with shape (N,) containing angles in radians.
            A scalar angle is accepted for a single image.
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
        target: Optional tensor with shape (N, C, H, W) or (C, H, W).
        return_field: If True, include displacement in the output.
        return_warped_source: If True, include source warped by +displacement.
        return_warped_target: If True, include target warped by -displacement.
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


def _as_batched_field(field, batch_size, spatial_shape, reference):
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

    if field.shape[0] == 1 and batch_size > 1:
        field = field.expand(batch_size, -1, -1, -1)
    elif field.shape[0] != batch_size:
        raise ValueError(
            f"Field batch size {field.shape[0]} must match image batch size {batch_size}"
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
        target: Optional tensor with shape (N, C, H, W) or (C, H, W).
        return_field: If True, include displacement as first returned item.
        return_warped_source: If True, include source warped by +displacement.
        return_warped_target: If True, include target warped by -displacement.

    Returns:
        A tensor or tuple of tensors, depending on selected return flags.
    """
    if not return_field and not return_warped_source and not return_warped_target:
        raise ValueError("At least one of return_field/return_warped_source/return_warped_target must be True")

    source_batched, source_was_unbatched = _as_batched_image(source, "source")
    field_batched, field_was_unbatched = _as_batched_field(
        displacement,
        batch_size=source_batched.shape[0],
        spatial_shape=source_batched.shape[2:],
        reference=source_batched,
    )

    target_batched = None
    target_was_unbatched = False
    if return_warped_target:
        if target is None:
            raise ValueError("target must be provided when return_warped_target=True")
        target_batched, target_was_unbatched = _as_batched_image(target, "target")
        if target_batched.shape[0] == 1 and source_batched.shape[0] > 1:
            target_batched = target_batched.expand(source_batched.shape[0], -1, -1, -1)
        elif target_batched.shape[0] != source_batched.shape[0]:
            raise ValueError(
                f"Target batch size {target_batched.shape[0]} must match source batch size {source_batched.shape[0]}"
            )
        if tuple(target_batched.shape[2:]) != tuple(source_batched.shape[2:]):
            raise ValueError(
                f"Target spatial shape {tuple(target_batched.shape[2:])} must match source spatial shape {tuple(source_batched.shape[2:])}"
            )
        target_batched = target_batched.to(
            device=source_batched.device, dtype=source_batched.dtype
        )

    transformer = modules.SpatialTransformer()

    outputs = []

    if return_field:
        field_out = field_batched
        if source_was_unbatched and field_was_unbatched:
            field_out = field_out.squeeze(0)
        outputs.append(field_out)

    if return_warped_source:
        warped_source = transformer(source_batched, field_batched)
        if source_was_unbatched:
            warped_source = warped_source.squeeze(0)
        outputs.append(warped_source)

    if return_warped_target:
        warped_target = transformer(target_batched, -field_batched)
        if target_was_unbatched:
            warped_target = warped_target.squeeze(0)
        outputs.append(warped_target)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)
