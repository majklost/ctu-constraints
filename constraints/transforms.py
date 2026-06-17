import torch


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
