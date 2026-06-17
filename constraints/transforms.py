import torch


def differentiable_rotation(image, angle):
    # Create a grid of coordinates corresponding to the input image.
    # Build theta from tensors so gradients can flow back to angle.
    angle = angle.reshape(-1).to(device=image.device, dtype=image.dtype)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    theta = torch.stack(
        [
            cos,
            -sin,
            zeros,
            sin,
            cos,
            zeros,
        ],
        dim=1,
    ).reshape(-1, 2, 3)  # Shape: (N, 2, 3)

    # print("theta_shape:", theta.shape)
    # print("image_shape:", image.shape)

    grid = torch.nn.functional.affine_grid(theta, image.size(), align_corners=False)
    rotated_image = torch.nn.functional.grid_sample(image, grid, align_corners=False)
    return rotated_image.squeeze(0)
