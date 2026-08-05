import torch

from constraints.models.helpers import (
    MomentsAffineAlignment,
    rigid_matrix_to_grid_params,
)


def _rotation(angle: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)],
            ]
        ]
    )


def _sampled_pixel(angle, dx, dy, height, width, output_pixel):
    """Pixel of the input that affine_grid samples for a given output pixel."""
    theta = torch.stack(
        [
            torch.cos(angle),
            -torch.sin(angle),
            dx,
            torch.sin(angle),
            torch.cos(angle),
            dy,
        ],
        dim=1,
    ).reshape(1, 2, 3)
    grid = torch.nn.functional.affine_grid(
        theta, (1, 1, height, width), align_corners=False
    )
    x, y = output_pixel
    normalized = grid[0, y, x]
    return ((normalized + 1) * torch.tensor([width, height], dtype=torch.float32) - 1) / 2


def test_rigid_matrix_to_grid_params_reproduces_forward_pixel_mapping():
    """The grid must sample the template exactly where p_tmpl = R @ p_src + t."""
    height = width = 64
    rotation = _rotation(torch.tensor(0.45))
    translation = torch.tensor([[7.0, -5.0]])

    angle, dx, dy = rigid_matrix_to_grid_params(rotation, translation, height, width)

    for output_pixel in [(10, 10), (32, 32), (50, 20)]:
        sampled = _sampled_pixel(angle, dx, dy, height, width, output_pixel)
        p_src = torch.tensor(output_pixel, dtype=torch.float32)
        expected = rotation[0] @ p_src + translation[0]
        assert torch.allclose(sampled, expected, atol=1e-4)


def test_rigid_matrix_to_grid_params_pure_translation():
    """Without rotation the centre offset term vanishes: dx, dy are just 2t/size."""
    height = width = 64
    identity = _rotation(torch.tensor(0.0))
    translation = torch.tensor([[8.0, -4.0]])

    angle, dx, dy = rigid_matrix_to_grid_params(identity, translation, height, width)

    assert torch.allclose(angle, torch.zeros_like(angle), atol=1e-6)
    assert torch.allclose(dx, torch.tensor([2 * 8.0 / width]), atol=1e-6)
    assert torch.allclose(dy, torch.tensor([2 * -4.0 / height]), atol=1e-6)


def _delta_masks(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """One-hot volume (1, C+1, H, W) with a single lit pixel per foreground class."""
    volume = torch.zeros(1, points.shape[0] + 1, height, width)
    for index, (x, y) in enumerate(points):
        volume[0, index + 1, int(y.round().item()), int(x.round().item())] = 1.0
    return volume


def test_moments_alignment_recovers_known_rigid_transform():
    """Warping the template with the recovered params must land on the source."""
    height = width = 64
    angle_true = torch.tensor(0.6)
    rotation = _rotation(angle_true)[0]
    translation = torch.tensor([5.0, -3.0])

    template_points = torch.tensor([[20.0, 20.0], [40.0, 22.0], [30.0, 45.0]])
    # p_tmpl = R @ p_src + t  =>  p_src = R^T @ (p_tmpl - t)
    source_points = (rotation.T @ (template_points - translation).T).T

    template = _delta_masks(template_points, height, width)
    source = _delta_masks(source_points, height, width)

    spec = MomentsAffineAlignment().forward(source, template)

    assert spec.rigid is not None
    assert spec.steps is None and spec.field is None
    angle, dx, dy = spec.rigid.angle, spec.rigid.dx, spec.rigid.dy

    assert torch.allclose(angle, angle_true.reshape(1), atol=2e-2)

    # Every source centroid must map back onto its template centroid.
    for source_point, template_point in zip(source_points, template_points):
        pixel = (int(source_point[0].round().item()), int(source_point[1].round().item()))
        sampled = _sampled_pixel(angle, dx, dy, height, width, pixel)
        assert torch.allclose(sampled, template_point, atol=1.0)
