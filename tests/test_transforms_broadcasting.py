import torch

from constraints.transforms.transforms import (
    differentiable_rigid,
    differentiable_rotation,
    field_application,
)


def test_differentiable_rigid_broadcasts_single_image_to_parameter_batch():
    image = torch.arange(1 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8)

    angles = torch.tensor([0.0, 0.2, -0.35], dtype=torch.float32)
    dx = torch.tensor([0.0, 0.1, -0.15], dtype=torch.float32)
    dy = torch.tensor([0.0, -0.05, 0.2], dtype=torch.float32)

    out = differentiable_rigid(image, angles, dx, dy)

    image_batched = image.unsqueeze(0).expand(angles.numel(), -1, -1, -1)
    out_expected = differentiable_rigid(image_batched, angles, dx, dy)

    assert out.shape == (angles.numel(), *image.shape)
    assert torch.allclose(out, out_expected, atol=1e-6, rtol=1e-6)


def test_differentiable_rotation_broadcasts_single_image_to_angle_batch():
    image = torch.arange(1 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8)
    angles = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float32)

    out = differentiable_rotation(image, angles)

    image_batched = image.unsqueeze(0).expand(angles.numel(), -1, -1, -1)
    out_expected = differentiable_rotation(image_batched, angles)

    assert out.shape == (angles.numel(), *image.shape)
    assert torch.allclose(out, out_expected, atol=1e-6, rtol=1e-6)


def test_field_application_broadcasts_single_source_to_field_batch():
    source = torch.arange(1 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8)

    displacement = torch.zeros(3, 2, 8, 8, dtype=torch.float32)
    displacement[1, 0, :, :] = 0.5
    displacement[2, 1, :, :] = -0.75

    result = field_application(source, displacement)
    assert result.warped_source is not None
    warped = result.warped_source

    source_batched = source.unsqueeze(0).expand(displacement.shape[0], -1, -1, -1)
    result_expected = field_application(source_batched, displacement)
    assert result_expected.warped_source is not None
    warped_expected = result_expected.warped_source

    assert warped.shape == (displacement.shape[0], *source.shape)
    assert torch.allclose(warped, warped_expected, atol=1e-6, rtol=1e-6)
