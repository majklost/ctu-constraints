import torch

from constraints.losses_metrics.losses import BlurredMSELoss


def test_blurred_mse_accepts_integer_one_hot_targets():
    labels = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.nn.functional.one_hot(labels, num_classes=2).movedim(-1, 1)
    prediction = torch.zeros_like(target, dtype=torch.float32, requires_grad=True)

    loss = BlurredMSELoss(kernel_size=1)(prediction, target)

    assert loss.dtype == prediction.dtype
    assert torch.allclose(loss, torch.tensor(0.5))
    loss.backward()
    assert prediction.grad is not None
