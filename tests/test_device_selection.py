import pytest
import torch

from constraints.devices import resolve_compute_device
from constraints.utils import signed_distance_kornia


def test_auto_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_compute_device() == torch.device("cuda")


def test_auto_device_uses_mps_before_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_compute_device() == torch.device("mps")


def test_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_compute_device() == torch.device("cpu")


def test_explicit_unavailable_accelerator_fails_early(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_compute_device("cuda")


def test_kornia_sdf_can_force_cpu_and_preserves_tensor_placement() -> None:
    mask = torch.zeros((1, 9, 9), dtype=torch.float32)
    mask[:, 3:6, 3:6] = 1

    sdf = signed_distance_kornia(mask, device="cpu")

    assert isinstance(sdf, torch.Tensor)
    assert sdf.shape == mask.shape
    assert sdf.device == mask.device
