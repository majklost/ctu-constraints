"""Shared compute-device selection for offline artifact preparation."""

import torch

type DeviceSelection = torch.device | str | None


def resolve_compute_device(device: DeviceSelection = "auto") -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU.

    ``None`` is treated as ``"auto"``. Explicit accelerator requests fail
    early when unavailable instead of reaching a less informative tensor
    operation later.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return resolved
