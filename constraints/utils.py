import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from kornia.contrib import distance_transform as kornia_distance_transform
from scipy import ndimage


def get_repo_root() -> Path:
    """Traverse upwards to find the repository root marker."""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".mutagen").exists():
            return parent
    return current  # Fallback to cwd if marker not found


# Define your base paths relative to the repo root
REPO_ROOT = get_repo_root()
LOGS_DIR = REPO_ROOT / "logs"


def get_experiment_folder(experiment_name: str | Path) -> Path:
    """Returns the path to the experiment folder within the repository."""
    folder = REPO_ROOT / "outputs" / "notebooks" / experiment_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_data_folder()->Path:
    folder = REPO_ROOT / "data"
    assert folder.exists(), f"Data folder does not exist: {folder}"
    return folder

def rad2deg(rad):
    return rad * 180 / np.pi


def deg2rad(deg):
    return deg * np.pi / 180


def save_manifest(output_dir, args):
    manifest = {
        "args": vars(args),
        "timestamp": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        manifest["git_commit"] = None

    with open(Path(output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def signed_distance_scipy(
    mask: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Computes the signed distance field.

    Args:
        mask: np.ndarray of shape (H, W, C) or torch.Tensor of shape
            (B, C, H, W) / (C, H, W).
            Interpreted as boolean where True/1 = foreground, False/0 = background.

    Returns:
        sdf: Signed distance field matching the input type and shape.
             > 0 inside the object
             < 0 outside the object
             0 on the boundary
    """
    if isinstance(mask, torch.Tensor):
        if mask.ndim not in (3, 4):
            raise ValueError(
                f"Expected torch.Tensor of shape (B, C, H, W) or (C, H, W), got "
                f"{mask.ndim}D"
            )

        device = mask.device
        original_shape = mask.shape

        # Detach, move to CPU, and cast to boolean numpy array
        np_mask = mask.detach().cpu().numpy().astype(bool)

        # Flatten batch and channel dimensions together (N, H, W) to simplify the loop
        if mask.ndim == 4:
            np_mask = np_mask.reshape(-1, original_shape[-2], original_shape[-1])

        sdf = np.empty_like(np_mask, dtype=np.float32)

        # Apply scipy EDT independently over spatial dimensions
        for i in range(np_mask.shape[0]):
            m = np_mask[i]
            inside = np.asarray(ndimage.distance_transform_edt(~m), dtype=np.float32)
            outside = np.asarray(ndimage.distance_transform_edt(m), dtype=np.float32)
            sdf[i] = inside - outside

        # Reshape back to original torch shape and push back to original device
        sdf = sdf.reshape(original_shape)
        return torch.from_numpy(sdf).to(device)

    elif isinstance(mask, np.ndarray):
        if mask.ndim != 3:
            raise ValueError(
                f"Expected np.ndarray of shape (H, W, C), got {mask.ndim}D"
            )

        mask_bool = mask.astype(bool)

        # Original logic for (H, W, C)
        sdf = np.stack(
            [
                np.asarray(
                    ndimage.distance_transform_edt(~mask_bool[..., c]), dtype=np.float32
                )
                - np.asarray(
                    ndimage.distance_transform_edt(mask_bool[..., c]), dtype=np.float32
                )
                for c in range(mask_bool.shape[-1])
            ],
            axis=-1,
        )

        return sdf.astype(np.float32)

    else:
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(mask)}")


def signed_distance_kornia(
    mask: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Computes the signed distance field using Kornia's distance transform.

    Args:
        mask: np.ndarray of shape (H, W, C) or torch.Tensor of shape
            (B, C, H, W) / (C, H, W).
            Interpreted as boolean where True/1 = foreground, False/0 = background.

    Returns:
        sdf: Signed distance field matching the input type and shape.
             > 0 inside the object
             < 0 outside the object
             0 on the boundary
    """
    if isinstance(mask, torch.Tensor):
        if mask.ndim not in (3, 4):
            raise ValueError(
                f"Expected torch.Tensor of shape (B, C, H, W) or (C, H, W), got "
                f"{mask.ndim}D"
            )

        device = mask.device
        original_shape = mask.shape

        torch_mask = mask.detach().to(dtype=torch.float32)

        if mask.ndim == 3:
            torch_mask = torch_mask.unsqueeze(0)

        sdf = kornia_distance_transform(torch_mask)

        if mask.ndim == 3:
            sdf = sdf.squeeze(0)

        return sdf.to(device=device).reshape(original_shape)

    elif isinstance(mask, np.ndarray):
        if mask.ndim != 3:
            raise ValueError(
                f"Expected np.ndarray of shape (H, W, C), got {mask.ndim}D"
            )

        np_mask = mask.astype(np.float32)
        torch_mask = torch.from_numpy(np_mask).permute(2, 0, 1).unsqueeze(0)
        sdf = kornia_distance_transform(torch_mask)
        sdf = sdf.squeeze(0).permute(1, 2, 0).numpy()
        return sdf.cpu().numpy().astype(np.float32)

    else:
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(mask)}")
