import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from typing import Union


def get_repo_root() -> Path:
    """Traverses upwards to find the repository root marked by a .git or .mutagen folder."""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".mutagen").exists():
            return parent
    return current  # Fallback to cwd if marker not found


# Define your base paths relative to the repo root
REPO_ROOT = get_repo_root()
LOGS_DIR = REPO_ROOT / "logs"


def get_experiment_folder(experiment_name: str|Path) -> Path:
    """Returns the path to the experiment folder within the repository."""
    folder = REPO_ROOT / "outputs" / "notebooks" / experiment_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def rad2deg(rad):
    return rad * 180 / np.pi


def deg2rad(deg):
    return deg * np.pi / 180

def signed_distance_scipy(mask: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the signed distance field.
    
    Args:
        mask: np.ndarray of shape (H, W, C) or torch.Tensor of shape (B, C, H, W) / (C, H, W).
              Interpreted as boolean where True/1 = foreground, False/0 = background.
              
    Returns:
        sdf: Signed distance field matching the input type and shape.
             > 0 inside the object
             < 0 outside the object
             0 on the boundary
    """
    if isinstance(mask, torch.Tensor):
        if mask.ndim not in (3, 4):
            raise ValueError(f"Expected torch.Tensor of shape (B, C, H, W) or (C, H, W), got {mask.ndim}D")
            
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
            sdf[i] = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
            
        # Reshape back to original torch shape and push back to original device
        sdf = sdf.reshape(original_shape)
        return torch.from_numpy(sdf).to(device)

    elif isinstance(mask, np.ndarray):
        if mask.ndim != 3:
            raise ValueError(f"Expected np.ndarray of shape (H, W, C), got {mask.ndim}D")
            
        mask_bool = mask.astype(bool)
        
        # Original logic for (H, W, C)
        sdf = np.stack([
            ndimage.distance_transform_edt(~mask_bool[..., c]) -
            ndimage.distance_transform_edt(mask_bool[..., c])
            for c in range(mask_bool.shape[-1])
        ], axis=-1)
        
        return sdf.astype(np.float32)

    else:
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(mask)}")
