import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from pathlib import Path
import numpy as np
from typing import Literal, NotRequired, TypedDict

SDFMode = Literal["kornia", "scipy"]
ArtificialMaskLabel = Literal["background", "boundary", "lumen", "plaque"]
ArtificialForegroundMaskLabel = Literal["boundary", "lumen", "plaque"]

ARTIFICIAL_MASK_FOREGROUND_CHANNELS: dict[ArtificialForegroundMaskLabel, int] = {
    "boundary": 0,
    "lumen": 1,
    "plaque": 2,
}
ARTIFICIAL_MASK_LABEL_IDS: dict[ArtificialMaskLabel, int] = {
    "background": 0,
    "boundary": 1,
    "lumen": 2,
    "plaque": 3,
}
ARTIFICIAL_MASK_CLASS_LABELS: dict[int, str] = {
    label_id: label for label, label_id in ARTIFICIAL_MASK_LABEL_IDS.items()
}
ARTIFICIAL_MASK_NUM_CLASSES = len(ARTIFICIAL_MASK_LABEL_IDS)
ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS = len(ARTIFICIAL_MASK_FOREGROUND_CHANNELS)


def artificial_mask_to_label_map(mask: torch.Tensor) -> torch.Tensor:
    """Convert artificial masks to semantic labels with background=0."""
    if mask.ndim == 3:
        return artificial_mask_to_label_map(mask.unsqueeze(0))
    if mask.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] or [C, H, W] mask, got {tuple(mask.shape)}")
    if mask.shape[1] == ARTIFICIAL_MASK_NUM_CLASSES:
        return mask.argmax(dim=1).long()
    if mask.shape[1] != ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS:
        raise ValueError(
            f"Expected {ARTIFICIAL_MASK_NUM_CLASSES} semantic classes or "
            f"{ARTIFICIAL_MASK_NUM_FOREGROUND_CHANNELS} foreground channels, got {mask.shape[1]}"
        )

    foreground_scores = mask.float()
    foreground_labels = foreground_scores.argmax(dim=1).long() + 1
    has_foreground = foreground_scores.amax(dim=1) > 0
    background = torch.zeros_like(foreground_labels)
    return torch.where(has_foreground, foreground_labels, background)


def artificial_foreground_mask_to_explicit(mask: torch.Tensor) -> torch.Tensor:
    """Convert boundary/lumen/plaque channels to background/boundary/lumen/plaque."""
    if mask.ndim == 3:
        if mask.shape[0] == ARTIFICIAL_MASK_NUM_CLASSES:
            return mask.float()
        foreground_scores = mask.float()
        background = 1.0 - foreground_scores.amax(dim=0, keepdim=True)
        return torch.cat([background.clamp(0.0, 1.0), foreground_scores], dim=0)
    if mask.ndim == 4:
        if mask.shape[1] == ARTIFICIAL_MASK_NUM_CLASSES:
            return mask.float()
        foreground_scores = mask.float()
        background = 1.0 - foreground_scores.amax(dim=1, keepdim=True)
        return torch.cat([background.clamp(0.0, 1.0), foreground_scores], dim=1)
    raise ValueError(f"Expected [C, H, W] or [B, C, H, W] mask, got {tuple(mask.shape)}")


class Sample(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    template: torch.Tensor
    sdf: torch.Tensor
    transform: NotRequired[torch.Tensor]  # key may be missing entirely

class CachedArtificalDataset(Dataset):
    def __init__(self, folder:Path, augmentation:transforms.Compose|None=None, sdf_mode:SDFMode="scipy", return_transform:bool=False):
        assert augmentation is None, "Augmentation is not supported now"
        self._images = np.load(f'{folder}/img.npy', mmap_mode='r')
        self._masks  = np.load(f'{folder}/mask.npy',  mmap_mode='r')
        self._sdf_kornia   = np.load(f'{folder}/sdf_kornia.npy',   mmap_mode='r')
        self._sdf_scipy    = np.load(f'{folder}/sdf_scipy.npy',    mmap_mode='r')
        self._template      = np.load(f'{folder}/template.npy')
        self._transform = np.load(f'{folder}/transform.npy', mmap_mode='r')
        if sdf_mode not in set(SDFMode.__args__):
            raise ValueError(f"Unknown sdf_mode: {sdf_mode}")
        self._sdf_mode = sdf_mode
        self._return_transform = return_transform
    
    def __len__(self):
        return len(self._images)
    def __getitem__(self, idx) -> Sample:
        if self._sdf_mode == "kornia":
            sdf = torch.from_numpy(np.array(self._sdf_kornia[idx]))
        elif self._sdf_mode == "scipy":
            sdf = torch.from_numpy(np.array(self._sdf_scipy[idx]))
        else:
            raise ValueError(f"Unknown sdf_mode: {self._sdf_mode}")

        mask = torch.from_numpy(np.array(self._masks[idx]))
        template = torch.from_numpy(self._template)

        sample: Sample = {
            'image': torch.from_numpy(np.array(self._images[idx])),
            'mask': artificial_foreground_mask_to_explicit(mask),
            'template': artificial_foreground_mask_to_explicit(template),
            'sdf': sdf,
        }
        if self._return_transform:
            sample['transform'] = torch.from_numpy(np.array(self._transform[idx]))

        return sample
