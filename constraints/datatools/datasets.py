import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from pathlib import Path
import numpy as np
from typing import Literal, TypedDict

SDFMode = Literal["kornia", "scipy"]

class Sample(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    template: torch.Tensor
    sdf: torch.Tensor | None

class CachedArtificalDataset(Dataset):
    def __init__(self, folder:Path, augmentation:transforms.Compose|None=None, sdf_mode:SDFMode="scipy"):
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
    
    def __len__(self):
        return len(self._images)
    def __getitem__(self, idx) -> Sample:
        sdf = None
        if self._sdf_mode == "kornia":
            sdf = torch.from_numpy(np.array(self._sdf_kornia[idx]))
        elif self._sdf_mode == "scipy":
            sdf = torch.from_numpy(np.array(self._sdf_scipy[idx]))
        else:
            raise ValueError(f"Unknown sdf_mode: {self._sdf_mode}")
        return {
            'image': torch.from_numpy(np.array(self._images[idx])),
            'mask':  torch.from_numpy(np.array(self._masks[idx])),
            'template': torch.from_numpy(self._template), # always the same template
            'sdf':  sdf,
        }
