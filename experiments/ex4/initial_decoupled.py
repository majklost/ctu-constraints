"""
Initial comparision of different losses
We use them in decoupled manner - we pass GT into projector
similar to ex3 but more losses and better metrics measurement

Tested pairs (segm + reg):
- BCE + OneSideSDFSquared x
- BCE + OneSideSDFPlain x
- BCE + BCE x
- BCE + CentroidLoss x
- BCE + BlurredLoss x
- BCE + DSDF_MSE x
- BCE + SDFTEMPLATE_MSE
- BCE + SDFTEMPLATE_OneSideSDFSQUARE
- OneSideSDFSquared + OneSideSDFSquared x
- OneSideSDFPlain + OneSideSDFPlain
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
import wandb

from constraints.lightning_wrappers.modules import ProjectLightning, UnetLightning
from constraints.datatools.datasets import CachedArtificalDataset
from constraints import get_experiment_folder, get_data_folder, show_torch_image, show_torch_mask
from constraints.transforms.transformers import RigidTransformer,DeformableTransformer
