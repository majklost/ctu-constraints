
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from constraints import get_experiment_folder, get_data_folder
from constraints.datatools.datasets import (
    ARTIFICIAL_MASK_CLASS_LABELS,
    ARTIFICIAL_MASK_NUM_CLASSES,
    CachedArtificalDataset,
    artificial_mask_to_label_map,
)

from constraints.lightning_wrappers.modules import UnetLightning as UnetProjectLightning


FOLDER = get_experiment_folder(Path("ex3")/"project_arch_unet")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
FILE_NAME = Path(__file__).stem




def main(args):
    print(f"Experiment folder: {FOLDER}")
    print(f"W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    if args.modality == "affine":
        TRN_FOLDER = DATA / "trn" / "affine"
        VAL_FOLDER = DATA / "val" / "affine"

    elif args.modality == "deformed":
        TRN_FOLDER = DATA / "trn" / "deformed"
        VAL_FOLDER = DATA / "val" / "deformed"
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    

    trn_dataset = CachedArtificalDataset(TRN_FOLDER, sdf_mode="scipy")
    val_dataset = CachedArtificalDataset(VAL_FOLDER, sdf_mode="scipy")

    module = UnetProjectLightning(learning_rate=args.learning_rate)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"ex3-{FILE_NAME}-unet-{args.modality}",
        tags=["baseline", "unet", "overlay", "ex3", FILE_NAME, args.modality],
        settings=wandb.Settings(console="wrap"),
    )
    if not args.smoke_test:
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    logger = False if args.smoke_test else wandb_logger

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        fast_dev_run=args.smoke_test,
        callbacks=[],
    )

    trainer.fit(module, train_dataloaders=trn_loader, val_dataloaders=val_loader)

    if not args.smoke_test:
        wandb_logger.experiment.finish()

if  __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--modality", type=str, choices=["affine", "deformed"])
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick test with a small dataset and fewer epochs.")
    args = parser.parse_args()

    main(args)
