"""
First project experiments
Extension of notebook project_firstex.ipynb in ex3
"""
from argparse import ArgumentParser
from pathlib import Path
import torch
import pytorch_lightning as pl
import wandb

from constraints.lightning_wrappers.modules import ProjectLightning
from constraints.datatools.datasets import CachedArtificalDataset
from constraints import get_experiment_folder, get_data_folder, show_torch_image
from constraints.transforms.transformers import RigidTransformer,DeformableTransformer
from constraints.computers.loss_computers import ProjectLossComputer
from constraints.losses import OneSideSDFSquare
from constraints.models.affine import ProjectWithTemplateA 
from constraints.models.deform_only import ProjectWithTemplateD
from constraints.computers.loss_computers import CrossEntrAndOneSide, CrossEntrOnly,OneSideOnly
from constraints.computers.metric_computers import DefaultSegmentationMetricComputer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger


FOLDER = get_experiment_folder(Path("ex3")/"project_arch_initial")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
LOSS_MODES = ["sanityS", "sanityD", "naive", "fullSDF", "fullCE"]


def main(args):

    if args.smoke_test:
        args.max_epochs = 1
        args.limit_train_batches = 1
        args.limit_val_batches = 1
    print(f"Experiment folder: {FOLDER}")
    print(f"W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    if args.modality == "affine":
        TRN_FOLDER = DATA / "trn" / "affine"
        VAL_FOLDER = DATA / "val" / "affine"
        transformer = RigidTransformer()
        net = ProjectWithTemplateA(max_translation=0.5)
    elif args.modality == "deformed":
        TRN_FOLDER = DATA / "trn" / "deformed"
        VAL_FOLDER = DATA / "val" / "deformed"
        transformer = DeformableTransformer()
        net = ProjectWithTemplateD()
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    
    trn_dataset = CachedArtificalDataset(TRN_FOLDER, sdf_mode="scipy")
    val_dataset = CachedArtificalDataset(VAL_FOLDER, sdf_mode="scipy")

    if args.loss_mode == "sanityS":
        loss_computer = CrossEntrAndOneSide(seg_loss_weight=20.0, sdf_loss_weight=0)
    elif args.loss_mode == "sanityD":
        loss_computer = CrossEntrAndOneSide(seg_loss_weight=0, sdf_loss_weight=1.0)
    elif args.loss_mode == "naive":
        loss_computer = CrossEntrAndOneSide(seg_loss_weight=20.0, sdf_loss_weight=1.0)
    elif args.loss_mode == "fullSDF":
        loss_computer = OneSideOnly(seg_loss_weight=1.0, sdf_loss_weight=1.0)
    elif args.loss_mode == "fullCE":
        loss_computer = CrossEntrOnly(seg_loss_weight=1.0, template_loss_weight=1.0)
    else:
        raise ValueError(f"Unknown loss mode: {args.loss_mode}")
    metric_computer = DefaultSegmentationMetricComputer(
    )

    module = ProjectLightning(
    net,
    transformer,
    loss_computer,
    metric_computer=metric_computer,
)

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.max_epochs
    LR = args.learning_rate

    trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


    
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="ex3-project-arch-initial",
        tags=["scratch", "overlay", "ex3", 'project_arch_initial'],
        settings=wandb.Settings(console="wrap"),  # pass settings through here instead
    )
    # Log all CLI arguments into W&B run config.
    if not args.smoke_test:
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    logger = False if args.smoke_test else wandb_logger

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[],
    )

    trainer.fit(module, train_dataloaders=trn_loader, val_dataloaders=val_loader)

    if not args.smoke_test:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--modality", type=str, choices=["affine", "deformed"])
    parser.add_argument("--loss_mode", type=str, choices=LOSS_MODES, default="naive")
    parser.add_argument("--smoke_test", action="store_true",
                     help="Run 1 train + 1 val batch, 1 epoch, no W&B, to sanity-check the run.")
    args = parser.parse_args()

    main(args)




