"""
Determine coupling of UNET and registration network
Extension of project_arch_initial.py in ex3
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
from constraints.computers.loss_computers import BlurredMSEComputer, CentroidComputer, ProjectLossComputer, DSDFComputer
from constraints.losses import OneSideSDFSquare
from constraints.models.affine import ProjectWithTemplateA 
from constraints.models.deform_only import ProjectWithTemplateD
from constraints.computers.loss_computers import CrossEntrAndOneSide, CrossEntrOnly,OneSideOnly
from constraints.computers.metric_computers import DefaultSegmentationMetricComputer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from constraints.lightning_wrappers.sample_strategy import AlwaysGt

FOLDER = get_experiment_folder(Path("ex3")/"project_arch_coupling")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
MODES = ["decoupledOneSideSDF", "decoupledCE", "decoupledStandard", "decoupledDSDF", "decoupledCentroid", "decoupledBlurred"]
FILE_NAME = Path(__file__).stem


def main(args):
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
    sample_strategy = AlwaysGt()


    if args.mode == "decoupledOneSideSDF":
        loss_computer = OneSideOnly(seg_loss_weight=1.0, sdf_loss_weight=1.0)
    elif args.mode == "decoupledCE":
        loss_computer = CrossEntrOnly(seg_loss_weight=1.0, template_loss_weight=1.0)
    elif args.mode == "decoupledStandard":
        loss_computer = CrossEntrAndOneSide(seg_loss_weight=1.0, sdf_loss_weight=1.0)
    elif args.mode == "decoupledDSDF":
        loss_computer = DSDFComputer()
    elif args.mode == "decoupledCentroid":
        loss_computer = CentroidComputer()
    elif args.mode == "decoupledBlurred":
        loss_computer = BlurredMSEComputer()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    metric_computer = DefaultSegmentationMetricComputer(
    )

    optimizer_callback = lambda module: torch.optim.Adam(module.parameters(), lr=args.learning_rate)

    module = ProjectLightning(
    net,
    transformer,
    loss_computer,
    metric_computer=metric_computer,
    optimizer_callback=optimizer_callback,
    gt_strategy=sample_strategy
    )

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.max_epochs

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
        name=f"ex3-{FILE_NAME}-{args.mode}-{args.modality}",
        tags=["scratch", "overlay", "ex3", f"{FILE_NAME}", f"{args.mode}", f"{args.modality}"],
        settings=wandb.Settings(console="wrap"),  # pass settings through here instead
    )




    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.max_epochs

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
        name=f"ex3-{FILE_NAME}-{args.mode}-{args.modality}",
        tags=["scratch", "overlay", "ex3", f"{FILE_NAME}", f"{args.mode}", f"{args.modality}"],
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
    fast_dev_run=args.smoke_test,
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
    parser.add_argument("--mode", type=str, choices=MODES)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use Lightning's fast_dev_run to sanity-check the run.")
    args = parser.parse_args()

    main(args)
