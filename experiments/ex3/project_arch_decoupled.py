"""
Determine coupling of UNET and registration network
Extension of project_arch_initial.py in ex3
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from constraints import get_data_folder, get_experiment_folder, show_torch_image
from constraints.computers.loss_computers import ProjectLossComputer
from constraints.datatools.datasets import CachedArtificialDataset
from constraints.datatools.template_sources import PerSampleTemplateSource
from constraints.factories.losses import create_loss_computer
from constraints.factories.metrics import create_default_staged_metrics
from constraints.lightning_wrappers.modules import ProjectLightning
from constraints.lightning_wrappers.sample_strategy import AlwaysGt
from constraints.losses_metrics import OneSideSDFSquare
from constraints.models.affine import ProjectWithTemplateA
from constraints.models.deform_only import ProjectWithTemplateD
from constraints.transforms.transformers import DeformableTransformer, RigidTransformer

FOLDER = get_experiment_folder(Path("ex3") / "project_arch_coupling")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
MODES = [
    "decoupledOneSideSDF",
    "decoupledCE",
    "decoupledStandard",
    "decoupledDSDF",
    "decoupledCentroid",
    "decoupledBlurred",
]
FILE_NAME = Path(__file__).stem


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def main(args):
    print(f"Experiment folder: {FOLDER}")
    print(f"W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    configure_reproducibility(seed=args.seed)
    print(f"Seed: {args.seed}")
    print("Determinism check: warn_only")

    if args.modality == "affine":
        TRN_FOLDER = DATA / "trn" / "affine"
        VAL_FOLDER = DATA / "val" / "affine"
        transformer = RigidTransformer()
    elif args.modality == "deformed":
        TRN_FOLDER = DATA / "trn" / "deformed"
        VAL_FOLDER = DATA / "val" / "deformed"
        transformer = DeformableTransformer()
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    trn_dataset = CachedArtificialDataset(TRN_FOLDER, sdf_mode="scipy")
    val_dataset = CachedArtificialDataset(VAL_FOLDER, sdf_mode="scipy")
    label_schema = trn_dataset.label_schema
    template_source = PerSampleTemplateSource(
        trn_dataset.template_assets, label_schema
    )
    if args.modality == "affine":
        net = ProjectWithTemplateA(label_schema, max_translation=0.5)
    else:
        net = ProjectWithTemplateD(label_schema)
    sample_strategy = AlwaysGt(label_schema)

    if args.mode == "decoupledOneSideSDF":
        loss_computer = create_loss_computer("one_side_sdf_squared_one_side_sdf_squared", label_schema)
    elif args.mode == "decoupledCE":
        loss_computer = create_loss_computer("bce_bce", label_schema)
    elif args.mode == "decoupledStandard":
        loss_computer = create_loss_computer("bce_one_side_sdf_squared", label_schema, segmentation_weight=1.0)
    elif args.mode == "decoupledDSDF":
        loss_computer = create_loss_computer("bce_dsdf_mse", label_schema)
    elif args.mode == "decoupledCentroid":
        loss_computer = create_loss_computer("bce_centroid", label_schema)
    elif args.mode == "decoupledBlurred":
        loss_computer = create_loss_computer("bce_blurred_mse", label_schema)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    staged_metric_computer = create_default_staged_metrics(label_schema)

    optimizer_callback = lambda module: torch.optim.Adam(
        module.parameters(), lr=args.learning_rate
    )

    module = ProjectLightning(
        model=net,
        spatial_transform=transformer,
        loss_computer=loss_computer,
        template_source=template_source,
        label_schema=label_schema,
        staged_metric_computer=staged_metric_computer,
        optimizer_callback=optimizer_callback,
        gt_strategy=sample_strategy,
    )

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.max_epochs
    train_generator = torch.Generator().manual_seed(args.seed)

    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
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
        group=f"ex4-{FILE_NAME}-{args.mode}-{args.modality}",
        name=f"seed{args.seed}-bs{args.batch_size}-lr{args.learning_rate}-epochs{args.max_epochs}",
        tags=[
            "scratch",
            "overlay",
            "ex4",
            f"{FILE_NAME}",
            f"{args.mode}",
            f"{args.modality}",
        ],
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
        deterministic="warn",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality", type=str, choices=["affine", "deformed"])
    parser.add_argument("--mode", type=str, choices=MODES)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use Lightning's fast_dev_run to sanity-check the run.",
    )
    args = parser.parse_args()

    main(args)
