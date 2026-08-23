"""
First project experiments
Extension of notebook project_firstex.ipynb in ex3

Results
- investigated fully coupled (gradient was passed via segmentation logits)
rigid
    - converged fullSDF, naive
- naive - 0.87 IoU warp, 0.97 IoU pred
- fullSDF - 0.84 IoU warp, 0.96 IoU pred

- sanity checks were not really sanity (due to gradient passing)

deformed
    - converged only fullSDF



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
from constraints.datatools.datasets import CachedArtificialDataset
from constraints.datatools.template_sources import PerSampleTemplateSource
from constraints.factories.losses import create_loss_computer
from constraints.factories.metrics import create_default_staged_metrics
from constraints.lightning_wrappers.modules import ProjectLightning
from constraints.models.rigid import ProjectWithTemplateRigid
from constraints.models.deform_only import ProjectWithTemplateD
from constraints.transforms.transformers import DeformableTransformer, RigidTransformer

FOLDER = get_experiment_folder(Path("ex3")/"project_arch_initial")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
LOSS_MODES = ["sanityS", "sanityD", "naive", "fullSDF", "fullCE"]
FILE_NAME = Path(__file__).stem


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main(args):


    print(f"Experiment folder: {FOLDER}")
    print(f"W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    configure_reproducibility(seed=args.seed)
    print(f"Seed: {args.seed}")
    print("Determinism check: warn_only")

    if args.modality == "rigid":
        TRN_FOLDER = DATA / "trn" / "rigid"
        VAL_FOLDER = DATA / "trn" / "rigid"
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
    if args.modality == "rigid":
        net = ProjectWithTemplateRigid(label_schema, max_translation=0.5)
    else:
        net = ProjectWithTemplateD(label_schema)

    if args.loss_mode == "sanityS":
        loss_computer = create_loss_computer("bce_one_side_sdf_squared", label_schema, registration_weight=0)
    elif args.loss_mode == "sanityD":
        loss_computer = create_loss_computer("bce_one_side_sdf_squared", label_schema, segmentation_weight=0)
    elif args.loss_mode == "naive":
        loss_computer = create_loss_computer("bce_one_side_sdf_squared", label_schema)
    elif args.loss_mode == "fullSDF":
        loss_computer = create_loss_computer("one_side_sdf_squared_one_side_sdf_squared", label_schema)
    elif args.loss_mode == "fullCE":
        loss_computer = create_loss_computer("bce_bce", label_schema)
    else:
        raise ValueError(f"Unknown loss mode: {args.loss_mode}")
    staged_metric_computer = create_default_staged_metrics(label_schema)
    LR = args.learning_rate
    optimizer_callback = lambda module: torch.optim.Adam(module.parameters(), lr=LR)

    module = ProjectLightning(
    model=net,
    spatial_transform=transformer,
    loss_computer=loss_computer,
    template_source=template_source,
    label_schema=label_schema,
    staged_metric_computer=staged_metric_computer,
    optimizer_callback=optimizer_callback
)

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EPOCHS = args.max_epochs
    data_generator = torch.Generator().manual_seed(args.seed)

    trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    generator=data_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=data_generator,
    )


    
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"ex3-{FILE_NAME}-{args.loss_mode}-{args.modality}",
        tags=["scratch", "overlay", "ex3", f"{FILE_NAME}", f"{args.loss_mode}", f"{args.modality}"],
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
    parser.add_argument("--modality", type=str, choices=["rigid", "deformed"])
    parser.add_argument("--loss_mode", type=str, choices=LOSS_MODES, default="naive")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use Lightning's fast_dev_run to sanity-check the run.")
    args = parser.parse_args()

    main(args)
