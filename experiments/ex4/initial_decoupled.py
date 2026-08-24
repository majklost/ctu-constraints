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
- BCE + SDFTEMPLATE_MSE x
- BCE + SDFTEMPLATE_OneSideSDFSQUARE x
- OneSideSDFSquared + OneSideSDFSquared x
- OneSideSDFPlain + OneSideSDFPlain x
- UNET
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from constraints import (
    get_data_folder,
    get_experiment_folder,
    show_torch_image,
    show_torch_mask,
)
from constraints.computers.loss_computers import ProjectLossComputer
from constraints.computers.metric_computers import StagedMetricComputer
from constraints.computers.overlay_computers import SegmentationOverlayComputer
from constraints.datatools.datasets import CachedArtificialDataset
from constraints.datatools.label_schema import LabelSchema
from constraints.datatools.template_sources import (
    PerSampleTemplateSource,
    TemplateSource,
)
from constraints.factories.losses import create_loss_computer
from constraints.factories.metrics import create_default_staged_metrics
from constraints.lightning_wrappers.callbacks import (
    SegmentationRegistrationEarlyStopping,
)
from constraints.lightning_wrappers.modules import ProjectLightning, UnetLightning
from constraints.lightning_wrappers.sample_strategy import AlwaysGt, NoGt
from constraints.logging.wandb_factory import create_wandb_logger
from constraints.models.deform_only import ProjectWithTemplateD
from constraints.models.rigid import ProjectWithTemplateRigid
from constraints.models.rigid_deform import (
    ProjectWithTemplateBCalcRigid,
    ProjectWithTemplateBDeepRigid,
)
from constraints.models.segmentator import set_segmentator_encoder_weights
from constraints.transforms.transformers import (
    DeformableTransformer,
    RigidTransformer,
    SequentialTransformer,
    SpatialTransformer,
)
from constraints.types import OverlayPolicy

FOLDER = get_experiment_folder(Path("ex4") / "initial_decoupled")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints2"
WANDB_ENTITY = "mrkosmic-ctu"
# COUPLING_OPTIONS = ["full", "decoupled"]
SAMPLE_STRATEGY_OPTIONS = ["always_gt", "no_gt"]
SAMPLE_STRATEGY_EXTRA = SAMPLE_STRATEGY_OPTIONS + ["none"]
MODES = [
    "UNET",
    "BCE_OneSideSDFSquared",
    "BCE_OneSideSDFPlain",
    "BCE_BCE",
    "BCE_CentroidLoss",
    "BCE_BlurredLoss",
    "BCE_DSDF_MSE",
    "BCE_SDFTEMPLATE_MSE",
    "BCE_SDFTEMPLATE_OneSideSDFSQUARE",
    "OneSideSDFSquared_OneSideSDFSquared",
    "OneSideSDFPlain_OneSideSDFPlain",
]

MODALITIES = ["rigid", "deformed", "both"]
RIGID_DEF_MODES = ["calc", "deep"]
LOSS_PRESETS = {
    "BCE_OneSideSDFSquared": "bce_one_side_sdf_squared",
    "BCE_OneSideSDFPlain": "bce_one_side_sdf_plain",
    "BCE_BCE": "bce_bce",
    "BCE_CentroidLoss": "bce_centroid",
    "BCE_BlurredLoss": "bce_blurred_mse",
    "BCE_DSDF_MSE": "bce_dsdf_mse",
    "BCE_SDFTEMPLATE_MSE": "bce_sdf_template_mse",
    "BCE_SDFTEMPLATE_OneSideSDFSQUARE": "bce_sdf_template_one_side_sdf_squared",
    "OneSideSDFSquared_OneSideSDFSquared": "one_side_sdf_squared_one_side_sdf_squared",
    "OneSideSDFPlain_OneSideSDFPlain": "one_side_sdf_plain_one_side_sdf_plain",
}


FILE_NAME = Path(__file__).stem


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(seed, workers=True)


def handle_unet(
    args, staged_metric_computer: StagedMetricComputer, ls: LabelSchema
) -> pl.LightningModule:
    return UnetLightning(
        learning_rate=args.learning_rate,
        staged_metric_computer=staged_metric_computer,
        label_schema=ls,
    )


def handle_decoupled(
    args,
    transformer: SpatialTransformer,
    staged_metric_computer: StagedMetricComputer,
    label_schema: LabelSchema,
    template_source: TemplateSource,
) -> pl.LightningModule:
    match args.modality:
        case "rigid":
            net = ProjectWithTemplateRigid(max_translation=0.5, ls=label_schema)
        case "deformed":
            net = ProjectWithTemplateD(ls=label_schema)
        case "both":
            if args.rigid_def_mode == "calc":
                net = ProjectWithTemplateBCalcRigid(
                    label_schema=label_schema, max_translation=0.5
                )
            elif args.rigid_def_mode == "deep":
                net = ProjectWithTemplateBDeepRigid(
                    ls=label_schema, max_translation=0.5
                )
            else:
                raise ValueError(f"Unknown rigid_def_mode: {args.rigid_def_mode}")
        case _:
            raise ValueError(f"Unknown modality: {args.modality}")

    try:
        preset = LOSS_PRESETS[args.mode]
    except KeyError as exc:
        raise ValueError(f"Unknown mode: {args.mode}") from exc
    loss_computer = create_loss_computer(
        preset,
        label_schema,
        field_regularization_weight=args.deformation_regularization_weight,
    )
    optimizer_callback = lambda module: torch.optim.Adam(
        module.parameters(), lr=args.learning_rate
    )

    match args.learning_sample_strategy:
        case "always_gt":
            sample_strategy = AlwaysGt(label_schema=label_schema)
        case "no_gt":
            sample_strategy = NoGt(detach_seg=True)
        case _:
            raise ValueError(
                f"Unknown learning_sample_strategy: {args.learning_sample_strategy}"
            )

    match args.validation_sample_strategy:
        case "always_gt":
            validation_strategy = AlwaysGt(label_schema=label_schema)
        case "no_gt":
            validation_strategy = NoGt(detach_seg=True)
        case "none":
            validation_strategy = None
        case _:
            raise ValueError(
                f"Unknown validation_sample_strategy: {args.validation_sample_strategy}"
            )

    overlay_policy = OverlayPolicy(
        stages=frozenset({"val"}),
        every_n_epochs=1,
        first_n_samples=2,
    )

    overlay_computer = SegmentationOverlayComputer(label_schema, overlay_policy)

    module = ProjectLightning(
        model=net,
        spatial_transform=transformer,
        loss_computer=loss_computer,
        staged_metric_computer=staged_metric_computer,
        optimizer_callback=optimizer_callback,
        gt_strategy=sample_strategy,
        validation_strategy=validation_strategy,
        label_schema=label_schema,
        template_source=template_source,
        overlay_computers=(overlay_computer,),
    )
    return module


def main(args):
    print(f"Experiment folder: {FOLDER}")
    print(f"W&B project: {WANDB_ENTITY}/{WANDB_PROJECT}")
    configure_reproducibility(seed=args.seed)
    print(f"Seed: {args.seed}")
    if args.rigid_def_mode is not None and args.modality != "both":
        raise ValueError("rigid_def_mode can only be used with modality 'both'")
    if args.modality == "both" and args.rigid_def_mode is None:
        raise ValueError(
            "When modality is 'both', rigid_def_mode must be specified (either 'calc' or 'deep')"
        )

    if args.segmentator_unlearned:
        set_segmentator_encoder_weights(None)

    if args.modality == "rigid":
        TRN_FOLDER = DATA / "rigid" / "trn"
        VAL_FOLDER = DATA / "rigid" / "val"
        transformer = RigidTransformer()
    elif args.modality == "deformed":
        TRN_FOLDER = DATA / "deformed" / "trn"
        VAL_FOLDER = DATA / "deformed" / "val"
        transformer = DeformableTransformer()
    elif args.modality == "both":
        TRN_FOLDER = DATA / "rigid_deformed" / "trn"
        VAL_FOLDER = DATA / "rigid_deformed" / "val"
        transformer = SequentialTransformer()
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    return_template_sdf = args.mode in [
        "BCE_SDFTEMPLATE_MSE",
        "BCE_SDFTEMPLATE_OneSideSDFSQUARE",
    ]
    trn_dataset = CachedArtificialDataset(
        TRN_FOLDER, sdf_mode=args.sdf_mode, return_template_sdf=return_template_sdf
    )
    val_dataset = CachedArtificialDataset(
        VAL_FOLDER, sdf_mode=args.sdf_mode, return_template_sdf=return_template_sdf
    )
    template_source = PerSampleTemplateSource(
        trn_dataset.template_assets, trn_dataset.label_schema
    )

    staged_metric_computer = create_default_staged_metrics(trn_dataset.label_schema)

    if args.mode == "UNET":
        module = handle_unet(args, staged_metric_computer, ls=trn_dataset.label_schema)
    else:
        module = handle_decoupled(
            args,
            transformer,
            staged_metric_computer,
            label_schema=trn_dataset.label_schema,
            template_source=template_source,
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

    group_name = (
        f"ex3-{FILE_NAME}-{args.mode}-{args.modality}"  # identifies the "approach"
    )
    if args.rigid_def_mode is not None:
        # calc and deep are different approaches - they must not share a group.
        group_name += f"-{args.rigid_def_mode}"

    TAGS = [
        "scratch",
        "overlay",
        "ex3",
        FILE_NAME,
        args.mode,
        args.modality,
        "newer",
    ]
    if args.rigid_def_mode is not None:
        TAGS.append(args.rigid_def_mode)
    if args.special_tag:
        TAGS.append(args.special_tag)

    wandb_logger = create_wandb_logger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{group_name}-seed{args.seed}",  # unique per run, human-readable
        group=group_name,  # ties all seeds of this approach together
        job_type="train",  # distinguishes from later "aggregate" runs
        tags=TAGS,
        config=None if args.smoke_test else vars(args),
    )

    logger = False if args.smoke_test else wandb_logger
    callbacks = []
    if not args.smoke_test:
        callbacks.append(
            SegmentationRegistrationEarlyStopping(
                patience=args.early_stopping_patience,
                segmentation_min_delta=args.early_stopping_min_delta,
                registration_min_delta=args.registration_early_stopping_min_delta,
            )
        )

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
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=trn_loader, val_dataloaders=val_loader)

    if not args.smoke_test:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=MODES, required=True)
    parser.add_argument("--modality", type=str, choices=MODALITIES, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--deformation_regularization_weight",
        type=float,
        default=0.0,
        help="Weight of VoxelMorph L2 diffusion regularization for displacement fields.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help=(
            "Stop after this many validation epochs without a meaningful IoU "
            "improvement."
        ),
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-3,
        help="Minimum validation macro-IoU increase required to reset patience.",
    )
    parser.add_argument(
        "--registration_early_stopping_min_delta",
        type=float,
        default=1e-3,
        help="Minimum validation registration-IoU increase required to reset patience.",
    )
    parser.add_argument("--smoke_test", action="store_true")
    # parser.add_argument(
    #     "--coupling", type=str, choices=COUPLING_OPTIONS, default="decoupled"
    # )
    parser.add_argument(
        "--learning_sample_strategy",
        type=str,
        choices=SAMPLE_STRATEGY_OPTIONS,
        default="always_gt",
    )
    parser.add_argument(
        "--validation_sample_strategy",
        type=str,
        choices=SAMPLE_STRATEGY_EXTRA,
        default="no_gt",
    )
    parser.add_argument(
        "--sdf_mode", type=str, choices=["scipy", "kornia"], default="scipy"
    )
    parser.add_argument(
        "--segmentator_unlearned",
        action="store_true",
        help="Use an unlearned segmentator.",
    )
    parser.add_argument(
        "--special_tag", type=str, default="", help="Add a special tag to the W&B run."
    )
    parser.add_argument(
        "--rigid_def_mode",
        type=str,
        choices=RIGID_DEF_MODES,
        default=None,
        help="Choose the rigid-deformable registration mode.",
    )
    args = parser.parse_args()

    main(args)
