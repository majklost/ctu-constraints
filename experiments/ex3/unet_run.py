
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_jaccard_index
from pytorch_lightning.loggers import WandbLogger

from constraints import get_experiment_folder, get_data_folder
from constraints.datatools.datasets import (
    ARTIFICIAL_MASK_CLASS_LABELS,
    ARTIFICIAL_MASK_NUM_CLASSES,
    CachedArtificalDataset,
    artificial_mask_to_label_map,
)
import segmentation_models_pytorch as smp



FOLDER = get_experiment_folder(Path("ex3")/"project_arch_unet")
DATA = get_data_folder() / "artificial" / "downloaded"
WANDB_PROJECT = "Constraints"
WANDB_ENTITY = "ksicht"
FILE_NAME = Path(__file__).stem


class UnetProjectLightning(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3, num_classes: int = ARTIFICIAL_MASK_NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.unet = smp.Unet(
            "resnet18", encoder_weights="imagenet", in_channels=1, classes=num_classes
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.unet(image.float())

    def _prepare_wandb_image(self, image_batch: torch.Tensor, sample_idx: int) -> np.ndarray:
        image = image_batch[sample_idx].detach().cpu().float()
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim == 3 and image.shape[0] in (3, 4):
            image = image[:3].permute(1, 2, 0).contiguous()
        elif image.ndim == 3:
            image = image[0]

        image_min = float(image.min().item())
        image_max = float(image.max().item())
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        return (image.clamp(0.0, 1.0).numpy() * 255).astype("uint8")

    def _log_wandb_overlay(
        self,
        image: torch.Tensor,
        target_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        stage: str,
        batch_idx: int,
    ) -> None:
        if stage != "val" or batch_idx != 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is None or trainer.global_rank != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return

        experiment = trainer.logger.experiment
        key = f"{stage}/labels_overlay"
        experiment.define_metric(key, step_metric=f"{stage}/epoch")

        sample_idx = 0
        experiment.log(
            {
                key: [
                    wandb.Image(
                        self._prepare_wandb_image(image, sample_idx),
                        masks={
                            "ground_truth": {
                                "mask_data": target_labels[sample_idx].detach().cpu().numpy().astype("int32"),
                                "class_labels": ARTIFICIAL_MASK_CLASS_LABELS,
                            },
                            "predicted": {
                                "mask_data": pred_labels[sample_idx].detach().cpu().numpy().astype("int32"),
                                "class_labels": ARTIFICIAL_MASK_CLASS_LABELS,
                            },
                        },
                        caption=f"GT | pred | epoch={int(self.current_epoch)}",
                    )
                ],
                f"{stage}/epoch": int(self.current_epoch),
            }
        )

    def _shared_step(self, batch, batch_idx, stage: str):
        image = batch["image"]
        target_labels = artificial_mask_to_label_map(batch["mask"])
        logits = self.forward(image)
        loss = F.cross_entropy(logits, target_labels)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage == "train"),
        )

        if stage == "val":
            pred_labels = logits.argmax(dim=1)
            iou = multiclass_jaccard_index(
                preds=pred_labels,
                target=target_labels,
                num_classes=self.num_classes,
                average="macro",
            )
            self.log("val/iou/pred_vs_gt", iou, on_step=False, on_epoch=True, prog_bar=True)
            self._log_wandb_overlay(image, target_labels, pred_labels, stage, batch_idx)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



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
