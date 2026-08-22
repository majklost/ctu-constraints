from collections.abc import Callable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# Whatever configure_optimizers() itself may legally return
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn

from ..computers.loss_computers import ProjectLossComputer
from ..computers.metric_computers import StagedMetricComputer
from ..computers.metric_terms import CompositeMetric
from ..datatools.datasets.types import Batch, TemplateBatch
from ..datatools.label_schema import LabelSchema
from ..datatools.template_refiners import IdentityTemplateRefiner, TemplateRefiner
from ..datatools.template_sources import TemplateSource
from ..logging.logging_providers import LightningWandbLoggingProvider
from ..models.segmentator import get_segmentator
from ..transforms.transformers import SpatialTransformer
from ..types import (
    STAGES,
    DiscreteSegmentation,
    LoggingProvider,
    LossInput,
    LossResult,
    MetricInput,
    StepContext,
    WarpResult,
)
from .sample_strategy import GtStrategy, NoGt

OptimizerFactory = Callable[[nn.Module], OptimizerLRScheduler]
LoggingProviderFactory = Callable[
    [Callable[..., None], Callable[[], bool]], LoggingProvider
]


_identity_refiner = IdentityTemplateRefiner()
_no_gt = NoGt()


def _default_logging_provider(
    scalar_log: Callable[..., None],
    is_global_zero: Callable[[], bool],
) -> LoggingProvider:
    return LightningWandbLoggingProvider(
        scalar_log=scalar_log,
        is_global_zero=is_global_zero,
    )


def _empty_staged_metrics() -> StagedMetricComputer:
    return StagedMetricComputer(
        {
            "train": CompositeMetric([]),
            "val": CompositeMetric([]),
            "val_extra": CompositeMetric([]),
            "test": CompositeMetric([]),
        }
    )



class ProjectLightning(pl.LightningModule):
    """The project architecture implemented in Lightning.

    Args:
        gt_strategy: Chooses whether ground truth is passed to the model during
            training and the primary validation pass. Its results are logged as
            ``train/*`` and ``val/*``.
        validation_strategy: Optionally runs a second validation forward pass
            for every batch. It is intended for a logits-only strategy such as
            ``NoGt()`` and logs independently as ``val_logits/*``. Set it to
            ``None`` to run only the primary validation pass.
    """

    def __init__(
        self,
        model: nn.Module,
        spatial_transform: SpatialTransformer,
        loss_computer: ProjectLossComputer,
        template_source: TemplateSource,
        label_schema: LabelSchema,
        template_refiner: TemplateRefiner = _identity_refiner,
        staged_metric_computer: StagedMetricComputer | None = None,
        logging_provider_factory: LoggingProviderFactory = _default_logging_provider,
        optimizer_callback: OptimizerFactory = lambda module: torch.optim.Adam(
            module.parameters(), lr=1e-3
        ),
        gt_strategy: GtStrategy = _no_gt,
        validation_strategy: GtStrategy | None = None,
    ):

        super().__init__()
        self._model = model
        self._spatial_transform = spatial_transform
        self._loss_computer = loss_computer
        self._staged_metric_computer = (
            staged_metric_computer
            if staged_metric_computer is not None
            else _empty_staged_metrics()
        )
        self._logging_provider = logging_provider_factory(
            self.log,
            lambda: self._trainer is None or self._trainer.is_global_zero,
        )
        self._optimizer_callback = optimizer_callback
        self._template_source = template_source
        self._template_refiner = template_refiner
        self._gt_strategy = gt_strategy
        self._validation_strategy = validation_strategy
        self._label_schema = label_schema

    def forward(
        self,
        img: torch.Tensor,
        template: torch.Tensor,
        template_sdf: torch.Tensor | None = None,
        gt: torch.Tensor | None = None,
        detach_seg: bool = False,
    ):
        segmentation_logits, transform_spec = self._model(
            img, template, gt=gt, detach_seg=detach_seg
        )
        mask_warp_result = self._spatial_transform(template, transform_spec)
        warped_template_sdf = None
        if template_sdf is not None:
            warped_template_sdf = self._spatial_transform(
                template_sdf, transform_spec
            ).warped_template
        warp_result = WarpResult(
            warped_template=mask_warp_result.warped_template,
            transform_spec=mask_warp_result.transform_spec,
            warped_mask=mask_warp_result.warped_mask,
            warped_template_sdf=warped_template_sdf,
        )
        return segmentation_logits, warp_result

    def _shared_step(
        self,
        batch: Batch,
        batch_idx,
        stage: STAGES,
        strategy: GtStrategy,
    ):
        context = StepContext(
            stage=stage,
            batch_idx=batch_idx,
            current_epoch=int(self.current_epoch),
            global_step=int(self.global_step),
        )
        img = batch["image"]
        template_batch = self._template_refiner(self._template_source(batch))

        decision = strategy.decide(batch, context)
        segmentation_logits, warp_result = self.forward(
            img,
            template_batch.masks,
            gt=decision.gt,
            detach_seg=decision.detach_seg,
            template_sdf=template_batch.sdfs,
        )
        # Plug into loss computer
        gt = DiscreteSegmentation(
            labels=batch["target_labels"],
            label_schema=self._label_schema,
        )
        loss_input = LossInput(
            segmentation_logits=segmentation_logits,
            warped_template=warp_result.warped_template,
            warped_template_sdf=warp_result.warped_template_sdf,
            gt=gt,
            gt_mask_sdf=batch.get("sdf"),
            transform_spec=warp_result.transform_spec,
        )
        loss_output = self._loss_computer.compute(loss_input)

        self._logging_provider.log_loss(
            context,
            loss_output,
            prog_bar=(stage == "train"),
        )

        metric_input = MetricInput(
            image=img,
            segmentation_logits=segmentation_logits,
            warped_template=warp_result.warped_template,
            gt=gt,
            gt_mask_sdf=batch.get("sdf"),
            transform_spec=warp_result.transform_spec,
            sample_ids=tuple(batch.get("sample_id", ())),
        )
        metric_result = self._staged_metric_computer.update(context, metric_input)
        self._logging_provider.log_batch(context, metric_result)

        return loss_output.total

    def training_step(self, batch: Batch, batch_idx):
        return self._shared_step(
            batch, batch_idx, stage="train", strategy=self._gt_strategy
        )

    def validation_step(self, batch: Batch, batch_idx):
        loss = self._shared_step(
            batch, batch_idx, stage="val", strategy=self._gt_strategy
        )
        if self._validation_strategy is not None:
            self._shared_step(
                batch,
                batch_idx,
                stage="val_extra",
                strategy=self._validation_strategy,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        context = StepContext("train", 0, int(self.current_epoch), int(self.global_step))
        result = self._staged_metric_computer.compute(context)
        self._logging_provider.log_epoch(context, result)
        self._staged_metric_computer.reset(context)

    def on_validation_epoch_end(self) -> None:
        for stage in ("val", "val_extra"):
            context = StepContext(stage, 0, int(self.current_epoch), int(self.global_step))
            result = self._staged_metric_computer.compute(context)
            self._logging_provider.log_epoch(context, result)
            self._staged_metric_computer.reset(context)

    def configure_optimizers(self):
        return self._optimizer_callback(self)


class UnetLightning(pl.LightningModule):
    def __init__(
        self,
        label_schema: LabelSchema,
        learning_rate: float = 1e-3,
        staged_metric_computer: StagedMetricComputer | None = None,
        logging_provider_factory: LoggingProviderFactory = _default_logging_provider,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "staged_metric_computer",
                "logging_provider_factory",
                "label_schema",
            ]
        )
        self._learning_rate = learning_rate
        self._label_schema = label_schema
        self._unet = get_segmentator(self._label_schema.num_classes)
        self._staged_metric_computer = (
            staged_metric_computer
            if staged_metric_computer is not None
            else _empty_staged_metrics()
        )
        self._logging_provider = logging_provider_factory(
            self.log,
            lambda: self._trainer is None or self._trainer.is_global_zero,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._unet(image.float())

    def _shared_step(self, batch: Batch, batch_idx: int, stage: STAGES):
        context = StepContext(
            stage=stage,
            batch_idx=batch_idx,
            current_epoch=int(self.current_epoch),
            global_step=int(self.global_step),
        )
        image = batch["image"]
        target_labels = batch["target_labels"]
        logits = self.forward(image)
        loss = F.cross_entropy(logits, target_labels)

        self._logging_provider.log_loss(
            context,
            LossResult(total=loss),
            prog_bar=(stage == "train"),
        )

        gt = DiscreteSegmentation(
            labels=target_labels,
            label_schema=self._label_schema,
        )
        metric_result = self._staged_metric_computer.update(
            context,
            MetricInput(
                image=image,
                segmentation_logits=logits,
                gt=gt,
                sample_ids=tuple(batch.get("sample_id", ())),
            ),
        )
        self._logging_provider.log_batch(context, metric_result)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def on_train_epoch_end(self) -> None:
        context = StepContext("train", 0, int(self.current_epoch), int(self.global_step))
        result = self._staged_metric_computer.compute(context)
        self._logging_provider.log_epoch(context, result)
        self._staged_metric_computer.reset(context)

    def on_validation_epoch_end(self) -> None:
        context = StepContext("val", 0, int(self.current_epoch), int(self.global_step))
        result = self._staged_metric_computer.compute(context)
        self._logging_provider.log_epoch(context, result)
        self._staged_metric_computer.reset(context)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)
