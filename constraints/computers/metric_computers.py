from abc import ABC, abstractmethod
from typing import cast

import torch
from torch import nn

from ..types import MetricInput, MetricResult, WandbOverlay
from ..visu.helpers import to_label_map


class MetricComputer(nn.Module, ABC):
    """Base class for configurable metric computation."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute(self, metric_input: MetricInput) -> MetricResult:
        """Compute structured metric output from one batch."""


class ProjectMetricComputer(MetricComputer):
    """Project-specific metric computer used by ProjectLightning."""

    @abstractmethod
    def compute(self, metric_input: MetricInput) -> MetricResult:
        """Implement project-specific metrics from the typed MetricInput contract."""


class NoOpMetricComputer(ProjectMetricComputer):
    """Default metric computer that emits nothing."""

    def compute(self, metric_input: MetricInput) -> MetricResult:
        del metric_input
        return MetricResult()


class CompositeMetricComputer(ProjectMetricComputer):
    """Compose multiple metric computers and merge their MetricResult outputs."""

    def __init__(self, metric_computers: list[ProjectMetricComputer]) -> None:
        super().__init__()
        self.metric_computers = nn.ModuleList(metric_computers)

    def compute(self, metric_input: MetricInput) -> MetricResult:
        merged_logs: dict[str, float | torch.Tensor] = {}
        merged_wandb_overlays: dict[str, WandbOverlay] = {}

        for metric_module in self.metric_computers:
            metric_computer = cast(ProjectMetricComputer, metric_module)
            result = metric_computer.compute(metric_input)

            if result.logs:
                for key, value in result.logs.items():
                    if key in merged_logs:
                        raise ValueError(f"Duplicate metric log key in CompositeMetricComputer: {key}")
                    merged_logs[key] = value

            if result.wandb_overlays:
                for key, value in result.wandb_overlays.items():
                    if key in merged_wandb_overlays:
                        raise ValueError(f"Duplicate metric overlay key in CompositeMetricComputer: {key}")
                    merged_wandb_overlays[key] = value

        return MetricResult(
            logs=merged_logs or None,
            wandb_overlays=merged_wandb_overlays or None,
        )


class LabelTripletImageMetricComputer(ProjectMetricComputer):
    """Log GT, warped, and predicted label maps as W&B mask overlays."""

    def __init__(
        self,
        stage: str = "val",
        every_n_epochs: int = 1,
        sample_idx: int = 0,
        num_classes: int | None = None,
        image_tag: str = "labels_overlay",
    ) -> None:
        super().__init__()
        if stage not in {"train", "val"}:
            raise ValueError("stage must be 'train' or 'val'")
        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be > 0")
        if sample_idx < 0:
            raise ValueError("sample_idx must be >= 0")

        self.stage = stage
        self.every_n_epochs = every_n_epochs
        self.sample_idx = sample_idx
        self.num_classes = num_classes
        self.image_tag = image_tag

    def _build_class_labels(self, inferred_num_classes: int) -> dict[int, str]:
        class_count = inferred_num_classes
        if self.num_classes is not None:
            class_count = max(class_count, int(self.num_classes))
        return {idx: f"class_{idx}" for idx in range(max(class_count, 1))}

    def _prepare_background_image(
        self,
        image_batch: torch.Tensor | None,
        sample_idx: int,
        fallback_shape: tuple[int, int],
    ) -> torch.Tensor:
        if image_batch is None:
            return torch.zeros(fallback_shape, dtype=torch.float32)

        if image_batch.ndim == 4:
            sample = image_batch[sample_idx]
        elif image_batch.ndim == 3:
            sample = image_batch
        else:
            raise ValueError(f"Unsupported image shape for W&B overlays: {tuple(image_batch.shape)}")

        sample = sample.detach().cpu().float()
        if sample.ndim == 3 and sample.shape[0] == 1:
            return sample[0]
        if sample.ndim == 3 and sample.shape[0] in (3, 4):
            return sample[:3].permute(1, 2, 0).contiguous()
        if sample.ndim == 2:
            return sample

        # Unknown channel layout: fall back to first channel.
        if sample.ndim == 3:
            return sample[0]
        return torch.zeros(fallback_shape, dtype=torch.float32)

    def should_compute(self, metric_input: MetricInput) -> bool:
        if metric_input.stage != self.stage:
            return False
        if metric_input.batch_idx != 0:
            return False
        if metric_input.current_epoch is None:
            return False
        if (metric_input.current_epoch + 1) % self.every_n_epochs != 0:
            return False
        return True

    def compute(self, metric_input: MetricInput) -> MetricResult:
        if not self.should_compute(metric_input):
            return MetricResult()

        pred_mask_logits = metric_input.segmentation_logits
        warped_template = metric_input.warped_template
        gt_mask = metric_input.gt_mask

        if pred_mask_logits is None or warped_template is None or gt_mask is None:
            return MetricResult()

        gt_labels = to_label_map(gt_mask)
        warped_labels = to_label_map(warped_template)
        pred_labels = to_label_map(pred_mask_logits)

        batch_size = min(gt_labels.shape[0], warped_labels.shape[0], pred_labels.shape[0])
        if self.sample_idx < 0 or self.sample_idx >= batch_size:
            return MetricResult()

        gt_sample = gt_labels[self.sample_idx].detach().cpu().long()
        warped_sample = warped_labels[self.sample_idx].detach().cpu().long()
        pred_sample = pred_labels[self.sample_idx].detach().cpu().long()

        inferred_classes = int(
            max(gt_sample.max().item(), warped_sample.max().item(), pred_sample.max().item()) + 1
        )
        class_labels = self._build_class_labels(inferred_classes)
        height, width = int(gt_sample.shape[0]), int(gt_sample.shape[1])

        background = self._prepare_background_image(
            image_batch=metric_input.image,
            sample_idx=self.sample_idx,
            fallback_shape=(height, width),
        )

        overlay = WandbOverlay(
            image=background,
            masks={
                "ground_truth": gt_sample,
                "warped": warped_sample,
                "predicted": pred_sample,
            },
            class_labels=class_labels,
            caption="GT | warped | pred",
        )
        return MetricResult(wandb_overlays={self.image_tag: overlay})
