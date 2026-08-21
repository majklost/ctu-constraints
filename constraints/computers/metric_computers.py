from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from ..types import STAGES, MetricInput, MetricResult
from .metric_terms import StatefulMetric


class StagedMetrics(nn.Module):
    def __init__(self, by_stage: dict[STAGES, StatefulMetric]) -> None:
        self.by_stage = nn.ModuleDict(by_stage)

    def _get_stage(self, stage: STAGES) -> StatefulMetric:
        computer = self.by_stage[stage]
        if not isinstance(computer, StatefulMetric):
            raise TypeError(
                f"Expected MetricComputer for stage '{stage}', got {type(computer)}"
            )
        return computer

    def update(
        self,
        stage: STAGES,
        metric_input: MetricInput,
    ) -> MetricResult:
        return self._get_stage(stage).update(metric_input)

    def compute(self, stage: STAGES) -> MetricResult:
        return self._get_stage(stage).compute()

    def reset(self, stage: STAGES) -> None:
        self._get_stage(stage).reset()


# from torchmetrics.functional.classification import multiclass_jaccard_index

# from ..datatools.label_schema import LabelSchema
# from ..losses_metrics.constraint_function import does_violation_occur_with_wall
# from ..types import MetricInput, MetricResult, WandbOverlay
# from ..visu.helpers import to_label_map


# @dataclass(frozen=True)
# class LabelTriplet:
#     segmentation: torch.Tensor
#     registration: torch.Tensor | None
#     ground_truth: torch.Tensor


# def _label_triplet(
#     metric_input: MetricInput, label_schema: LabelSchema
# ) -> LabelTriplet | None:
#     pred_mask_logits = metric_input.segmentation_logits
#     warped_template = metric_input.warped_template
#     gt_mask = metric_input.gt_mask

#     if pred_mask_logits is None or gt_mask is None:
#         return None

#     return LabelTriplet(
#         segmentation=to_label_map(pred_mask_logits),
#         registration=(
#             label_schema.one_hot_to_label_map(warped_template)
#             if warped_template is not None
#             else None
#         ),
#         ground_truth=label_schema.one_hot_to_label_map(gt_mask),
#     )


# class MetricComputer(nn.Module, ABC):
#     """Base class for configurable metric computation."""

#     def __init__(self) -> None:
#         super().__init__()

#     @abstractmethod
#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         """Compute structured metric output from one batch."""


# class ProjectMetricComputer(MetricComputer):
#     """Project-specific metric computer used by ProjectLightning."""

#     @abstractmethod
#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         """Implement project-specific metrics from the typed MetricInput contract."""


# class NoOpMetricComputer(ProjectMetricComputer):
#     """Default metric computer that emits nothing."""

#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         del metric_input
#         return MetricResult()


# class CompositeMetricComputer(ProjectMetricComputer):
#     """Compose multiple metric computers and merge their MetricResult outputs."""

#     def __init__(self, metric_computers: list[ProjectMetricComputer]) -> None:
#         super().__init__()
#         self.metric_computers = nn.ModuleList(metric_computers)

#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         merged_logs: dict[str, float | torch.Tensor] = {}
#         merged_sum_logs: dict[str, float | torch.Tensor] = {}
#         merged_wandb_overlays: dict[str, WandbOverlay] = {}

#         for metric_module in self.metric_computers:
#             metric_computer = cast(ProjectMetricComputer, metric_module)
#             result = metric_computer.compute(metric_input)

#             if result.logs:
#                 for key, value in result.logs.items():
#                     if key in merged_logs or key in merged_sum_logs:
#                         raise ValueError(
#                             "Duplicate metric log key in CompositeMetricComputer: "
#                             f"{key}"
#                         )
#                     merged_logs[key] = value

#             if result.sum_logs:
#                 for key, value in result.sum_logs.items():
#                     if key in merged_logs or key in merged_sum_logs:
#                         raise ValueError(
#                             "Duplicate metric log key in CompositeMetricComputer: "
#                             f"{key}"
#                         )
#                     merged_sum_logs[key] = value

#             if result.wandb_overlays:
#                 for key, value in result.wandb_overlays.items():
#                     if key in merged_wandb_overlays:
#                         raise ValueError(
#                             "Duplicate metric overlay key in CompositeMetricComputer: "
#                             f"{key}"
#                         )
#                     merged_wandb_overlays[key] = value

#         return MetricResult(
#             logs=merged_logs or None,
#             sum_logs=merged_sum_logs or None,
#             wandb_overlays=merged_wandb_overlays or None,
#         )


# class SegmentationIoUMetricComputer(ProjectMetricComputer):
#     """Compute macro and per-class IoU from predicted and warped label maps."""

#     def __init__(
#         self,
#         label_schema: LabelSchema,
#     ) -> None:
#         super().__init__()
#         self.label_schema = label_schema

#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         labels = _label_triplet(metric_input, self.label_schema)
#         if labels is None:
#             return MetricResult()

#         iou_pred_vs_gt = multiclass_jaccard_index(
#             preds=labels.segmentation,
#             target=labels.ground_truth,
#             num_classes=self.label_schema.num_classes,
#             average="macro",
#         )
#         per_class_iou = multiclass_jaccard_index(
#             preds=labels.segmentation,
#             target=labels.ground_truth,
#             num_classes=self.label_schema.num_classes,
#             average="none",
#         )

#         logs: dict[str, float | torch.Tensor] = {
#             "segmentation/iou/pred_vs_gt": iou_pred_vs_gt,
#         }
#         for class_index, class_iou in enumerate(per_class_iou):
#             class_name = self.label_schema.names.get(
#                 class_index, f"class_{class_index}"
#             )
#             logs[f"segmentation/iou/{class_name}_vs_gt"] = class_iou

#         if labels.registration is not None:
#             logs["registration/iou/warped_vs_gt"] = multiclass_jaccard_index(
#                 preds=labels.registration,
#                 target=labels.ground_truth,
#                 num_classes=self.label_schema.num_classes,
#                 average="macro",
#             )
#             registration_per_class_iou = multiclass_jaccard_index(
#                 preds=labels.registration,
#                 target=labels.ground_truth,
#                 num_classes=self.label_schema.num_classes,
#                 average="none",
#             )
#             for class_index, class_iou in enumerate(registration_per_class_iou):
#                 class_name = self.label_schema.names.get(
#                     class_index, f"class_{class_index}"
#                 )
#                 logs[f"registration/iou/{class_name}_vs_gt"] = class_iou

#         return MetricResult(logs=logs)


# class ConstraintViolationMetricComputer(ProjectMetricComputer):
#     """Count validation predictions that violate the vessel topology constraints."""

#     def __init__(
#         self,
#         label_schema: LabelSchema,
#         stage: str = "val",
#         blob_threshold: int = 50,
#         check_wall_integrity: bool = True,
#     ) -> None:
#         super().__init__()
#         if blob_threshold <= 0:
#             raise ValueError("blob_threshold must be > 0")

#         self.label_schema = label_schema
#         self.stage = stage
#         self.blob_threshold = blob_threshold
#         self.check_wall_integrity = check_wall_integrity

#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         if metric_input.stage != self.stage:
#             return MetricResult()

#         labels = _label_triplet(metric_input, self.label_schema)
#         if labels is None:
#             return MetricResult()

#         logs: dict[str, float | torch.Tensor] = {}
#         sum_logs: dict[str, float | torch.Tensor] = {}
#         self._add_violation_logs(
#             labels=labels.segmentation,
#             metric_prefix="segmentation/constraint",
#             logs=logs,
#             sum_logs=sum_logs,
#         )
#         if labels.registration is not None:
#             self._add_violation_logs(
#                 labels=labels.registration,
#                 metric_prefix="registration/constraint",
#                 logs=logs,
#                 sum_logs=sum_logs,
#             )

#         return MetricResult(logs=logs or None, sum_logs=sum_logs or None)

#     def _add_violation_logs(
#         self,
#         labels: torch.Tensor,
#         metric_prefix: str,
#         logs: dict[str, float | torch.Tensor],
#         sum_logs: dict[str, float | torch.Tensor],
#     ) -> None:
#         sample_count = labels.shape[0]
#         if sample_count == 0:
#             return

#         violating_samples = sum(
#             does_violation_occur_with_wall(
#                 prediction,
#                 label_schema=self.label_schema,
#                 blob_threshold=self.blob_threshold,
#                 check_wall_integrity=self.check_wall_integrity,
#             )[0]
#             for prediction in labels
#         )
#         count_tensor = torch.tensor(float(violating_samples), device=labels.device)
#         total_tensor = torch.tensor(float(sample_count), device=labels.device)
#         logs[f"{metric_prefix}/violation_rate"] = count_tensor / total_tensor
#         sum_logs[f"{metric_prefix}/violating_samples"] = count_tensor
#         sum_logs[f"{metric_prefix}/total_samples"] = total_tensor


# class SegmentationOverlayMetricComputer(ProjectMetricComputer):
#     """Log GT, warped, and predicted label maps as W&B mask overlays."""

#     def __init__(
#         self,
#         label_schema: LabelSchema,
#         stage: str = "val",
#         every_n_epochs: int = 1,
#         sample_indices: list[int] | None = None,
#         image_tag: str = "labels_overlay",
#     ) -> None:
#         super().__init__()
#         if stage not in {"train", "val", "val_extra"}:
#             raise ValueError("stage must be 'train', 'val', or 'val_extra'")
#         if every_n_epochs <= 0:
#             raise ValueError("every_n_epochs must be > 0")
#         if sample_indices is not None and any(idx < 0 for idx in sample_indices):
#             raise ValueError("sample_indices must contain only values >= 0")

#         self.label_schema = label_schema
#         self.stage = stage
#         self.every_n_epochs = every_n_epochs
#         self.sample_indices = sample_indices
#         self.image_tag = image_tag

#     def _resolved_sample_indices(self) -> list[int]:
#         if self.sample_indices is None:
#             return [0]

#         deduped_indices: list[int] = []
#         for idx in self.sample_indices:
#             if idx not in deduped_indices:
#                 deduped_indices.append(int(idx))
#         return deduped_indices

#     def _build_class_labels(self, inferred_num_classes: int) -> dict[int, str]:
#         if inferred_num_classes > self.label_schema.num_classes:
#             raise ValueError("Overlay labels contain a class outside the label schema.")
#         return dict(self.label_schema.names)

#     def _prepare_background_image(
#         self,
#         image_batch: torch.Tensor | None,
#         sample_idx: int,
#         fallback_shape: tuple[int, int],
#     ) -> torch.Tensor:
#         if image_batch is None:
#             return torch.zeros(fallback_shape, dtype=torch.float32)

#         if image_batch.ndim == 4:
#             sample = image_batch[sample_idx]
#         elif image_batch.ndim == 3:
#             sample = image_batch
#         else:
#             raise ValueError(
#                 f"Unsupported image shape for W&B overlays: {tuple(image_batch.shape)}"
#             )

#         sample = sample.detach().cpu().float()
#         if sample.ndim == 3 and sample.shape[0] == 1:
#             return sample[0]
#         if sample.ndim == 3 and sample.shape[0] in (3, 4):
#             return sample[:3].permute(1, 2, 0).contiguous()
#         if sample.ndim == 2:
#             return sample

#         # Unknown channel layout: fall back to first channel.
#         if sample.ndim == 3:
#             return sample[0]
#         return torch.zeros(fallback_shape, dtype=torch.float32)

#     def should_compute(self, metric_input: MetricInput) -> bool:
#         if metric_input.stage != self.stage:
#             return False
#         if metric_input.batch_idx != 0:
#             return False
#         if metric_input.current_epoch is None:
#             return False
#         if (metric_input.current_epoch + 1) % self.every_n_epochs != 0:
#             return False
#         return True

#     def compute(self, metric_input: MetricInput) -> MetricResult:
#         if not self.should_compute(metric_input):
#             return MetricResult()

#         labels = _label_triplet(metric_input, self.label_schema)
#         if labels is None:
#             return MetricResult()

#         batch_sizes = [labels.ground_truth.shape[0], labels.segmentation.shape[0]]
#         if labels.registration is not None:
#             batch_sizes.append(labels.registration.shape[0])
#         batch_size = min(batch_sizes)
#         if batch_size <= 0:
#             return MetricResult()

#         overlays: dict[str, WandbOverlay] = {}
#         for sample_idx in self._resolved_sample_indices():
#             if sample_idx >= batch_size:
#                 continue

#             gt_sample = labels.ground_truth[sample_idx].detach().cpu().long()
#             pred_sample = labels.segmentation[sample_idx].detach().cpu().long()

#             masks = {"ground_truth": gt_sample, "predicted": pred_sample}
#             caption_parts = ["GT"]
#             inferred_label_max = max(gt_sample.max().item(), pred_sample.max().item())
#             if labels.registration is not None:
#                 warped_sample = labels.registration[sample_idx].detach().cpu().long()
#                 masks["warped"] = warped_sample
#                 caption_parts.append("warped")
#                 inferred_label_max = max(inferred_label_max, warped_sample.max().item())
#             caption_parts.append("pred")
#             inferred_classes = int(inferred_label_max + 1)
#             class_labels = self._build_class_labels(inferred_classes)
#             height, width = int(gt_sample.shape[0]), int(gt_sample.shape[1])

#             background = self._prepare_background_image(
#                 image_batch=metric_input.image,
#                 sample_idx=sample_idx,
#                 fallback_shape=(height, width),
#             )

#             overlay = WandbOverlay(
#                 image=background,
#                 masks=masks,
#                 class_labels=class_labels,
#                 caption=(
#                     f"{' | '.join(caption_parts)} | {self.stage} sample={sample_idx}"
#                 ),
#             )
#             overlays[f"{self.image_tag}_{self.stage}_s{sample_idx}"] = overlay

#         if not overlays:
#             return MetricResult()

#         return MetricResult(wandb_overlays=overlays)


# class LabelTripletImageMetricComputer(SegmentationOverlayMetricComputer):
#     """Backward-compatible alias for SegmentationOverlayMetricComputer."""


# class DefaultSegmentationMetricComputer(CompositeMetricComputer):
#     """Precomposed favorite metrics: IoU scalars + W&B segmentation overlays."""

#     def __init__(
#         self,
#         label_schema: LabelSchema,
#         overlay_val_stage: str = "val",
#         overlay_train_stage: str = "train",
#         overlay_every_n_epochs: int = 1,
#         overlay_val_sample_indices: list[int] | None = None,
#         overlay_train_sample_indices: list[int] | None = None,
#         overlay_image_tag: str = "labels_overlay",
#     ) -> None:
#         if overlay_val_sample_indices is None:
#             overlay_val_sample_indices = [0, 1]
#         if overlay_train_sample_indices is None:
#             overlay_train_sample_indices = [0]

#         super().__init__(
#             metric_computers=[
#                 SegmentationIoUMetricComputer(label_schema=label_schema),
#                 ConstraintViolationMetricComputer(
#                     label_schema=label_schema, stage=overlay_val_stage
#                 ),
#                 ConstraintViolationMetricComputer(
#                     label_schema=label_schema, stage="val_extra"
#                 ),
#                 SegmentationOverlayMetricComputer(
#                     label_schema=label_schema,
#                     stage=overlay_val_stage,
#                     every_n_epochs=overlay_every_n_epochs,
#                     sample_indices=overlay_val_sample_indices,
#                     image_tag=overlay_image_tag,
#                 ),
#                 SegmentationOverlayMetricComputer(
#                     label_schema=label_schema,
#                     stage="val_extra",
#                     every_n_epochs=overlay_every_n_epochs,
#                     sample_indices=overlay_val_sample_indices,
#                     image_tag=overlay_image_tag,
#                 ),
#                 SegmentationOverlayMetricComputer(
#                     label_schema=label_schema,
#                     stage=overlay_train_stage,
#                     every_n_epochs=overlay_every_n_epochs,
#                     sample_indices=overlay_train_sample_indices,
#                     image_tag=overlay_image_tag,
#                 ),
#             ]
#         )
