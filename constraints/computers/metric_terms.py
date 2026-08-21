from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn
from torchmetrics import JaccardIndex

from ..datatools.label_schema import LabelSchema
from ..losses_metrics.metrics import ConstraintViolationCounter
from ..types import ConstraintViolationSamples, MetricInput, MetricResult


class StatefulMetric(nn.Module, ABC):
    @abstractmethod
    def update(self, inputs: MetricInput) -> MetricResult:
        """Observe one batch, update state, optionally emit batch metrics."""

    @abstractmethod
    def compute(self) -> MetricResult:
        """Return epoch-aggregated metrics."""

    @abstractmethod
    def reset(self) -> None:
        """Clear state after an epoch."""


class MetricTerm(StatefulMetric):
    def __init__(self, label_schema: LabelSchema) -> None:
        super().__init__()
        self._label_schema = label_schema


class CompositeMetric(StatefulMetric):
    """Compose stateful metrics and merge their scalar outputs."""

    def __init__(
        self,
        metrics: list[StatefulMetric],
        prefix: str | None = None,
    ) -> None:
        super().__init__()
        self._metrics = nn.ModuleList(metrics)
        normalized_prefix = prefix.strip("/") if prefix else ""
        if prefix is not None and not normalized_prefix:
            raise ValueError("CompositeMetric prefix must contain non-slash characters")
        self._prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    def update(self, inputs: MetricInput) -> MetricResult:
        return self._merge(
            (type(metric).__name__, metric.update(inputs)) for metric in self._metrics
        )

    def compute(self) -> MetricResult:
        return self._merge(
            (type(metric).__name__, metric.compute()) for metric in self._metrics
        )

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def _merge(self, results: Iterable[tuple[str, MetricResult]]) -> MetricResult:
        scalars: dict[str, torch.Tensor | float] = {}
        scalar_sources: dict[str, str] = {}
        misc: dict[str, object] = {}
        constraint_violation_samples: dict[str, ConstraintViolationSamples] = {}

        for source, result in results:
            for name, value in result.scalars.items():
                if not name:
                    raise ValueError(f"{source} emitted an empty metric scalar name")
                key = f"{self._prefix}{name}"
                if key in scalars:
                    raise ValueError(
                        f"Duplicate metric scalar name '{key}' emitted by "
                        f"{scalar_sources[key]} and {source}"
                    )
                scalars[key] = value
                scalar_sources[key] = source

            for name, value in result.misc.items():
                if name in misc:
                    raise ValueError(
                        f"Duplicate metric misc key '{name}' emitted by {source} "
                        "and another metric"
                    )
                misc[name] = value

            for prefix, samples in result.constraint_violation_samples.items():
                key = f"{self._prefix}{prefix}"
                if key in constraint_violation_samples:
                    raise ValueError(
                        f"Duplicate constraint-violation sample key '{key}' emitted "
                        f"by {source} and another metric"
                    )
                constraint_violation_samples[key] = samples

        return MetricResult(
            scalars=scalars,
            misc=misc,
            constraint_violation_samples=constraint_violation_samples,
        )


class SegmentationIoUTerm(MetricTerm):
    """Dataset-level IoU between segmentation logits and ground truth."""

    def __init__(self, label_schema: LabelSchema) -> None:
        super().__init__(label_schema)
        self._macro_iou = JaccardIndex(
            task="multiclass",
            num_classes=label_schema.num_classes,
            average="macro",
            ignore_index=label_schema.ignore_index,
        )
        self._per_class_iou = JaccardIndex(
            task="multiclass",
            num_classes=label_schema.num_classes,
            average="none",
            ignore_index=label_schema.ignore_index,
        )

    def update(self, inputs: MetricInput) -> MetricResult:
        if inputs.segmentation_logits is None or inputs.gt is None:
            return MetricResult(scalars={})

        predictions = inputs.segmentation_logits.detach().argmax(dim=1)
        targets = inputs.gt.labels.detach()
        self._macro_iou.update(predictions, targets)
        self._per_class_iou.update(predictions, targets)
        return MetricResult(scalars={})

    def compute(self) -> MetricResult:
        scalars: dict[str, torch.Tensor | float] = {
            "segmentation/iou/pred_vs_gt": self._macro_iou.compute()
        }
        for class_index, class_iou in enumerate(self._per_class_iou.compute()):
            class_name = self._label_schema.names.get(
                class_index, f"class_{class_index}"
            )
            scalars[f"segmentation/iou/{class_name}_vs_gt"] = class_iou
        return MetricResult(scalars=scalars)

    def reset(self) -> None:
        self._macro_iou.reset()
        self._per_class_iou.reset()


class RegistrationIoUTerm(MetricTerm):
    """Dataset-level IoU between the warped template and ground truth."""

    def __init__(self, label_schema: LabelSchema) -> None:
        super().__init__(label_schema)
        self._macro_iou = JaccardIndex(
            task="multiclass",
            num_classes=label_schema.num_classes,
            average="macro",
            ignore_index=label_schema.ignore_index,
        )
        self._per_class_iou = JaccardIndex(
            task="multiclass",
            num_classes=label_schema.num_classes,
            average="none",
            ignore_index=label_schema.ignore_index,
        )

    def update(self, inputs: MetricInput) -> MetricResult:
        if inputs.warped_template is None or inputs.gt is None:
            return MetricResult(scalars={})

        predictions = inputs.warped_template.detach().argmax(dim=1)
        targets = inputs.gt.labels.detach()
        self._macro_iou.update(predictions, targets)
        self._per_class_iou.update(predictions, targets)
        return MetricResult(scalars={})

    def compute(self) -> MetricResult:
        scalars: dict[str, torch.Tensor | float] = {
            "registration/iou/warped_vs_gt": self._macro_iou.compute()
        }
        for class_index, class_iou in enumerate(self._per_class_iou.compute()):
            class_name = self._label_schema.names.get(
                class_index, f"class_{class_index}"
            )
            scalars[f"registration/iou/{class_name}_vs_gt"] = class_iou
        return MetricResult(scalars=scalars)

    def reset(self) -> None:
        self._macro_iou.reset()
        self._per_class_iou.reset()


class _ConstraintViolationTerm(MetricTerm, ABC):
    """Shared stateful implementation for constraint-violation metrics."""

    metric_prefix: str

    def __init__(
        self,
        label_schema: LabelSchema,
        blob_threshold: int = 50,
        check_wall_integrity: bool = True,
        track_violating_samples: bool = False,
    ) -> None:
        super().__init__(label_schema)
        if blob_threshold <= 0:
            raise ValueError("blob_threshold must be > 0")
        self._counter = ConstraintViolationCounter(
            label_schema=label_schema,
            blob_threshold=blob_threshold,
            check_wall_integrity=check_wall_integrity,
        )
        self._track_violating_samples = track_violating_samples

    @abstractmethod
    def _prediction_labels(self, inputs: MetricInput) -> torch.Tensor | None:
        """Return discrete [B, H, W] predictions, or None when unavailable."""

    def update(self, inputs: MetricInput) -> MetricResult:
        predictions = self._prediction_labels(inputs)
        if predictions is None:
            return MetricResult(scalars={})

        if not self._track_violating_samples:
            self._counter.update(predictions.detach())
            return MetricResult()
        if not inputs.sample_ids:
            raise ValueError(
                "track_violating_samples=True requires MetricInput.sample_ids"
            )

        violations = self._counter.classify(predictions.detach())
        self._counter.update(predictions.detach(), violations=violations)

        violating_indices = [
            index for index, (occurred, _) in enumerate(violations) if occurred
        ]
        if not violating_indices:
            return MetricResult()
        return MetricResult(
            constraint_violation_samples={
                self.metric_prefix: ConstraintViolationSamples(
                    sample_ids=tuple(inputs.sample_ids[index] for index in violating_indices),
                    details=tuple(violations[index][1] for index in violating_indices),
                )
            }
        )

    def compute(self) -> MetricResult:
        if not self._counter.update_called:
            return MetricResult(scalars={})
        violating_samples, total_samples = self._counter.compute()
        if total_samples.item() == 0:
            return MetricResult(scalars={})
        return MetricResult(
            scalars={
                f"{self.metric_prefix}/violation_rate": (
                    violating_samples.float() / total_samples
                )
            }
        )

    def reset(self) -> None:
        self._counter.reset()


class SegmentationConstraintViolationTerm(_ConstraintViolationTerm):
    metric_prefix = "segmentation/constraint"

    def _prediction_labels(self, inputs: MetricInput) -> torch.Tensor | None:
        if inputs.segmentation_logits is None:
            return None
        return inputs.segmentation_logits.argmax(dim=1)


class RegistrationConstraintViolationTerm(_ConstraintViolationTerm):
    metric_prefix = "registration/constraint"

    def _prediction_labels(self, inputs: MetricInput) -> torch.Tensor | None:
        if inputs.warped_template is None:
            return None
        return inputs.warped_template.argmax(dim=1)
