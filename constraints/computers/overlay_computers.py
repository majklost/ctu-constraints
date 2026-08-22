from abc import ABC, abstractmethod

import torch

from ..datatools.label_schema import LabelSchema
from ..types import MetricInput, OverlayPolicy, OverlayResult, StepContext


class OverlayComputer(ABC):
    """Create generic, display-ready overlays from one semantic metric input."""

    @abstractmethod
    def compute(
        self, metric_input: MetricInput, context: StepContext
    ) -> dict[str, OverlayResult]:
        """Return artifact-name to overlay mappings for the current batch."""


class SegmentationOverlayComputer(OverlayComputer):
    """Create GT, segmentation, and optional registration label overlays."""

    def __init__(
        self,
        label_schema: LabelSchema,
        policy: OverlayPolicy,
        artifact_name: str = "labels",
    ) -> None:
        normalized_name = artifact_name.strip("/")
        if not normalized_name:
            raise ValueError("artifact_name must contain non-slash characters")
        self._label_schema = label_schema
        self._policy = policy
        self._artifact_name = normalized_name

    def compute(
        self, metric_input: MetricInput, context: StepContext
    ) -> dict[str, OverlayResult]:
        if not self._policy.allows(context) or not metric_input.sample_ids:
            return {}

        positions = self._policy.batch_positions(metric_input.sample_ids)
        if not positions:
            return {}

        segmentation = self._label_maps(metric_input.segmentation_logits)
        registration = self._label_maps(metric_input.warped_template)
        ground_truth = metric_input.gt.labels if metric_input.gt is not None else None
        if segmentation is None and registration is None and ground_truth is None:
            return {}

        overlays: dict[str, OverlayResult] = {}
        for position in positions:
            masks: dict[str, torch.Tensor] = {}
            if ground_truth is not None:
                masks["ground_truth"] = ground_truth[position].detach().cpu().long()
            if registration is not None:
                masks["warped"] = registration[position]
            if segmentation is not None:
                masks["predicted"] = segmentation[position]
            if not masks:
                continue

            sample_id = metric_input.sample_ids[position]
            overlays[f"{self._artifact_name}/{sample_id}"] = OverlayResult(
                image=self._display_image(metric_input.image, position),
                masks=masks,
                class_labels=dict(self._label_schema.names),
                caption=self._caption(context, sample_id, masks),
            )
        return overlays

    def _label_maps(self, channels: torch.Tensor | None) -> torch.Tensor | None:
        if channels is None:
            return None
        if channels.ndim != 4:
            raise ValueError(
                "Overlay channels must have shape [B, C, H, W], "
                f"got {tuple(channels.shape)}"
            )
        if channels.shape[1] != self._label_schema.num_classes:
            raise ValueError(
                "Overlay channels do not match the label schema: expected "
                f"{self._label_schema.num_classes}, got {channels.shape[1]}"
            )
        return channels.detach().argmax(dim=1).cpu().long()

    @staticmethod
    def _display_image(image_batch: torch.Tensor, position: int) -> torch.Tensor:
        if image_batch.ndim != 4:
            raise ValueError(
                "Overlay images must have shape [B, C, H, W], "
                f"got {tuple(image_batch.shape)}"
            )
        image = image_batch[position].detach().cpu().float()
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[0] >= 3:
            image = image[:3].movedim(0, -1)
        else:
            image = image[0]

        image_min = image.amin()
        image_max = image.amax()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        return image.clamp(0.0, 1.0)

    @staticmethod
    def _caption(
        context: StepContext, sample_id: str, masks: dict[str, torch.Tensor]
    ) -> str:
        labels = " | ".join(masks)
        return f"{context.stage} | sample={sample_id} | {labels}"
