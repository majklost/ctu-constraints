import torch
from torchmetrics import Metric

from constraints.losses_metrics.constraint_function import is_annular

from ..datatools.label_schema import LabelSchema
from ..losses_metrics.constraint_function import does_violation_occur_with_wall


class ConstraintViolationCounter(Metric):
    """Distributed-safe counts behind an epoch-level violation rate."""

    full_state_update = False

    def __init__(
        self,
        label_schema: LabelSchema,
        blob_threshold: int,
        check_wall_integrity: bool = True,
        max_ignored_enclosed_background_area: int = 2,
    ) -> None:
        super().__init__()
        self._label_schema = label_schema
        self._blob_threshold = blob_threshold
        self._check_wall_integrity = check_wall_integrity
        self._max_ignored_enclosed_background_area = (
            max_ignored_enclosed_background_area
        )
        self.add_state(
            "violating_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        predictions: torch.Tensor,
        violations: tuple[tuple[bool, tuple[str, ...]], ...] | None = None,
    ) -> None:
        if violations is None:
            violations = self.classify(predictions)
        self.violating_samples += sum(occurred for occurred, _ in violations)
        self.total_samples += predictions.shape[0]

    def classify(
        self, predictions: torch.Tensor
    ) -> tuple[tuple[bool, tuple[str, ...]], ...]:
        if predictions.ndim != 3:
            raise ValueError(
                "Constraint violation predictions must have shape [B, H, W], "
                f"got {tuple(predictions.shape)}"
            )

        violations = tuple(
            (occurred, tuple(details))
            for prediction in predictions
            for occurred, details in [
                does_violation_occur_with_wall(
                    prediction,
                    label_schema=self._label_schema,
                    blob_threshold=self._blob_threshold,
                    check_wall_integrity=self._check_wall_integrity,
                    max_ignored_enclosed_background_area=(
                        self._max_ignored_enclosed_background_area
                    ),
                )
            ]
        )
        return violations

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.violating_samples, self.total_samples


class ACDCAnnularityViolationCounter(Metric):
    """Distributed-safe count of non-annular predicted myocardium masks."""

    full_state_update = False

    def __init__(
        self,
        label_schema: LabelSchema,
        min_hole_area: int = 10,
        min_component_area: int | None = 5,
    ) -> None:
        super().__init__()
        if min_hole_area <= 0:
            raise ValueError("min_hole_area must be > 0")
        if min_component_area is not None and min_component_area <= 0:
            raise ValueError("min_component_area must be > 0 or None")
        myocardium_ids = [
            class_id
            for class_id, name in label_schema.names.items()
            if name == "myocardium"
        ]
        if len(myocardium_ids) != 1:
            raise ValueError(
                "ACDC annularity metrics require exactly one 'myocardium' label."
            )
        self._myocardium_id = myocardium_ids[0]
        self._min_hole_area = min_hole_area
        self._min_component_area = min_component_area
        self.add_state(
            "violating_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def classify(
        self, predictions: torch.Tensor
    ) -> tuple[tuple[bool, tuple[str, ...]], ...]:
        if predictions.ndim != 3:
            raise ValueError(
                "ACDC constraint predictions must have shape [B, H, W], "
                f"got {tuple(predictions.shape)}"
            )

        myocardium_masks = predictions.detach().eq(self._myocardium_id).cpu().numpy()
        violations = []
        for mask in myocardium_masks:
            annular, details = is_annular(
                mask,
                min_hole_area=self._min_hole_area,
                min_component_area=self._min_component_area,
            )
            violations.append((not annular, tuple(details)))
        return tuple(violations)

    def update(
        self,
        predictions: torch.Tensor,
        violations: tuple[tuple[bool, tuple[str, ...]], ...] | None = None,
    ) -> None:
        if violations is None:
            violations = self.classify(predictions)
        self.violating_samples += sum(occurred for occurred, _ in violations)
        self.total_samples += predictions.shape[0]

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.violating_samples, self.total_samples
