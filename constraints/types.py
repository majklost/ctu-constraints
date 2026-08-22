from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import torch

if TYPE_CHECKING:
    from .datatools.label_schema import LabelSchema

STAGES = Literal["train", "val", "val_extra", "test"]


@dataclass
class RigidParams:
    angle: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor


@dataclass
class FieldParams:
    field: torch.Tensor


@dataclass
class FieldApplicationResult:
    field: torch.Tensor | None = None
    warped_source: torch.Tensor | None = None
    warped_target: torch.Tensor | None = None


@dataclass
class TransformSpec:
    rigid: RigidParams | None = None
    field: FieldParams | None = None
    steps: tuple["TransformSpec", ...] | None = None
    meta: dict | None = None


@dataclass
class WarpResult:
    warped_template: torch.Tensor
    transform_spec: TransformSpec
    warped_mask: torch.Tensor | None = None
    warped_template_sdf: torch.Tensor | None = None


@dataclass
class DiscreteSegmentation:
    """A discrete segmentation with label IDs as its sole source of truth.

    ``one_hot`` is derived lazily for losses and model paths that require
    channel-wise masks.  It is intentionally not accepted from callers, so it
    cannot diverge from ``labels``.
    """

    labels: torch.Tensor
    label_schema: "LabelSchema"
    _one_hot: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def one_hot(self) -> torch.Tensor:
        if self._one_hot is None:
            self._one_hot = self.label_schema.label_map_to_one_hot(self.labels).float()
        return self._one_hot


@dataclass
class LossInput:
    """Canonical model output contract used by loss computers.

    Keep common outputs explicit and route ablation-specific outputs through
    `extras` to avoid changing call signatures during experiments.
    """

    segmentation_logits: torch.Tensor | None = None
    warped_template: torch.Tensor | None = None
    warped_template_sdf: torch.Tensor | None = None
    gt: DiscreteSegmentation | None = None
    gt_mask_sdf: torch.Tensor | None = None
    transform_spec: TransformSpec | None = None
    extras: dict[str, torch.Tensor] | None = None


@dataclass
class LossResult:
    """Structured loss output with total scalar for backward()."""

    total: torch.Tensor
    components: dict[str, torch.Tensor] | None = None
    logs: dict[str, float | torch.Tensor] | None = None


@dataclass
class MetricInput:
    """Canonical model output contract used by metric computers."""

    image: torch.Tensor
    segmentation_logits: torch.Tensor | None = None
    warped_template: torch.Tensor | None = None
    gt: DiscreteSegmentation | None = None
    gt_mask_sdf: torch.Tensor | None = None
    transform_spec: TransformSpec | None = None
    # Stable dataset IDs in batch order.  Overlay computers use them to select
    # samples independently of DataLoader ordering and batch size; metric terms
    # simply ignore them.
    sample_ids: tuple[str, ...] = ()
    extras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sample_ids and len(self.sample_ids) != self.image.shape[0]:
            raise ValueError(
                "sample_ids must be empty or contain one ID per batch sample; "
                f"got {len(self.sample_ids)} IDs for batch size {self.image.shape[0]}"
            )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique within a metric batch")


@dataclass
class MetricResult:
    scalars: dict[str, torch.Tensor | float] = field(default_factory=dict)
    misc: dict[str, Any] = field(default_factory=dict)
    constraint_violation_samples: dict[str, "ConstraintViolationSamples"] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ConstraintViolationSamples:
    """Violating samples observed in one batch, keyed by a metric prefix."""

    sample_ids: tuple[str, ...]
    details: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.details):
            raise ValueError(
                "ConstraintViolationSamples requires one details entry per sample ID"
            )


@dataclass(frozen=True)
class StepContext:
    stage: STAGES
    batch_idx: int
    current_epoch: int
    global_step: int


@dataclass
class OverlayResult:
    """Generic image payload with one or more semantic mask overlays."""

    image: torch.Tensor
    masks: dict[str, torch.Tensor]
    class_labels: dict[int, str] | None = None
    caption: str | None = None


@dataclass(frozen=True)
class OverlayPolicy:
    """Immutable scheduling and stable-sample selection for an overlay computer.

    An empty collection of overlay computers disables overlays.  A configured
    policy must name its target stages explicitly; it never uses ``None`` as a
    special "all stages" value.
    """

    stages: frozenset[STAGES]
    every_n_epochs: int
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        valid_stages = set(get_args(STAGES))
        if not self.stages:
            raise ValueError("OverlayPolicy.stages must not be empty")
        unknown_stages = set(self.stages) - valid_stages
        if unknown_stages:
            raise ValueError(f"OverlayPolicy has unknown stages: {unknown_stages}")
        if self.every_n_epochs <= 0:
            raise ValueError("OverlayPolicy.every_n_epochs must be > 0")
        if any(not sample_id for sample_id in self.sample_ids):
            raise ValueError("OverlayPolicy.sample_ids must not contain empty IDs")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("OverlayPolicy.sample_ids must be unique")

    def allows(self, context: StepContext) -> bool:
        """Whether an overlay is scheduled for this lifecycle point."""
        return (
            context.stage in self.stages
            and context.current_epoch % self.every_n_epochs == 0
        )

    def batch_positions(self, batch_sample_ids: Sequence[str]) -> tuple[int, ...]:
        """Resolve configured IDs in O(batch_size + configured_samples) time.

        Positions are returned in policy order, keeping artifact output stable
        regardless of DataLoader ordering or batch size.
        """
        positions_by_id = {
            sample_id: position for position, sample_id in enumerate(batch_sample_ids)
        }
        return tuple(
            positions_by_id[sample_id]
            for sample_id in self.sample_ids
            if sample_id in positions_by_id
        )


class LoggingProvider(Protocol):
    def log_loss(
        self, context: StepContext, result: LossResult, *, prog_bar: bool
    ) -> None: ...

    def log_batch(self, context: StepContext, result: MetricResult) -> None:
        pass

    def log_epoch(self, context: StepContext, result: MetricResult) -> None:
        pass

    def log_overlay(
        self, context: StepContext, name: str, result: OverlayResult
    ) -> None: ...
