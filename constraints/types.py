from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import torch

if TYPE_CHECKING:
    from .computers.loss_terms import LossTerm
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

    ``segmentation_logits`` contains raw, unnormalized model logits.  A loss
    that needs class probabilities must apply softmax itself.  In contrast,
    ``warped_template`` is already a soft class mask produced by interpolation
    and must not be treated as logits or passed through softmax.

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


@dataclass(frozen=True)
class WeightedLossTerm:
    """An explicitly weighted term in a composite optimization objective."""

    weight: float
    term: "LossTerm"


@dataclass
class LossResult:
    """Structured loss output with total scalar for backward()."""

    total: torch.Tensor
    # Weighted contributions, whose sum is ``total``.
    components: dict[str, torch.Tensor] | None = None
    logs: dict[str, float | torch.Tensor] | None = None


@dataclass
class MetricInput:
    """Canonical model output contract used by metric computers.

    ``segmentation_logits`` contains raw, unnormalized model logits, while
    ``warped_template`` is an interpolated soft class mask.  Metrics requiring
    discrete masks convert either representation with ``argmax(dim=1)``.
    """

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
    """Immutable scheduling and sample selection for an overlay computer.

    An empty collection of overlay computers disables overlays.  A configured
    policy must name its target stages explicitly; it never uses ``None`` as a
    special "all stages" value.  Select either stable ``sample_ids`` or the
    first ``first_n_samples`` in the first batch of each scheduled epoch.
    """

    stages: frozenset[STAGES]
    every_n_epochs: int
    sample_ids: tuple[str, ...] = ()
    first_n_samples: int = 0

    def __post_init__(self) -> None:
        valid_stages = set(get_args(STAGES))
        if not self.stages:
            raise ValueError("OverlayPolicy.stages must not be empty")
        unknown_stages = set(self.stages) - valid_stages
        if unknown_stages:
            raise ValueError(f"OverlayPolicy has unknown stages: {unknown_stages}")
        if self.every_n_epochs <= 0:
            raise ValueError("OverlayPolicy.every_n_epochs must be > 0")
        if self.first_n_samples < 0:
            raise ValueError("OverlayPolicy.first_n_samples must be >= 0")
        if self.sample_ids and self.first_n_samples:
            raise ValueError(
                "OverlayPolicy must use either sample_ids or first_n_samples, not both"
            )
        if not self.sample_ids and not self.first_n_samples:
            raise ValueError(
                "OverlayPolicy must configure sample_ids or first_n_samples"
            )
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

    def selects_batch(self, context: StepContext) -> bool:
        """Whether this batch can contain samples selected by the policy."""
        return bool(self.sample_ids) or context.batch_idx == 0

    def batch_positions(self, batch_sample_ids: Sequence[str]) -> tuple[int, ...]:
        """Resolve selected samples to batch positions.

        Stable IDs are returned in policy order.  ``first_n_samples`` returns
        the leading positions in batch order; callers should use
        :meth:`selects_batch` to limit that mode to the first batch of an epoch.
        """
        if self.first_n_samples:
            return tuple(range(min(self.first_n_samples, len(batch_sample_ids))))
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
