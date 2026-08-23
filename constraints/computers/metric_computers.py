from collections.abc import Mapping
from typing import get_args

from torch import nn

from ..types import STAGES, MetricInput, MetricResult, StepContext
from .metric_terms import StatefulMetric


class StagedMetricComputer(nn.Module):
    """Route stateful metric composites by the stage in a shared context."""

    def __init__(self, by_stage: dict[STAGES, StatefulMetric]) -> None:
        super().__init__()
        expected_stages = set(get_args(STAGES))
        configured_stages = set(by_stage)
        missing_stages = expected_stages - configured_stages
        unknown_stages = configured_stages - expected_stages
        if missing_stages or unknown_stages:
            details = []
            if missing_stages:
                details.append(f"missing stages: {sorted(missing_stages)}")
            if unknown_stages:
                details.append(f"unknown stages: {sorted(unknown_stages)}")
            raise ValueError(
                "StagedMetricComputer requires exactly one metric computer for "
                f"every supported stage ({'; '.join(details)})"
            )
        for stage, metric in by_stage.items():
            if not isinstance(metric, StatefulMetric):
                raise TypeError(
                    f"Expected StatefulMetric for stage '{stage}', got {type(metric)}"
                )
        self._stage_metrics = nn.ModuleDict(
            {self._module_key(stage): metric for stage, metric in by_stage.items()}
        )

    @staticmethod
    def _module_key(stage: STAGES) -> str:
        # `train` is an nn.Module method and therefore cannot itself be a
        # ModuleDict child name.
        return f"stage__{stage}"

    @property
    def by_stage(self) -> Mapping[STAGES, StatefulMetric]:
        """Logical stage mapping; internal ModuleDict keys are PyTorch-safe."""
        return {
            stage: self._stage_metrics[self._module_key(stage)]
            for stage in get_args(STAGES)
        }

    def _get_stage(self, stage: STAGES) -> StatefulMetric:
        computer = self._stage_metrics[self._module_key(stage)]
        if not isinstance(computer, StatefulMetric):
            raise TypeError(
                f"Expected StatefulMetric for stage '{stage}', got {type(computer)}"
            )
        return computer

    def update(
        self,
        context: StepContext,
        metric_input: MetricInput,
    ) -> MetricResult:
        return self._get_stage(context.stage).update(metric_input)

    def compute(self, context: StepContext) -> MetricResult:
        return self._get_stage(context.stage).compute()

    def reset(self, context: StepContext) -> None:
        self._get_stage(context.stage).reset()
