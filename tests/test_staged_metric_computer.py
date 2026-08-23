import pytest

from constraints.computers.metric_computers import StagedMetricComputer
from constraints.computers.metric_terms import CompositeMetric
from constraints.types import StepContext


def _all_stages() -> dict[str, CompositeMetric]:
    return {
        "train": CompositeMetric([]),
        "val": CompositeMetric([]),
        "val_extra": CompositeMetric([]),
        "test": CompositeMetric([]),
    }


def test_staged_metric_computer_registers_explicit_metric_free_stages() -> None:
    metrics = StagedMetricComputer(_all_stages())

    assert set(metrics.by_stage) == {"train", "val", "val_extra", "test"}
    assert metrics.compute(
        StepContext(stage="test", batch_idx=0, current_epoch=0, global_step=0)
    ).scalars == {}


def test_staged_metric_computer_rejects_missing_stage() -> None:
    metrics = _all_stages()
    del metrics["val_extra"]

    with pytest.raises(ValueError, match=r"missing stages: \['val_extra'\]"):
        StagedMetricComputer(metrics)


def test_staged_metric_computer_rejects_unknown_stage() -> None:
    metrics = _all_stages()
    metrics["invalid"] = CompositeMetric([])

    with pytest.raises(ValueError, match=r"unknown stages: \['invalid'\]"):
        StagedMetricComputer(metrics)
