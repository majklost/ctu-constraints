import pytest
import torch

from constraints.types import MetricInput, OverlayPolicy, StepContext


def test_overlay_policy_resolves_global_ids_in_policy_order() -> None:
    policy = OverlayPolicy(
        stages=frozenset({"val"}),
        every_n_epochs=2,
        sample_ids=("sample-c", "sample-a", "not-in-batch"),
    )

    assert policy.batch_positions(("sample-a", "sample-b", "sample-c")) == (2, 0)
    assert policy.allows(
        StepContext(stage="val", batch_idx=4, current_epoch=2, global_step=10)
    )
    assert not policy.allows(
        StepContext(stage="train", batch_idx=4, current_epoch=2, global_step=10)
    )


def test_overlay_policy_selects_first_samples_of_the_epoch() -> None:
    policy = OverlayPolicy(
        stages=frozenset({"val"}),
        every_n_epochs=1,
        first_n_samples=2,
    )

    assert policy.batch_positions(("sample-a", "sample-b", "sample-c")) == (0, 1)
    assert policy.selects_batch(
        StepContext(stage="val", batch_idx=0, current_epoch=1, global_step=10)
    )
    assert not policy.selects_batch(
        StepContext(stage="val", batch_idx=1, current_epoch=1, global_step=11)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stages": frozenset()}, "must not be empty"),
        ({"every_n_epochs": 0}, "must be > 0"),
        ({"sample_ids": ("a", "a")}, "must be unique"),
        ({"sample_ids": (), "first_n_samples": 0}, "must configure"),
        ({"sample_ids": ("a",), "first_n_samples": 2}, "either sample_ids"),
    ],
)
def test_overlay_policy_validates_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {
        "stages": frozenset({"val"}),
        "every_n_epochs": 1,
        "sample_ids": ("a",),
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        OverlayPolicy(**defaults)  # type: ignore[arg-type]


def test_metric_input_requires_ids_to_match_the_batch_size() -> None:
    with pytest.raises(ValueError, match="one ID per batch sample"):
        MetricInput(
            image=torch.zeros((2, 1, 4, 4)),
            sample_ids=("only-one",),
        )
