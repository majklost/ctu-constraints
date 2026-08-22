import torch

from constraints.datatools.label_schema import LabelSchema
from constraints.lightning_wrappers.sample_strategy import AlwaysGt, NoGt, WarmupGt
from constraints.types import StepContext


SCHEMA = LabelSchema.from_lists(
    ["background", "foreground"], [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
)
BATCH = {
    "image": torch.zeros((1, 1, 2, 2)),
    "target_labels": torch.tensor([[[0, 1], [1, 0]]]),
    "sample_id": ["sample-1"],
}


def _context(stage: str, epoch: int) -> StepContext:
    return StepContext(stage=stage, batch_idx=2, current_epoch=epoch, global_step=5)


def test_gt_strategies_accept_a_shared_step_context() -> None:
    assert NoGt(detach_seg=True).decide(BATCH, _context("val", 3)).detach_seg
    assert AlwaysGt(SCHEMA).decide(BATCH, _context("val", 3)).gt is not None

    warmup = WarmupGt(2, SCHEMA, detach_seg=True)
    assert warmup.decide(BATCH, _context("train", 1)).gt is not None
    assert warmup.decide(BATCH, _context("train", 2)).gt is None
    assert warmup.decide(BATCH, _context("val", 0)).gt is None
