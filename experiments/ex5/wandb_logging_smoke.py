"""Manual smoke run for the generic logging provider.

Run online after authenticating with W&B:

    .venv/bin/python experiments/ex5/wandb_logging_smoke.py --project YOUR_PROJECT

Use ``--mode offline`` to inspect the same payload without contacting W&B.
This script is intentionally not a pytest test.
"""

from argparse import ArgumentParser

import torch
import wandb

from constraints.logging.logging_providers import LightningWandbLoggingProvider
from constraints.types import LossResult, MetricResult, OverlayResult, StepContext


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()

    run = wandb.init(project=args.project, entity=args.entity, mode=args.mode)
    assert run is not None

    def scalar_log(name: str, value: torch.Tensor | float, **_: object) -> None:
        run.log({name: value})

    provider = LightningWandbLoggingProvider(
        scalar_log=scalar_log,
        wandb_run_getter=lambda: run,
    )
    context = StepContext(
        stage="val_extra", batch_idx=0, current_epoch=0, global_step=1
    )
    provider.log_loss(
        context,
        LossResult(
            total=torch.tensor(0.42),
            components={"segmentation/cross_entropy": torch.tensor(0.31)},
        ),
        prog_bar=False,
    )
    provider.log_batch(
        context,
        MetricResult(scalars={"segmentation/iou/pred_vs_gt": torch.tensor(0.76)}),
    )
    provider.log_epoch(
        context,
        MetricResult(scalars={"segmentation/iou/pred_vs_gt": torch.tensor(0.81)}),
    )
    provider.log_overlay(
        context,
        "labels/smoke-sample",
        OverlayResult(
            image=torch.tensor(
                [[0.0, 0.25, 0.5], [0.75, 1.0, 0.5], [0.25, 0.0, 0.75]]
            ),
            masks={
                "ground_truth": torch.tensor([[0, 1, 1], [0, 2, 2], [0, 0, 1]]),
                "predicted": torch.tensor([[0, 1, 2], [0, 2, 2], [0, 0, 1]]),
            },
            class_labels={0: "background", 1: "boundary", 2: "lumen"},
            caption="Logging-provider smoke sample",
        ),
    )
    run.finish()


if __name__ == "__main__":
    main()
