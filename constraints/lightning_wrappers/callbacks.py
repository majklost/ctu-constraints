"""
Various callbacks for Lightning Wrappers.
- debugging, inspection
- saving runs, etc...
"""
import math

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class GradientNormLogger(Callback):
    """Logs global gradient norm under `debug/grad_norm`.

    This callback is optional and intended for quick tuning/debug sessions.
    Throttling keeps overhead low in longer runs.
    """

    def __init__(self, every_n_steps: int = 50) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be > 0")
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer  # Callback computes across all module parameters.

        if trainer.global_step % self.every_n_steps != 0:
            return

        sq_norm_sum = 0.0
        has_grad = False
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().data.norm(2)
            sq_norm_sum += float(grad_norm.item()) ** 2
            has_grad = True

        if not has_grad:
            return

        total_norm = math.sqrt(sq_norm_sum)
        pl_module.log("debug/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False)
