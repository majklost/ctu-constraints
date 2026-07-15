# Lightning Logging and Callbacks: Research Workflow Foundation

This project uses one shared Lightning module for both run modes:

1. Quick notebook tuning/debug
2. Reproducible comparison runs

The baseline policy is:

- Module logging contains stable, always-on essentials.
- Callbacks contain optional or debug-only behavior.

## Mental Model

### Module logs (stable essentials)

`ProjectLightning._shared_step(...)` logs:

- `{stage}/loss`: total loss used for optimization
- `{stage}/loss/{component_name}`: structured loss components from `LossResult.components`
- `{stage}/{log_name}`: additional structured metrics from `LossResult.logs`

This keeps train/val metric naming consistent across all experiments.

### Callback logs (optional/debug)

Use callbacks for diagnostics that are not always needed.

Example: `GradientNormLogger` logs `debug/grad_norm` on `on_before_optimizer_step`, with throttling via `every_n_steps`.

## Example: Notebook Tuning with TensorBoard

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from constraints.lightning_wrappers.callbacks import GradientNormLogger

trainer = pl.Trainer(
    max_epochs=5,
    logger=TensorBoardLogger(save_dir="logs", name="notebook_tuning"),
    callbacks=[GradientNormLogger(every_n_steps=20)],
    log_every_n_steps=10,
)

trainer.fit(lightning_module, datamodule=datamodule)
```

Use this mode for fast iteration and debugging in notebooks.

## Example: Comparison Runs with W&B + Checkpointing

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

checkpoint_cb = ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="{epoch}-{step}-{val_loss:.4f}",
)

trainer = pl.Trainer(
    max_epochs=100,
    logger=WandbLogger(project="ctu-constraints-comparisons"),
    callbacks=[checkpoint_cb],
    deterministic=True,
)

trainer.fit(lightning_module, datamodule=datamodule)
```

In comparison mode, `.ckpt` files are the primary reproducibility artifact.

## Example: Gradient Norm Callback Usage

```python
from constraints.lightning_wrappers.callbacks import GradientNormLogger

# Cheap in long runs when throttled.
grad_norm_cb = GradientNormLogger(every_n_steps=50)

trainer = pl.Trainer(
    callbacks=[grad_norm_cb],
)
```

Without `GradientNormLogger`, `debug/grad_norm` is not emitted.

## Lightning Docs

- Callbacks: <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html>
- Logging: <https://lightning.ai/docs/pytorch/stable/extensions/logging.html>
- Trainer: <https://lightning.ai/docs/pytorch/stable/common/trainer.html>
- ModelCheckpoint: <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html>
- Reproducibility: <https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility>
