"""Consistent W&B setup for experiment scripts."""

import os
from collections.abc import Mapping
from typing import Any

from pytorch_lightning.loggers import WandbLogger

import wandb


def create_wandb_logger(
    *, config: Mapping[str, Any] | None = None, **logger_kwargs: Any
) -> WandbLogger:
    """Create a W&B logger and attach metadata supplied by the job environment.

    ``CTU_GIT_COMMIT`` is exported by ``remote_submit.sh`` because the synced
    Slurm worktree intentionally has no ``.git`` directory.  When it is not
    set, W&B's normal local Git discovery is left unchanged.
    """
    git_commit = os.environ.get("CTU_GIT_COMMIT")
    settings_kwargs: dict[str, Any] = {"console": "wrap"}
    if git_commit:
        settings_kwargs["git_commit"] = git_commit

    logger = WandbLogger(settings=wandb.Settings(**settings_kwargs), **logger_kwargs)

    if config is not None:
        run_config = dict(config)
        if git_commit:
            run_config["source_git_commit"] = git_commit
        if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
            run_config["slurm_job_id"] = slurm_job_id
        logger.experiment.config.update(run_config, allow_val_change=True)

    return logger
