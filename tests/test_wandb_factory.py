from constraints.logging import wandb_factory


class _FakeConfig:
    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, object], bool]] = []

    def update(self, values: dict[str, object], *, allow_val_change: bool) -> None:
        self.updates.append((values, allow_val_change))


class _FakeLogger:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.experiment = type("Experiment", (), {"config": _FakeConfig()})()


def test_factory_uses_exported_source_metadata(monkeypatch) -> None:
    monkeypatch.setenv("CTU_GIT_COMMIT", "abc123")
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setattr(wandb_factory, "WandbLogger", _FakeLogger)
    monkeypatch.setattr(wandb_factory.wandb, "Settings", lambda **kwargs: kwargs)

    logger = wandb_factory.create_wandb_logger(
        project="test", config={"batch_size": 32}
    )

    assert logger.kwargs == {
        "settings": {"console": "wrap", "git_commit": "abc123"},
        "project": "test",
    }
    assert logger.experiment.config.updates == [
        (
            {
                "batch_size": 32,
                "source_git_commit": "abc123",
                "slurm_job_id": "456",
            },
            True,
        )
    ]


def test_factory_leaves_local_git_discovery_to_wandb(monkeypatch) -> None:
    monkeypatch.delenv("CTU_GIT_COMMIT", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(wandb_factory, "WandbLogger", _FakeLogger)
    monkeypatch.setattr(wandb_factory.wandb, "Settings", lambda **kwargs: kwargs)

    logger = wandb_factory.create_wandb_logger(project="test", config={"seed": 42})

    assert logger.kwargs["settings"] == {"console": "wrap"}
    assert logger.experiment.config.updates == [({"seed": 42}, True)]
