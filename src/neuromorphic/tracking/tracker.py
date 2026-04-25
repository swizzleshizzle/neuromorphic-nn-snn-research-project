"""Unified experiment tracking — W&B and/or TensorBoard.

The ExperimentTracker is a thin wrapper that lets your training loop call
`tracker.log_metric("loss", 0.42, step=1)` without caring whether W&B is
initialized, TensorBoard is writing, both, or neither.

Design goal
-----------
You should never have to write `if use_wandb: wandb.log(...)` in your training
code. The tracker handles the dispatch. Swap trackers by changing the config,
not the code.

Usage
-----
    tracker = ExperimentTracker(config)
    tracker.start()  # initialize W&B run, open TensorBoard writer

    tracker.log_metric("train_loss", 0.5, step=1)
    tracker.log_metrics({"train_loss": 0.4, "test_acc": 0.95}, step=2)
    tracker.log_config()  # log the full config as hyperparameters

    tracker.finish()  # close everything cleanly

    # Or as a context manager:
    with ExperimentTracker(config) as tracker:
        tracker.log_metric(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuromorphic.config import ExperimentConfig


class ExperimentTracker:
    """Multiplexes logging to W&B and/or TensorBoard based on config."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._wandb_run: Any = None
        self._tb_writer: Any = None

        self.use_wandb = config.tracker in ("wandb", "both")
        self.use_tensorboard = config.tracker in ("tensorboard", "both")

    # --- Lifecycle ---

    def start(self) -> "ExperimentTracker":
        """Initialize all configured trackers. Idempotent."""
        if self.use_wandb and self._wandb_run is None:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.run_name,
                    notes=self.config.notes,
                    tags=self.config.tags,
                    config=self.config.to_dict(),
                    reinit=True,
                )
            except ImportError:
                print("[tracker] wandb not installed — skipping W&B logging.")
                self.use_wandb = False
            except Exception as e:
                print(f"[tracker] wandb init failed: {e!r} — continuing without W&B.")
                self.use_wandb = False

        if self.use_tensorboard and self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self.config.tensorboard_log_dir) / self.config.run_name
                log_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                print("[tracker] tensorboard not installed — skipping TB logging.")
                self.use_tensorboard = False

        return self

    def finish(self) -> None:
        """Close all trackers cleanly."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def __enter__(self) -> "ExperimentTracker":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.finish()

    # --- Logging ---

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric."""
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalar metrics in one call."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, global_step=step)

    def log_config(self) -> None:
        """Log the full config (W&B already does this on init; this is for TB)."""
        if self._tb_writer is not None:
            # TB doesn't have a great config logger; dump as text.
            import yaml
            self._tb_writer.add_text("config", f"```yaml\n{yaml.safe_dump(self.config.to_dict())}\n```")
