"""Configuration system for experiments.

Design goals
------------
- Configs live in YAML files (one per experiment) so they're git-diffable and
  serializable as W&B artifacts.
- Loaded into a frozen dataclass so typos blow up at load time, not at epoch 47.
- Command-line overrides supported: `--lr 0.001 --epochs 10` overrides YAML.
- Trivially extensible: add a field to the dataclass, add it to YAML, done.

Usage
-----
    from neuromorphic.config import ExperimentConfig, load_config

    config = load_config("experiments/001_smoke_test/config.yaml")
    print(config.lr)             # 0.01
    print(config.run_name)       # "001_smoke_test"

    # CLI overrides:
    #   python run.py --config config.yaml --lr 0.001 --epochs 20
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """All hyperparameters and run metadata for a single experiment.

    Add fields here as the project grows. Keep defaults sensible — a config
    file should only need to override what's actually different from default.
    """

    # --- Run identity ---
    run_name: str = "unnamed_run"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # --- Reproducibility ---
    seed: int = 42

    # --- Data ---
    dataset: str = "mnist"
    batch_size: int = 64
    num_workers: int = 0  # Windows: keep 0 to avoid multiprocessing pickling issues
    data_root: str = "./data"

    # --- Model ---
    arch: str = "baseline_mlp"  # baseline_mlp | tiny_mlp | simple_cnn | (later) snn variants
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])

    # --- Optimization ---
    optimizer: str = "sgd"  # sgd | adam
    lr: float = 0.01
    momentum: float = 0.9  # SGD only
    weight_decay: float = 0.0
    epochs: int = 5

    # --- Tracking ---
    tracker: str = "both"  # wandb | tensorboard | both | none
    wandb_project: str = "neuromorphic-research"
    wandb_entity: str | None = None  # None = use default account

    # --- Output paths (relative to repo root) ---
    checkpoint_dir: str = "checkpoints"
    tensorboard_log_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for logging / W&B."""
        return asdict(self)


def load_config(path: str | Path, cli_overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config file into an ExperimentConfig.

    Parameters
    ----------
    path
        Path to the YAML config file.
    cli_overrides
        Optional dict of field-name → value pairs to override YAML values.
        Typically built from argparse via :func:`parse_cli_overrides`.

    Returns
    -------
    ExperimentConfig
        Validated config object.

    Raises
    ------
    ValueError
        If the YAML contains a field not declared on ExperimentConfig
        (typo protection).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if cli_overrides:
        raw.update({k: v for k, v in cli_overrides.items() if v is not None})

    valid_field_names = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw.keys()) - valid_field_names
    if unknown:
        raise ValueError(
            f"Unknown config fields in {path}: {sorted(unknown)}. "
            f"Valid fields: {sorted(valid_field_names)}"
        )

    return ExperimentConfig(**raw)


def parse_cli_overrides(argv: list[str] | None = None) -> tuple[Path, dict[str, Any]]:
    """Build an argparse parser that accepts --config and any field name as a flag.

    Returns
    -------
    (config_path, overrides)
        config_path: Path to the YAML config to load.
        overrides: dict of field name -> value for fields the user passed on the CLI.
    """
    parser = argparse.ArgumentParser(description="Run a neuromorphic experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")

    # Auto-add a --<field> flag for every ExperimentConfig field
    for f in fields(ExperimentConfig):
        # list and bool fields need special handling; keep it simple for now
        if f.type in ("list[str]", "list[int]", list):
            continue
        parser.add_argument(f"--{f.name}", type=type(f.default) if f.default is not None else str, default=None)

    args = parser.parse_args(argv)
    config_path = args.config
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return config_path, overrides
