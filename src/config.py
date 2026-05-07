"""Configuration dataclass and YAML loader for the atlas-classifier pipeline.

Single source of truth for all hyperparameters. ``TrainingConfig`` validates its
fields in ``__post_init__`` so bad config is caught at load time, not mid-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Features: TypeAlias = np.ndarray  # shape (n_events, n_features)
Labels: TypeAlias = np.ndarray  # shape (n_events,) — 0/1
Weights: TypeAlias = np.ndarray  # shape (n_events,) — physics event weights
BatchDict: TypeAlias = "dict[str, torch.Tensor]"


# ---------------------------------------------------------------------------
# TrainingConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All knobs for the training pipeline.

    Fields are loaded from ``config.yaml`` via ``load_config()``. Defaults match
    the plan in ``CLAUDE.md`` and can be overridden by the YAML file.
    """

    # --- Architecture ----------------------------------------------------
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    input_noise_std: float = 0.01  # GaussianNoise stddev

    # --- Training --------------------------------------------------------
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 15  # early stopping patience (epochs without val_loss improvement)
    batch_size: int = 256
    working_point_signal_eff: float = 0.30  # per-epoch val Z diagnostic during training only; not used in evaluate.py

    # --- Data ------------------------------------------------------------
    lumi: float = 36.1  # fb⁻¹ — see notes/data_sources.md for choice rationale
    train_val_test_split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42

    # --- LR scheduling (ReduceLROnPlateau) -------------------------------
    lr_patience: int = 10
    lr_factor: float = 0.5

    # --- Event selection cuts (nested DSL — see src/utils.py:evaluate_cuts) -
    cuts: dict[str, Any] = field(default_factory=dict)
    # Cut-based baseline applied to test-set features for Asimov Z comparison.
    # Variable names must match feature names in split.h5 (composite names).
    baseline_cuts: dict[str, Any] = field(default_factory=dict)

    # --- Paths -----------------------------------------------------------
    raw_dir: str = "data/raw"
    processed_path: str = "data/processed/events.h5"
    split_path: str = "data/processed/split.h5"
    checkpoint_path: str = "data/processed/best_model.pt"
    loss_history_path: str = "data/processed/loss_history.json"
    log_path: str = "logs/training.log"
    output_dir: str = "data/processed/eval"

    def __post_init__(self) -> None:
        # Splits must form a valid 3-way partition
        if not abs(sum(self.train_val_test_split) - 1.0) < 1e-9:
            raise ValueError(
                f"train_val_test_split must sum to 1.0, got "
                f"{self.train_val_test_split} (sum={sum(self.train_val_test_split)})"
            )
        if any(f <= 0 for f in self.train_val_test_split):
            raise ValueError(
                f"All split fractions must be positive, got {self.train_val_test_split}"
            )

        # Architecture
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if len(self.hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer")
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes entries must be positive, got {self.hidden_sizes}")

        # Training
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not (0.0 < self.working_point_signal_eff < 1.0):
            raise ValueError(
                f"working_point_signal_eff must be in (0, 1), got {self.working_point_signal_eff}"
            )

        # Data
        if self.lumi <= 0:
            raise ValueError(f"lumi must be positive, got {self.lumi}")

        # LR scheduling
        if not (0.0 < self.lr_factor < 1.0):
            raise ValueError(f"lr_factor must be in (0, 1), got {self.lr_factor}")
        if self.lr_patience <= 0:
            raise ValueError(f"lr_patience must be positive, got {self.lr_patience}")


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path = "config.yaml") -> TrainingConfig:
    """Load and validate a ``TrainingConfig`` from a YAML file.

    Unknown keys raise a ``ValueError`` — this catches typos in config.yaml at
    load time rather than silently ignoring them.
    """
    path = Path(path)
    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    valid_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_fields)}"
        )

    # YAML loads tuples as lists — convert split back to tuple
    if "train_val_test_split" in raw:
        raw["train_val_test_split"] = tuple(raw["train_val_test_split"])

    return TrainingConfig(**raw)
