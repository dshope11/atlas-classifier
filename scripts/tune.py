"""Optuna hyperparameter search for the HWWClassifier.

Drives :func:`src.train.train` programmatically with each trial's hyperparameter
sample, and optimises validation AUC over architecture, dropout, learning rate,
and batch size. Per-trial checkpoints and loss history files are written to
isolated subdirectories under ``data/processed/tune/`` so trials don't clobber
each other; after the study completes, the best trial's checkpoint and history
are promoted to ``config.checkpoint_path`` / ``config.loss_history_path`` so
the standard ``src/evaluate.py`` workflow picks up the winner with no further
config edits.

Per-epoch chatter from the training loop is suppressed during tuning to keep
``logs/tune.log`` focused on per-trial hyperparameters and outcomes.

Usage:
    python scripts/tune.py [--n-trials 60] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import optuna

# Make src.* importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainingConfig, load_config  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import setup_logging  # noqa: E402

LOGGER = logging.getLogger(Path(__file__).stem)

HIDDEN_CHOICES: list[str] = [
    "32", "16-16", "64-32", "32-16", "64-32-16",
    "128-64", "128-64-32", "256-128-64",
]
BATCH_CHOICES: list[int] = [64, 128, 256, 512]


def _parse_hidden(s: str) -> list[int]:
    return [int(x) for x in s.split("-")]


def _trial_paths(base_dir: Path, trial_number: int) -> dict[str, str]:
    trial_dir = base_dir / f"trial_{trial_number}"
    return {
        "checkpoint_path": str(trial_dir / "best_model.pt"),
        "loss_history_path": str(trial_dir / "loss_history.json"),
        "output_dir": str(trial_dir / "eval"),
    }


def _format_params(params: dict[str, Any]) -> str:
    hs = _parse_hidden(params["hidden_sizes"])
    return (
        f"hidden_sizes={hs}  dropout={params['dropout']:.3f}  "
        f"lr={params['lr']:.2e}  batch_size={params['batch_size']}"
    )


def objective(trial: optuna.Trial, base_config: TrainingConfig, tune_dir: Path) -> float:
    params: dict[str, Any] = {
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", HIDDEN_CHOICES),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_CHOICES),
    }
    LOGGER.info("Trial %d | %s", trial.number, _format_params(params))

    paths = _trial_paths(tune_dir, trial.number)
    Path(paths["checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)

    trial_config = dataclasses.replace(
        base_config,
        hidden_sizes=_parse_hidden(params["hidden_sizes"]),
        dropout=params["dropout"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        checkpoint_path=paths["checkpoint_path"],
        loss_history_path=paths["loss_history_path"],
        output_dir=paths["output_dir"],
    )

    try:
        train(trial_config)
    except Exception as exc:
        LOGGER.warning("Trial %d failed during training: %s", trial.number, exc)
        shutil.rmtree(Path(paths["checkpoint_path"]).parent, ignore_errors=True)
        return float("nan")

    history_path = Path(paths["loss_history_path"])
    if not history_path.exists():
        # Training aborted (e.g. non-finite loss) before writing history
        LOGGER.warning("Trial %d: no history written - likely non-finite loss", trial.number)
        return float("nan")

    with history_path.open() as f:
        history = json.load(f)
    if not history:
        return float("nan")

    best_auc = max(float(epoch["val_auc"]) for epoch in history)
    LOGGER.info("Trial %d | val_auc=%.4f", trial.number, best_auc)
    return best_auc



def _trial_value(t: optuna.trial.FrozenTrial) -> float:
    """Return ``t.value`` as a finite float, or ``-inf`` for failed/missing trials.

    Used as a sort key over study.trials so that pre-filtered trials sort safely.
    """
    v = t.value
    return v if v is not None and not math.isnan(v) else -math.inf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--config", type=str, default="config.yaml", help="Base config path")
    parser.add_argument(
        "--log-path", type=str, default="logs/tune.log", help="Path for tuning log file"
    )
    parser.add_argument(
        "--keep-all-trials",
        action="store_true",
        help="Keep all trial directories; default deletes non-best dirs at end",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)

    setup_logging(args.log_path)
    # Mute per-epoch chatter from inner modules so the tune log stays focused.
    for noisy in ("train", "preprocessing", "data_loading", "model"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tune_dir = Path("data/processed/tune")
    tune_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 70)
    LOGGER.info("Starting Optuna study: n_trials=%d", args.n_trials)
    LOGGER.info("  hidden_sizes : %s", HIDDEN_CHOICES)
    LOGGER.info("  dropout      : uniform(0.1, 0.5)")
    LOGGER.info("  lr           : log-uniform(1e-4, 1e-2)")
    LOGGER.info("  batch_size   : %s", BATCH_CHOICES)
    LOGGER.info("=" * 70)

    sampler = optuna.samplers.TPESampler(seed=base_config.random_seed)
    study = optuna.create_study(direction="maximize", study_name="hww-tune", sampler=sampler)

    best_so_far: dict[str, float] = {"value": -math.inf, "trial": -1}

    def progress_cb(_study: optuna.Study, t: optuna.trial.FrozenTrial) -> None:
        v = _trial_value(t)
        if v > best_so_far["value"]:
            best_so_far["value"] = v
            best_so_far["trial"] = t.number
        if best_so_far["trial"] >= 0:
            LOGGER.info(
                "Trial %d done | best so far: %.4f @ trial %d",
                t.number, best_so_far["value"], int(best_so_far["trial"]),
            )

    study.optimize(
        lambda t: objective(t, base_config, tune_dir),
        n_trials=args.n_trials,
        callbacks=[progress_cb],
    )

    LOGGER.info("=" * 70)
    LOGGER.info("Study complete - %d trials", len(study.trials))

    valid_trials = sorted(
        (t for t in study.trials if t.value is not None and not math.isnan(t.value)),
        key=_trial_value,
        reverse=True,
    )

    if not valid_trials:
        LOGGER.error("No trials completed successfully - nothing to promote")
        return 1

    LOGGER.info("Top %d trials:", min(5, len(valid_trials)))
    for t in valid_trials[:5]:
        LOGGER.info("  trial %3d  AUC=%.4f  %s", t.number, _trial_value(t), _format_params(t.params))

    best = valid_trials[0]
    best_paths = _trial_paths(tune_dir, best.number)
    target_ckpt = Path(base_config.checkpoint_path)
    target_hist = Path(base_config.loss_history_path)
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_paths["checkpoint_path"], target_ckpt)
    shutil.copy2(best_paths["loss_history_path"], target_hist)
    LOGGER.info("Promoted trial %d checkpoint -> %s", best.number, target_ckpt)
    LOGGER.info("Promoted trial %d history    -> %s", best.number, target_hist)

    hs = _parse_hidden(best.params["hidden_sizes"])
    LOGGER.info(
        "Best config (paste into config.yaml):\n"
        "  hidden_sizes: %s\n"
        "  dropout: %.3f\n"
        "  lr: %.4e\n"
        "  batch_size: %d",
        hs, best.params["dropout"], best.params["lr"], best.params["batch_size"],
    )

    if not args.keep_all_trials:
        for t in study.trials:
            if t.number == best.number:
                continue
            td = tune_dir / f"trial_{t.number}"
            if td.exists():
                shutil.rmtree(td)
        LOGGER.info("Cleaned up non-best trial directories")

    return 0


if __name__ == "__main__":
    sys.exit(main())
