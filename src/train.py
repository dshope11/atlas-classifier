"""Training loop for the HWWClassifier.

Loads the split produced by :mod:`src.preprocessing`, builds the model with
``RobustScaler`` stats from the train split (in-model normalisation via
``register_buffer``), and trains with:

* ``BCEWithLogitsLoss`` with counts-based ``pos_weight = n_bg / n_sig`` -
  no physics weights in the loss (clean separation principle).
* Adam optimiser; ``ReduceLROnPlateau`` LR schedule on val loss.
* Early stopping on val loss (restores best weights).
* NaN/instability guard - halts training if the loss becomes non-finite.
* Per-epoch diagnostics: val loss, val AUC, **val Asimov significance**
  at the configured working point computed with physics weights - diagnostic
  only, never influences early stopping or gradient updates.

Saves a self-describing checkpoint (state_dict + feature_names + hyperparams)
plus a JSON loss/significance/AUC history for the twin-axis training curve.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Make src.* importable when this module is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainingConfig, load_config  # noqa: E402
from src.model import HWWClassifier  # noqa: E402
from src.preprocessing import load_split  # noqa: E402
from src.utils import asimov_significance, chronomat, print_timings, setup_logging  # noqa: E402

LOGGER = logging.getLogger(Path(__file__).stem)


def _select_device() -> torch.device:
    """Prefer MPS (Apple Silicon) -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _val_pass(
    model: HWWClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one full pass on ``loader`` in eval mode. Returns (avg_loss, logits, labels)."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits: list[Tensor] = []
    all_y: list[Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb_d, yb_d = xb.to(device), yb.to(device)
            logits = model(xb_d)
            loss = criterion(logits, yb_d)
            total_loss += loss.item() * len(xb_d)
            n += len(xb_d)
            all_logits.append(logits.cpu())
            all_y.append(yb_d.cpu())
    avg_loss = total_loss / n if n > 0 else float("nan")
    logits_arr = torch.cat(all_logits).numpy().flatten()
    y_arr = torch.cat(all_y).numpy().flatten()
    return avg_loss, logits_arr, y_arr


def _val_significance(
    logits: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    signal_eff_target: float,
) -> float:
    """Asimov Z at a threshold giving ``signal_eff_target`` (unweighted) on signal.

    The threshold is set so that ``signal_eff_target`` fraction of signal events
    pass (counts-based - simple working-point definition). Yields s and b on
    that side of the cut are computed with full physics weights ``w``.
    """
    proba = 1.0 / (1.0 + np.exp(-logits))
    sig_mask = y == 1
    sig_proba = proba[sig_mask]
    if len(sig_proba) == 0:
        return 0.0
    threshold = float(np.quantile(sig_proba, 1.0 - signal_eff_target))
    pass_mask = proba >= threshold
    s = float(w[pass_mask & sig_mask].sum())
    b = float(w[pass_mask & ~sig_mask].sum())
    return asimov_significance(s, b)


@chronomat
def train(config: TrainingConfig) -> None:
    """Run the full training loop and save the best checkpoint + history."""
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = _select_device()
    LOGGER.info("Device: %s", device)

    split = load_split(config.split_path)
    X_train = cast(np.ndarray, split["X_train"])
    y_train = cast(np.ndarray, split["y_train"])
    X_val = cast(np.ndarray, split["X_val"])
    y_val = cast(np.ndarray, split["y_val"])
    w_val = cast(np.ndarray, split["w_val"])
    feature_names = cast(list[str], split["feature_names"])
    median = cast(np.ndarray, split["scaler_median"])
    iqr = cast(np.ndarray, split["scaler_iqr"])

    # Counts-based class balance - see notes/data_sources.md and the README
    # weight-design discussion for why we don't put physics weights here.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    LOGGER.info(
        "Train counts - signal=%d  background=%d  pos_weight=%.4f",
        n_pos, n_neg, pos_weight.item(),
    )

    model = HWWClassifier(config, feature_names, median, iqr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )

    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, config.batch_size * 4, shuffle=False)
    LOGGER.info(
        "Loaders ready  train_batches=%d  val_batches=%d",
        len(train_loader), len(val_loader),
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb_d, yb_d = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb_d)
            loss = criterion(logits, yb_d)
            if not torch.isfinite(loss):
                LOGGER.error(
                    "Non-finite loss at epoch %d (loss=%s) - halting training",
                    epoch, loss.item(),
                )
                return
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb_d)
            train_n += len(xb_d)
        train_loss = train_loss_sum / train_n

        val_loss, val_logits, val_y = _val_pass(model, val_loader, criterion, device)
        val_auc = float(roc_auc_score(val_y, val_logits))
        val_z = _val_significance(val_logits, val_y, w_val, config.working_point_signal_eff)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            LOGGER.info("LR reduced: %.2e -> %.2e", prev_lr, new_lr)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": val_auc,
            "val_significance": val_z,
            "lr": float(new_lr),
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        marker = "(best)" if improved else f"({epochs_no_improve}/{config.patience})"
        LOGGER.info(
            "epoch %3d  train=%.5f  val=%.5f  AUC=%.4f  Z=%.3f  %s",
            epoch, train_loss, val_loss, val_auc, val_z, marker,
        )

        if epochs_no_improve >= config.patience:
            LOGGER.info(
                "Early stopping at epoch %d (best epoch: %d, best val_loss: %.5f)",
                epoch, best_epoch, best_val_loss,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        LOGGER.info("Restored best weights from epoch %d", best_epoch)

    # Save checkpoint
    ckpt_path = Path(config.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model.cpu()
    ckpt = model.make_checkpoint(config)
    ckpt["best_epoch"] = best_epoch
    ckpt["best_val_loss"] = best_val_loss
    torch.save(ckpt, ckpt_path)
    LOGGER.info("Saved checkpoint: %s", ckpt_path)

    # Save loss history
    history_path = Path(config.loss_history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    LOGGER.info("Saved loss history: %s", history_path)


def main() -> int:
    config = load_config("config.yaml")
    setup_logging(config.log_path)
    train(config)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
