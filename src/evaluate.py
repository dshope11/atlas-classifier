"""Full evaluation suite for the trained HWWClassifier.

Outputs (all written under ``config.output_dir`` as PNGs + a printed summary):

**Pre-fit diagnostics** (from the split, before model is touched):
- ``features_signal_vs_background.png`` — overlaid normalised distributions
- ``feature_correlation.png`` — heatmap, computed on signal events

**Post-fit**:
- ``training_curves.png`` — twin-axis: val loss (left) + val Asimov Z (right)
- ``roc.png`` — ROC + AUC with Clopper–Pearson 1σ bands and the cut-based
  working point as a single marker
- ``score_distributions.png`` — signal vs background, train/test overlaid,
  weighted by physics ``event_weight``
- ``feature_importance.png`` — permutation (global) and perturbation (local
  ±0.01σ) side by side; disagreement is interpretable

**Printed summary**: AUC, KS overtraining check (p-values), cut-based and DNN
Asimov significance at the working point.

All yields/significance use full physics weights ``event_weight``. ROC and
KS are unweighted (measure the discriminator quality / shape match, not yields).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import beta, ks_2samp
from sklearn.metrics import auc, roc_curve

# Make src.* importable when this module is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainingConfig, load_config  # noqa: E402
from src.model import HWWClassifier  # noqa: E402
from src.preprocessing import load_split  # noqa: E402
from src.utils import asimov_significance, chronomat, print_timings, setup_logging  # noqa: E402

LOGGER = logging.getLogger(Path(__file__).stem)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _score(model: HWWClassifier, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Return p(signal) for every row of ``X`` in eval mode."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float()
            logits = model(batch).cpu().numpy().flatten()
            out.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(out)


def _threshold_at_signal_eff(scores_signal: np.ndarray, eff: float) -> float:
    """Score threshold giving ``eff`` fraction of signal events above it (unweighted)."""
    return float(np.quantile(scores_signal, 1.0 - eff))


def _yields(scores: np.ndarray, y: np.ndarray, w: np.ndarray, threshold: float) -> tuple[float, float]:
    """Sum of physics weights for signal/background events with score >= threshold."""
    pass_mask = scores >= threshold
    s = float(w[pass_mask & (y == 1)].sum())
    b = float(w[pass_mask & (y == 0)].sum())
    return s, b


# ---------------------------------------------------------------------------
# Pre-fit plots
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], out_path: Path
) -> None:
    """Per-feature signal vs background, area-normalised."""
    n_feat = X.shape[1]
    n_cols = min(3, n_feat)
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    sig_mask = y == 1
    bkg_mask = y == 0
    for i, name in enumerate(feature_names):
        ax = axes[i // n_cols][i % n_cols]
        x_sig = X[sig_mask, i]
        x_bkg = X[bkg_mask, i]
        # Use the common range so the two histograms are comparable
        lo = float(np.percentile(np.concatenate([x_sig, x_bkg]), 1))
        hi = float(np.percentile(np.concatenate([x_sig, x_bkg]), 99))
        bins = np.linspace(lo, hi, 50)
        ax.hist(x_sig, bins=bins, density=True, histtype="step", color="C0", label="Signal", linewidth=1.5)
        ax.hist(x_bkg, bins=bins, density=True, histtype="step", color="C3", label="Background", linewidth=1.5)
        ax.set_xlabel(name)
        ax.set_ylabel("a.u.")
        ax.legend(fontsize=8)
    # Hide unused subplots
    for j in range(n_feat, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correlation_matrix(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], out_path: Path
) -> None:
    """Pearson correlation heatmap, computed on signal events only."""
    sig = X[y == 1]
    corr = np.corrcoef(sig, rowvar=False)
    fig, ax = plt.subplots(figsize=(0.7 * len(feature_names) + 2, 0.7 * len(feature_names) + 1))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        xticklabels=feature_names, yticklabels=feature_names, ax=ax, square=True,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Feature correlation (signal events)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-fit plots
# ---------------------------------------------------------------------------


def plot_training_curves(history_path: Path, out_path: Path) -> None:
    """Twin-axis: val loss (left, blue) + val Asimov Z (right, green)."""
    with history_path.open() as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_z = [h["val_significance"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, "C0--", label="train loss", alpha=0.6)
    ax1.plot(epochs, val_loss, "C0-", label="val loss", linewidth=1.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_z, "C2-", label=r"val Asimov $Z$", linewidth=1.5)
    ax2.set_ylabel(r"val Asimov $Z$ at WP", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.legend(loc="upper right")

    ax1.set_title("Training curves (twin axis)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _clopper_pearson(k: np.ndarray, n: np.ndarray, alpha: float = 0.3173) -> tuple[np.ndarray, np.ndarray]:
    """Clopper–Pearson 1σ binomial confidence interval for k successes in n trials.

    alpha = 1 − 0.6827 ≈ 0.3173 → returns the (lo, hi) bounds bracketing the
    central 68.27% of the distribution (one Gaussian σ).
    """
    lo = np.where(k == 0, 0.0, beta.ppf(alpha / 2, k, n - k + 1))
    hi = np.where(k == n, 1.0, beta.ppf(1 - alpha / 2, k + 1, n - k))
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


def plot_roc_with_bands(
    scores_test: np.ndarray,
    y_test: np.ndarray,
    cut_baseline_fpr: float,
    cut_baseline_tpr: float,
    out_path: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """ROC + AUC + Clopper–Pearson 1σ bands + cut-based working-point marker."""
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    roc_auc = auc(fpr, tpr)

    n_sig = int((y_test == 1).sum())
    n_bkg = int((y_test == 0).sum())
    tp = tpr * n_sig
    fp = fpr * n_bkg
    tpr_lo, tpr_hi = _clopper_pearson(tp, np.full_like(tp, n_sig, dtype=float))
    fpr_lo, fpr_hi = _clopper_pearson(fp, np.full_like(fp, n_bkg, dtype=float))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "C0-", label=f"DNN (AUC = {roc_auc:.3f})", linewidth=1.5)
    ax.fill_between(fpr, tpr_lo, tpr_hi, color="C0", alpha=0.20, label=r"DNN $\pm 1\sigma$ (Clopper–Pearson)")
    ax.fill_betweenx(tpr, fpr_lo, fpr_hi, color="C0", alpha=0.10)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.plot(
        cut_baseline_fpr, cut_baseline_tpr, "rs", markersize=10,
        label=f"cut-based ({cut_baseline_tpr:.2f}, {cut_baseline_fpr:.2f})",
    )
    ax.set_xlabel("Background efficiency (FPR)")
    ax.set_ylabel("Signal efficiency (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return float(roc_auc), fpr, tpr, np.column_stack([tpr_lo, tpr_hi])


def plot_score_distributions(
    scores_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    scores_test: np.ndarray, y_test: np.ndarray, w_test: np.ndarray,
    out_path: Path,
    ks_results: dict[str, float] | None = None,
) -> None:
    """Signal vs background score distributions, train and test overlaid."""
    bins = np.linspace(0, 1, 41).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    common = {"bins": bins, "density": True}
    ax.hist(scores_train[y_train == 1], weights=w_train[y_train == 1],
            histtype="stepfilled", color="C0", alpha=0.30, label="signal (train)", **common)
    ax.hist(scores_train[y_train == 0], weights=w_train[y_train == 0],
            histtype="stepfilled", color="C3", alpha=0.30, label="background (train)", **common)
    ax.hist(scores_test[y_test == 1], weights=w_test[y_test == 1],
            histtype="step", color="C0", linewidth=1.8, label="signal (test)", **common)
    ax.hist(scores_test[y_test == 0], weights=w_test[y_test == 0],
            histtype="step", color="C3", linewidth=1.8, label="background (test)", **common)
    ax.set_xlabel("DNN score (p(signal))")
    ax.set_ylabel("a.u. (weighted)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Score distributions (weighted)")
    if ks_results is not None:
        sig_flag = " *" if ks_results["signal_p"] < 0.05 else ""
        bkg_flag = " *" if ks_results["background_p"] < 0.05 else ""
        annotation = (
            f"KS overtraining (unweighted, train vs test)\n"
            f"  signal:  stat={ks_results['signal_ks']:.3f}  p={ks_results['signal_p']:.3g}{sig_flag}\n"
            f"  bkg:     stat={ks_results['background_ks']:.3f}  p={ks_results['background_p']:.3g}{bkg_flag}"
        )
        ax.text(
            0.02, 0.02, annotation,
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def ks_overtraining(
    scores_train: np.ndarray, y_train: np.ndarray,
    scores_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, float]:
    """KS test on score distribution shape (unweighted), signal & background separately."""
    sig_stat, sig_p = ks_2samp(scores_train[y_train == 1], scores_test[y_test == 1])
    bkg_stat, bkg_p = ks_2samp(scores_train[y_train == 0], scores_test[y_test == 0])
    return {
        "signal_ks": float(sig_stat),
        "signal_p": float(sig_p),
        "background_ks": float(bkg_stat),
        "background_p": float(bkg_p),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def permutation_importance(
    model: HWWClassifier, X: np.ndarray, y: np.ndarray, feature_names: list[str], rng: np.random.Generator
) -> dict[str, float]:
    """Per-feature AUC drop after shuffling that feature's values across the test set."""
    base_scores = _score(model, X)
    base_auc = float(auc(*roc_curve(y, base_scores)[:2]))
    drops: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        X_shuf = X.copy()
        X_shuf[:, i] = rng.permutation(X_shuf[:, i])
        scores = _score(model, X_shuf)
        shuf_auc = float(auc(*roc_curve(y, scores)[:2]))
        drops[name] = base_auc - shuf_auc
    return drops


def perturbation_importance(
    model: HWWClassifier, X: np.ndarray, feature_names: list[str], shift_sigma: float = 0.01
) -> dict[str, float]:
    """Per-feature mean |Δscore| after shifting that feature by ``shift_sigma * std`` per event.

    Local gradient sensitivity. Different from permutation importance:
    permutation captures global / correlation-aware reliance; perturbation captures
    local responsiveness. Disagreement between the two is interpretable.
    """
    base = _score(model, X)
    stds = X.std(axis=0)
    sensitivity: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        shift = shift_sigma * stds[i]
        X_up = X.copy()
        X_up[:, i] += shift
        X_dn = X.copy()
        X_dn[:, i] -= shift
        delta_up = np.abs(_score(model, X_up) - base)
        delta_dn = np.abs(_score(model, X_dn) - base)
        sensitivity[name] = float(0.5 * (delta_up.mean() + delta_dn.mean()))
    return sensitivity


def plot_importance(
    permutation: dict[str, float], perturbation: dict[str, float], out_path: Path
) -> None:
    names = list(permutation.keys())
    perm = [permutation[n] for n in names]
    pert = [perturbation[n] for n in names]
    y_pos = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 2))
    axes[0].barh(y_pos, perm, color="C0")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$\Delta$AUC after shuffling")
    axes[0].set_title("Permutation importance (global)")
    axes[1].barh(y_pos, pert, color="C2")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"mean |$\Delta$score| at $\pm 0.01\sigma$ shift")
    axes[1].set_title("Perturbation importance (local)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cut-based baseline
# ---------------------------------------------------------------------------


def cut_baseline_metrics(
    config: TrainingConfig,
    X_test: np.ndarray, y_test: np.ndarray, w_test: np.ndarray,
    feature_names: list[str],
) -> tuple[float, float, float, float]:
    """Apply the cut-based selection ``m_T < 125 GeV  AND  dphi_ll < 1.8 rad``.

    Returns (TPR, FPR, signal_yield, background_yield).
    """
    if "m_T" not in feature_names or "dphi_ll" not in feature_names:
        raise RuntimeError(
            "Cut-based baseline requires 'm_T' and 'dphi_ll' features in the feature set."
        )
    i_mT = feature_names.index("m_T")
    i_dphi = feature_names.index("dphi_ll")
    pass_mask = (X_test[:, i_mT] < 125.0) & (X_test[:, i_dphi] < 1.8)
    sig_mask = y_test == 1
    bkg_mask = y_test == 0
    n_sig_total = int(sig_mask.sum())
    n_bkg_total = int(bkg_mask.sum())
    tpr = float((pass_mask & sig_mask).sum() / n_sig_total) if n_sig_total else 0.0
    fpr = float((pass_mask & bkg_mask).sum() / n_bkg_total) if n_bkg_total else 0.0
    s = float(w_test[pass_mask & sig_mask].sum())
    b = float(w_test[pass_mask & bkg_mask].sum())
    return tpr, fpr, s, b


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@chronomat
def run_evaluation(config: TrainingConfig) -> None:
    """End-to-end evaluation; writes plots and prints a summary."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed)

    # --- Load data ----------------------------------------------------
    split = load_split(config.split_path)
    feature_names = cast(list[str], split["feature_names"])
    X_train = cast(np.ndarray, split["X_train"])
    y_train = cast(np.ndarray, split["y_train"])
    w_train = cast(np.ndarray, split["w_train"])
    X_test = cast(np.ndarray, split["X_test"])
    y_test = cast(np.ndarray, split["y_test"])
    w_test = cast(np.ndarray, split["w_test"])

    # --- Pre-fit plots ------------------------------------------------
    LOGGER.info("Pre-fit plots")
    plot_feature_distributions(
        X_train, y_train, feature_names, out_dir / "features_signal_vs_background.png"
    )
    plot_correlation_matrix(
        X_train, y_train, feature_names, out_dir / "feature_correlation.png"
    )

    # --- Load model ---------------------------------------------------
    LOGGER.info("Loading checkpoint: %s", config.checkpoint_path)
    ckpt = torch.load(config.checkpoint_path, weights_only=False)
    model = HWWClassifier.from_checkpoint(ckpt, config)
    model.eval()
    if model.feature_names != feature_names:
        raise RuntimeError(
            f"Feature schema mismatch: model expects {model.feature_names} "
            f"but split has {feature_names}"
        )

    # --- Score train / test -------------------------------------------
    scores_train = _score(model, X_train)
    scores_test = _score(model, X_test)

    # --- Cut-based baseline ------------------------------------------
    cut_tpr, cut_fpr, s_cut, b_cut = cut_baseline_metrics(
        config, X_test, y_test, w_test, feature_names
    )
    z_cut = asimov_significance(s_cut, b_cut)
    LOGGER.info(
        "Cut-based baseline (m_T<125 AND dphi_ll<1.8): TPR=%.4f  FPR=%.4f  s=%.3f  b=%.3f  Z=%.3f",
        cut_tpr, cut_fpr, s_cut, b_cut, z_cut,
    )

    # --- DNN at working point -----------------------------------------
    sig_scores_test = scores_test[y_test == 1]
    threshold = _threshold_at_signal_eff(sig_scores_test, config.working_point_signal_eff)
    s_dnn, b_dnn = _yields(scores_test, y_test, w_test, threshold)
    z_dnn = asimov_significance(s_dnn, b_dnn)
    LOGGER.info(
        "DNN @ %.0f%% signal eff (threshold=%.4f): s=%.3f  b=%.3f  Z=%.3f  (gain over cut-based: %+.3f)",
        100 * config.working_point_signal_eff, threshold, s_dnn, b_dnn, z_dnn, z_dnn - z_cut,
    )

    # --- ROC + score distributions + KS -------------------------------
    LOGGER.info("ROC + score distributions + KS")
    roc_auc, _fpr, _tpr, _bands = plot_roc_with_bands(
        scores_test, y_test, cut_fpr, cut_tpr, out_dir / "roc.png"
    )
    ks_results = ks_overtraining(scores_train, y_train, scores_test, y_test)
    plot_score_distributions(
        scores_train, y_train, w_train, scores_test, y_test, w_test,
        out_dir / "score_distributions.png",
        ks_results=ks_results,
    )

    # --- Training curves ----------------------------------------------
    LOGGER.info("Training curves (twin axis)")
    plot_training_curves(Path(config.loss_history_path), out_dir / "training_curves.png")

    # --- Feature importance ------------------------------------------
    LOGGER.info("Feature importance: permutation + perturbation")
    perm = permutation_importance(model, X_test, y_test, feature_names, rng)
    pert = perturbation_importance(model, X_test, feature_names)
    plot_importance(perm, pert, out_dir / "feature_importance.png")

    # --- Summary ------------------------------------------------------
    LOGGER.info("=" * 70)
    LOGGER.info("Test-set summary")
    LOGGER.info("=" * 70)
    LOGGER.info("AUC (test):                       %.4f", roc_auc)
    LOGGER.info("KS overtraining (signal):         stat=%.4f  p=%.3g  %s",
                ks_results["signal_ks"], ks_results["signal_p"],
                "OK" if ks_results["signal_p"] >= 0.05 else "FLAGGED")
    LOGGER.info("KS overtraining (background):     stat=%.4f  p=%.3g  %s",
                ks_results["background_ks"], ks_results["background_p"],
                "OK" if ks_results["background_p"] >= 0.05 else "FLAGGED")
    LOGGER.info("Cut-based  Z:                     %.3f  (s=%.2f  b=%.2f)", z_cut, s_cut, b_cut)
    LOGGER.info("DNN @ WP   Z:                     %.3f  (s=%.2f  b=%.2f)", z_dnn, s_dnn, b_dnn)
    LOGGER.info("Δ Z (DNN − cut-based):            %+.3f", z_dnn - z_cut)
    LOGGER.info("Permutation importance (sorted):")
    for name, val in sorted(perm.items(), key=lambda kv: -kv[1]):
        LOGGER.info("    %-15s  %+.4f", name, val)
    LOGGER.info("Perturbation importance (sorted):")
    for name, val in sorted(pert.items(), key=lambda kv: -kv[1]):
        LOGGER.info("    %-15s  %.4f", name, val)
    LOGGER.info("Plots written to: %s", out_dir)


def main() -> int:
    config = load_config("config.yaml")
    setup_logging(config.log_path)
    run_evaluation(config)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
