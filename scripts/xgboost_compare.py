"""XGBoost comparison for the HWWClassifier pipeline.

Trains an XGBoost classifier on the same ``data/processed/split.h5`` and features
as the DNN, runs an Optuna HPO search to find good hyperparameters, then compares:

- Cut-based baseline Asimov Z (``config.baseline_cuts``)
- DNN (loaded from ``config.checkpoint_path``) at optimal threshold
- XGBoost at optimal threshold

XGBoost is tree-based so no feature normalisation is needed; raw X arrays from
split.h5 are passed directly. Physics weights are used only in evaluation (same
convention as the DNN: no sample_weight in training).

Outputs:
- ``data/processed/eval/roc_comparison.png`` - overlaid ROC curves + cut-based WP
- Logged summary table

Usage:
    python scripts/xgboost_compare.py [--n-trials 50] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

# macOS ships two OpenMP runtimes in this process - one bundled with PyTorch's
# libomp, one with XGBoost's. Letting either auto-thread leads to a segfault on
# the second XGBClassifier.fit() call inside the HPO loop. Pinning to 1 thread
# before xgboost is imported is the canonical workaround. Per-trial fit is then
# single-threaded (~5-10x slower) but the 50-trial loop completes reliably.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import auc, roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainingConfig, load_config  # noqa: E402
from src.evaluate import cut_baseline_metrics, scan_thresholds, score_dnn  # noqa: E402
from src.model import HWWClassifier  # noqa: E402
from src.preprocessing import load_split  # noqa: E402
from src.utils import (  # noqa: E402
    asimov_significance,
    chronomat,
    clopper_pearson,
    print_timings,
    setup_logging,
)

LOGGER = logging.getLogger(Path(__file__).stem)


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------


def _xgb_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
    params: dict[str, Any] = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }
    clf = xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        early_stopping_rounds=20,
        device="cpu",
        n_jobs=1,
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))


def run_hpo(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            n_trials: int, seed: int) -> dict[str, Any]:
    """Return best hyperparameters from an Optuna TPE search on val AUC."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda t: _xgb_objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
    )
    best = study.best_trial
    LOGGER.info(
        "HPO complete - best trial %d: val_auc=%.4f  params=%s",
        best.number, best.value, best.params,
    )
    return best.params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc_comparison(
    dnn_scores: np.ndarray,
    xgb_scores: np.ndarray,
    y_test: np.ndarray,
    cut_fpr: float,
    cut_tpr: float,
    out_path: Path,
) -> tuple[float, float]:
    """Overlaid DNN + XGBoost ROC curves with Clopper-Pearson bands + cut-based WP.

    Returns (dnn_auc, xgb_auc).
    """
    n_sig = int((y_test == 1).sum())
    n_bkg = int((y_test == 0).sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    aucs: list[float] = []

    for scores, label, color in [
        (dnn_scores, "DNN", "C0"),
        (xgb_scores, "XGBoost", "C1"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = float(auc(fpr, tpr))
        aucs.append(roc_auc)

        tp = tpr * n_sig
        fp = fpr * n_bkg
        tpr_lo, tpr_hi = clopper_pearson(tp, np.full_like(tp, n_sig, dtype=float))
        fpr_lo, fpr_hi = clopper_pearson(fp, np.full_like(fp, n_bkg, dtype=float))

        ax.plot(fpr, tpr, f"{color}-", label=f"{label} (AUC = {roc_auc:.3f})", linewidth=1.5)
        ax.fill_between(fpr, tpr_lo, tpr_hi, color=color, alpha=0.15)
        ax.fill_betweenx(tpr, fpr_lo, fpr_hi, color=color, alpha=0.08)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.plot(
        cut_fpr, cut_tpr, "rs", markersize=10,
        label=f"cut-based ({cut_tpr:.2f}, {cut_fpr:.2f})",
    )
    ax.set_xlabel("Background efficiency (FPR)")
    ax.set_ylabel("Signal efficiency (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("ROC curve comparison: DNN vs XGBoost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOGGER.info("ROC comparison plot written to %s", out_path)
    return aucs[0], aucs[1]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@chronomat
def run_comparison(config: TrainingConfig, n_trials: int) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load split -------------------------------------------------------
    split = load_split(config.split_path)
    feature_names = cast(list[str], split["feature_names"])
    X_train = cast(np.ndarray, split["X_train"])
    y_train = cast(np.ndarray, split["y_train"])
    X_val   = cast(np.ndarray, split["X_val"])
    y_val   = cast(np.ndarray, split["y_val"])
    X_test  = cast(np.ndarray, split["X_test"])
    y_test  = cast(np.ndarray, split["y_test"])
    w_test  = cast(np.ndarray, split["w_test"])
    LOGGER.info(
        "Split loaded - train=%d  val=%d  test=%d  features=%d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )

    # --- Cut-based baseline -----------------------------------------------
    cut_tpr, cut_fpr, s_cut, b_cut = cut_baseline_metrics(
        config, X_test, y_test, w_test, feature_names
    )
    z_cut = asimov_significance(s_cut, b_cut)
    LOGGER.info(
        "Cut-based baseline: TPR=%.4f  FPR=%.4f  s=%.3f  b=%.3f  Z=%.3f",
        cut_tpr, cut_fpr, s_cut, b_cut, z_cut,
    )

    # --- XGBoost HPO ------------------------------------------------------
    LOGGER.info("Starting XGBoost HPO (%d trials) ...", n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_params = run_hpo(X_train, y_train, X_val, y_val, n_trials, config.random_seed)

    # --- Train final XGBoost ----------------------------------------------
    LOGGER.info("Training final XGBoost with best params ...")
    clf = xgb.XGBClassifier(
        **best_params,
        eval_metric="auc",
        early_stopping_rounds=20,
        device="cpu",
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_auc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    xgb_test_auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    LOGGER.info(
        "XGBoost trained - best_iteration=%d  val_auc=%.4f  test_auc=%.4f",
        clf.best_iteration, xgb_val_auc, xgb_test_auc,
    )

    # --- XGBoost feature importance (gain) --------------------------------
    importance: dict[str, float] = {
        k: float(v) for k, v in clf.get_booster().get_score(importance_type="gain").items()  # type: ignore[arg-type]
    }
    total = sum(importance.values()) or 1.0
    LOGGER.info("XGBoost feature importance (gain, normalised):")
    for fname in sorted(importance, key=lambda k: -importance[k]):
        fidx = int(fname[1:]) if fname.startswith("f") else -1
        display = feature_names[fidx] if 0 <= fidx < len(feature_names) else fname
        LOGGER.info("    %-15s  %.4f", display, importance[fname] / total)

    # --- Score test set ---------------------------------------------------
    xgb_scores = clf.predict_proba(X_test)[:, 1]

    # --- Load DNN + score test set ----------------------------------------
    LOGGER.info("Loading DNN checkpoint: %s", config.checkpoint_path)
    ckpt = torch.load(config.checkpoint_path, weights_only=True)
    dnn_model = HWWClassifier.from_checkpoint(ckpt, config)
    dnn_model.eval()
    dnn_scores = score_dnn(dnn_model, X_test)
    dnn_test_auc = float(auc(*roc_curve(y_test, dnn_scores)[:2]))
    LOGGER.info("DNN test_auc=%.4f", dnn_test_auc)

    # --- Threshold scans --------------------------------------------------
    (xgb_z_opt, xgb_tpr_opt, xgb_fpr_opt, xgb_s_opt, xgb_b_opt,
     xgb_z_cut_tpr, xgb_tpr_matched, xgb_fpr_matched,
     xgb_s_cut_tpr, xgb_b_cut_tpr) = scan_thresholds(
        xgb_scores, y_test, w_test, cut_tpr, z_cut, label="XGBoost",
    )

    (dnn_z_opt, dnn_tpr_opt, dnn_fpr_opt, dnn_s_opt, dnn_b_opt,
     dnn_z_cut_tpr, dnn_tpr_matched, dnn_fpr_matched,
     dnn_s_cut_tpr, dnn_b_cut_tpr) = scan_thresholds(
        dnn_scores, y_test, w_test, cut_tpr, z_cut, label="DNN",
    )

    # --- Comparison ROC plot ----------------------------------------------
    dnn_auc, xgb_auc = plot_roc_comparison(
        dnn_scores, xgb_scores, y_test, cut_fpr, cut_tpr,
        out_dir / "roc_comparison.png",
    )

    # --- Summary ----------------------------------------------------------
    LOGGER.info("=" * 70)
    LOGGER.info("Comparison summary")
    LOGGER.info("=" * 70)
    LOGGER.info("Cut-based  Z: %.3f  (s=%.2f  b=%.2f)", z_cut, s_cut, b_cut)
    LOGGER.info(
        "DNN        AUC: %.4f  Z @ opt: %.3f  (dZ=%+.3f)  Z @ cut-TPR: %.3f  (dZ=%+.3f)",
        dnn_auc, dnn_z_opt, dnn_z_opt - z_cut, dnn_z_cut_tpr, dnn_z_cut_tpr - z_cut,
    )
    LOGGER.info(
        "XGBoost    AUC: %.4f  Z @ opt: %.3f  (dZ=%+.3f)  Z @ cut-TPR: %.3f  (dZ=%+.3f)",
        xgb_auc, xgb_z_opt, xgb_z_opt - z_cut, xgb_z_cut_tpr, xgb_z_cut_tpr - z_cut,
    )
    LOGGER.info("Plots written to: %s", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna HPO trials")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--log-path", type=str, default="logs/xgboost_compare.log")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(args.log_path)
    run_comparison(config, args.n_trials)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
