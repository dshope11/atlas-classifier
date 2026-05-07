"""Feature engineering, normalisation stats, and 3-way stratified split.

Reads ``data/processed/events.h5`` (produced by :mod:`src.data_loading`)
and writes ``data/processed/split.h5`` containing:

* raw (un-normalised) feature arrays per split
* per-event labels and physics weights per split
* ``RobustScaler`` stats (median, IQR) computed from the **train split only**
  — these travel inside the model checkpoint via ``register_buffer()``, not
  applied to the saved features
* feature names (for inference-time validation against the model)

Composite features (initial set; ``n_features`` is parameterised — adding
a feature requires only extending :data:`FEATURE_FNS` and reprocessing):

================  ===================================================
``m_ll``          Dilepton invariant mass via 4-vector sum
``pT_ll``         Magnitude of the dilepton system transverse momentum
``dphi_ll``       |Δφ| between the two leptons, wrapped to [0, π]
``dphi_ll_met``   |Δφ| between the dilepton system and MET
``m_T``           Transverse mass: √(2 · pT_ll · MET · (1 − cos Δφ_ll,MET))
================  ===================================================

The 3-way split fractions and random seed live in ``TrainingConfig``.
``StratifiedShuffleSplit`` ensures ``is_signal`` proportions are preserved
in train/val/test — important because plain random shuffle can drift on
small classes.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Make src.* importable when this module is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Features, Labels, TrainingConfig, Weights, load_config  # noqa: E402
from src.utils import chronomat, print_timings, setup_logging  # noqa: E402

LOGGER = logging.getLogger(Path(__file__).stem)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle to [−π, π] (inclusive)."""
    return cast(np.ndarray, np.arctan2(np.sin(angle), np.cos(angle)))


def _abs_dphi(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """|Δφ| between two angle arrays, in [0, π]."""
    return cast(np.ndarray, np.abs(_wrap_to_pi(phi_a - phi_b)))


def _m_ll(events: np.ndarray) -> np.ndarray:
    """Dilepton invariant mass via 4-vector sum (all in float64 for precision)."""
    pt1 = events["lep_pt_lead"].astype(np.float64)
    pt2 = events["lep_pt_sublead"].astype(np.float64)
    eta1 = events["lep_eta_lead"].astype(np.float64)
    eta2 = events["lep_eta_sublead"].astype(np.float64)
    phi1 = events["lep_phi_lead"].astype(np.float64)
    phi2 = events["lep_phi_sublead"].astype(np.float64)
    e1 = events["lep_e_lead"].astype(np.float64)
    e2 = events["lep_e_sublead"].astype(np.float64)
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
    energy = e1 + e2
    m_sq = energy * energy - (px * px + py * py + pz * pz)
    # Float-precision negative values are physical zero — clip to avoid sqrt(NaN)
    return np.sqrt(np.clip(m_sq, 0.0, None))


def _ptll_pxy(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper: returns (pT_ll, px_ll, py_ll) in float64."""
    pt1 = events["lep_pt_lead"].astype(np.float64)
    pt2 = events["lep_pt_sublead"].astype(np.float64)
    phi1 = events["lep_phi_lead"].astype(np.float64)
    phi2 = events["lep_phi_sublead"].astype(np.float64)
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    return np.sqrt(px * px + py * py), px, py


def _pT_ll(events: np.ndarray) -> np.ndarray:
    return _ptll_pxy(events)[0]


def _dphi_ll(events: np.ndarray) -> np.ndarray:
    return _abs_dphi(
        events["lep_phi_lead"].astype(np.float64),
        events["lep_phi_sublead"].astype(np.float64),
    )


def _dphi_ll_met(events: np.ndarray) -> np.ndarray:
    _, px, py = _ptll_pxy(events)
    phi_ll = np.arctan2(py, px)
    return _abs_dphi(phi_ll, events["met_phi"].astype(np.float64))


def _m_T(events: np.ndarray) -> np.ndarray:
    """Transverse mass m_T = √(2 · pT_ll · MET · (1 − cos Δφ_ll,MET))."""
    ptll = _pT_ll(events)
    met = events["met"].astype(np.float64)
    dphi = _dphi_ll_met(events)
    return cast(np.ndarray, np.sqrt(2.0 * ptll * met * (1.0 - np.cos(dphi))))


def _pt_lead(events: np.ndarray) -> np.ndarray:
    return events["lep_pt_lead"].astype(np.float64)


def _pt_sublead(events: np.ndarray) -> np.ndarray:
    return events["lep_pt_sublead"].astype(np.float64)


def _met(events: np.ndarray) -> np.ndarray:
    return events["met"].astype(np.float64)


def _eta_lead(events: np.ndarray) -> np.ndarray:
    return events["lep_eta_lead"].astype(np.float64)


def _eta_sublead(events: np.ndarray) -> np.ndarray:
    return events["lep_eta_sublead"].astype(np.float64)


# Feature registry — order is the column order in X. Adding a feature here
# is the only change needed in code; the model and HDF5 schema adapt to
# len(FEATURE_FNS) automatically.
FEATURE_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Composite variables
    "m_ll":        _m_ll,
    "pT_ll":       _pT_ll,
    "dphi_ll":     _dphi_ll,
    "dphi_ll_met": _dphi_ll_met,
    "m_T":         _m_T,
    # Raw kinematics — motivated by Run 2 H→WW DNN (ANA-HIGP-2024-07)
    "pt_lead":     _pt_lead,
    "pt_sublead":  _pt_sublead,
    "met":         _met,
    "eta_lead":    _eta_lead,
    "eta_sublead": _eta_sublead,
}

FEATURE_NAMES: list[str] = list(FEATURE_FNS.keys())


def build_features(events_path: str | Path) -> tuple[Features, Labels, Weights]:
    """Read ``events.h5`` and compute the feature matrix, labels, and weights.

    Returns
    -------
    X
        Feature matrix, shape ``(n_events, n_features)``, dtype ``float32``.
    y
        Binary labels, shape ``(n_events,)``, dtype ``int8``.
    w
        Per-event physics weights, shape ``(n_events,)``, dtype ``float64``.
        Used in evaluation only — never in training.
    """
    with h5py.File(events_path, "r") as f:
        events = f["events"][:]
    columns = [fn(events) for fn in FEATURE_FNS.values()]
    X: Features = np.column_stack(columns).astype(np.float32)
    y: Labels = events["is_signal"].astype(np.int8)
    w: Weights = events["event_weight"].astype(np.float64)
    return X, y, w


# ---------------------------------------------------------------------------
# Scaler stats — RobustScaler (median ± IQR)
# ---------------------------------------------------------------------------


def compute_scaler_stats(X_train: Features) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, IQR) per feature, computed on the train split only.

    IQR floors at 1.0 to avoid division-by-zero when a feature is constant.
    These stats are passed to the model constructor and stored via
    ``register_buffer()`` — see ``src/model.py``.
    """
    median = np.median(X_train, axis=0).astype(np.float64)
    q75 = np.percentile(X_train, 75, axis=0).astype(np.float64)
    q25 = np.percentile(X_train, 25, axis=0).astype(np.float64)
    iqr = q75 - q25
    iqr = np.where(iqr > 0, iqr, 1.0)
    return median, iqr


# ---------------------------------------------------------------------------
# 3-way stratified split + persistence
# ---------------------------------------------------------------------------


@chronomat
def build_and_save_split(config: TrainingConfig) -> None:
    """Compute features, split, scaler stats, and save to ``config.split_path``."""
    X, y, w = build_features(config.processed_path)
    LOGGER.info(
        "Loaded %d events; computed %d features per event: %s",
        len(X), X.shape[1], FEATURE_NAMES,
    )

    train_frac, val_frac, test_frac = config.train_val_test_split
    rest_frac = val_frac + test_frac

    # First split: train vs (val+test)
    idx_all = np.arange(len(X))
    idx_train, idx_rest = train_test_split(
        idx_all,
        test_size=rest_frac,
        random_state=config.random_seed,
        stratify=y,
    )
    # Second split: val vs test out of "rest"
    val_within_rest = val_frac / rest_frac
    idx_val, idx_test = train_test_split(
        idx_rest,
        test_size=1.0 - val_within_rest,
        random_state=config.random_seed,
        stratify=y[idx_rest],
    )

    LOGGER.info(
        "Split sizes: train=%d (%.2f%%)  val=%d (%.2f%%)  test=%d (%.2f%%)",
        len(idx_train), 100 * len(idx_train) / len(X),
        len(idx_val), 100 * len(idx_val) / len(X),
        len(idx_test), 100 * len(idx_test) / len(X),
    )
    LOGGER.info(
        "Signal fractions: train=%.4f  val=%.4f  test=%.4f  (full=%.4f)",
        y[idx_train].mean(), y[idx_val].mean(), y[idx_test].mean(), y.mean(),
    )

    median, iqr = compute_scaler_stats(X[idx_train])
    LOGGER.info("Scaler median: %s", median.tolist())
    LOGGER.info("Scaler IQR:    %s", iqr.tolist())

    output = Path(config.split_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        # Raw feature arrays + labels + weights, per split
        for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
            grp = f.create_group(name)
            grp.create_dataset("X", data=X[idx], compression="lzf")
            grp.create_dataset("y", data=y[idx], compression="lzf")
            grp.create_dataset("w", data=w[idx], compression="lzf")
            grp.create_dataset("idx", data=idx, compression="lzf")

        # Scaler stats — feed into model constructor
        f.create_dataset("scaler_median", data=median)
        f.create_dataset("scaler_iqr", data=iqr)

        # Feature schema — used both at model construction and for inference validation
        f.create_dataset(
            "feature_names",
            data=np.array(FEATURE_NAMES, dtype=h5py.string_dtype(encoding="utf-8")),
        )

    LOGGER.info("Wrote %s", output)


def load_split(split_path: str | Path) -> dict[str, np.ndarray | list[str]]:
    """Load a saved split and return a dict with X/y/w for train/val/test plus metadata.

    The X arrays are **raw (un-normalised)** — normalisation is applied inside
    the model via ``register_buffer()``. The ``w`` arrays carry physics weights;
    they're passed through but only consumed by ``evaluate.py``.
    """
    with h5py.File(split_path, "r") as f:
        out: dict[str, np.ndarray | list[str]] = {}
        for split in ("train", "val", "test"):
            for var in ("X", "y", "w"):
                out[f"{var}_{split}"] = f[split][var][:]
        out["scaler_median"] = f["scaler_median"][:]
        out["scaler_iqr"] = f["scaler_iqr"][:]
        out["feature_names"] = [s.decode("utf-8") for s in f["feature_names"][:]]
    return out


def main() -> int:
    config = load_config("config.yaml")
    setup_logging(config.log_path)
    build_and_save_split(config)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
