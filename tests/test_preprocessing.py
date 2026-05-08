"""Tests for src.preprocessing - feature math, split correctness, scaler stats."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.config import TrainingConfig
from src.preprocessing import (
    FEATURE_NAMES,
    _abs_dphi,
    _m_T,
    _wrap_to_pi,
    build_and_save_split,
    build_features,
    compute_scaler_stats,
    load_split,
)


# ---------------------------------------------------------------------------
# dphi wrapping
# ---------------------------------------------------------------------------


def test_wrap_to_pi_identity_in_range() -> None:
    angles = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(_wrap_to_pi(angles), angles, atol=1e-7)


def test_wrap_to_pi_wraps_above() -> None:
    np.testing.assert_allclose(_wrap_to_pi(np.array([4.0])), np.array([4.0 - 2 * np.pi]), atol=1e-7)


def test_wrap_to_pi_wraps_below() -> None:
    np.testing.assert_allclose(_wrap_to_pi(np.array([-4.0])), np.array([-4.0 + 2 * np.pi]), atol=1e-7)


def test_abs_dphi_in_zero_pi() -> None:
    """|dphi| must always be in [0, pi], never larger."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-10, 10, size=1000)
    b = rng.uniform(-10, 10, size=1000)
    d = _abs_dphi(a, b)
    assert (d >= 0).all()
    assert (d <= np.pi + 1e-7).all()


def test_abs_dphi_collinear_is_zero() -> None:
    np.testing.assert_allclose(_abs_dphi(np.array([1.5]), np.array([1.5])), np.array([0.0]), atol=1e-7)


def test_abs_dphi_back_to_back_is_pi() -> None:
    np.testing.assert_allclose(_abs_dphi(np.array([0.0]), np.array([np.pi])), np.array([np.pi]), atol=1e-7)


# ---------------------------------------------------------------------------
# m_T formula - manual calculation
# ---------------------------------------------------------------------------


def _events_for_mT(pt1: float, pt2: float, met: float, met_phi: float, phi1: float = 0.0, phi2: float = 0.0) -> np.ndarray:
    """Synthesise a single-row structured array suitable for _m_T()."""
    dtype = np.dtype([
        ("lep_pt_lead", "f8"), ("lep_pt_sublead", "f8"),
        ("lep_phi_lead", "f8"), ("lep_phi_sublead", "f8"),
        ("met", "f8"), ("met_phi", "f8"),
    ])
    arr = np.zeros(1, dtype=dtype)
    arr["lep_pt_lead"] = pt1
    arr["lep_pt_sublead"] = pt2
    arr["lep_phi_lead"] = phi1
    arr["lep_phi_sublead"] = phi2
    arr["met"] = met
    arr["met_phi"] = met_phi
    return arr


def test_mT_collinear_dilepton_and_met_zero() -> None:
    """When dphi(ll, MET) = 0, m_T should be zero (independent of pT_ll, MET)."""
    # dilepton parallel to +x axis (phi=0), MET also at phi=0
    events = _events_for_mT(pt1=50.0, pt2=30.0, met=40.0, met_phi=0.0)
    mt = _m_T(events)
    np.testing.assert_allclose(mt, [0.0], atol=1e-7)


def test_mT_back_to_back_dilepton_and_met_max() -> None:
    """When dphi(ll, MET) = pi, m_T = sqrt(2 * pT_ll * MET * 2) = 2 * sqrt(pT_ll * MET)."""
    pt1, pt2, met = 60.0, 40.0, 50.0
    events = _events_for_mT(pt1=pt1, pt2=pt2, met=met, met_phi=np.pi)
    pt_ll = pt1 + pt2  # both leptons at phi=0 -> vector sum = scalar sum
    expected = np.sqrt(2 * pt_ll * met * 2)
    mt = _m_T(events)
    np.testing.assert_allclose(mt, [expected], atol=1e-5)


# ---------------------------------------------------------------------------
# Stratified split + scaler stats - end-to-end on a synthetic events.h5
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_events_h5(tmp_path: Path) -> Path:
    """Write a synthetic events.h5 file with 1000 events (30% signal)."""
    rng = np.random.default_rng(42)
    n = 1000
    is_sig = (rng.random(n) < 0.30).astype(np.int8)
    dtype = np.dtype([
        ("lep_pt_lead", "f4"), ("lep_pt_sublead", "f4"),
        ("lep_eta_lead", "f4"), ("lep_eta_sublead", "f4"),
        ("lep_phi_lead", "f4"), ("lep_phi_sublead", "f4"),
        ("lep_e_lead", "f4"), ("lep_e_sublead", "f4"),
        ("lep_charge_lead", "i4"), ("lep_charge_sublead", "i4"),
        ("lep_type_lead", "i4"), ("lep_type_sublead", "i4"),
        ("met", "f4"), ("met_phi", "f4"),
        ("jet_n", "i4"),
        ("event_weight", "f8"), ("is_signal", "i1"),
    ])
    arr = np.zeros(n, dtype=dtype)
    # signal: low m_ll mode (small dphi); background: high m_ll
    pt = rng.uniform(20, 100, n)
    arr["lep_pt_lead"] = pt
    arr["lep_pt_sublead"] = rng.uniform(15, 60, n)
    arr["lep_eta_lead"] = rng.uniform(-2.5, 2.5, n)
    arr["lep_eta_sublead"] = rng.uniform(-2.5, 2.5, n)
    arr["lep_phi_lead"] = rng.uniform(-np.pi, np.pi, n)
    # signal collinear, background back-to-back
    arr["lep_phi_sublead"] = np.where(
        is_sig, arr["lep_phi_lead"] + rng.normal(0, 0.3, n), arr["lep_phi_lead"] + np.pi + rng.normal(0, 0.3, n)
    )
    arr["lep_e_lead"] = arr["lep_pt_lead"] * np.cosh(arr["lep_eta_lead"])
    arr["lep_e_sublead"] = arr["lep_pt_sublead"] * np.cosh(arr["lep_eta_sublead"])
    arr["lep_charge_lead"] = rng.choice([-1, 1], n)
    arr["lep_charge_sublead"] = -arr["lep_charge_lead"]
    arr["lep_type_lead"] = rng.choice([11, 13], n)
    arr["lep_type_sublead"] = rng.choice([11, 13], n)
    arr["met"] = rng.uniform(0, 100, n)
    arr["met_phi"] = rng.uniform(-np.pi, np.pi, n)
    arr["jet_n"] = 0
    arr["event_weight"] = rng.uniform(0.001, 0.1, n)
    arr["is_signal"] = is_sig
    out = tmp_path / "events.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("events", data=arr)
    return out


def test_build_features_shape_and_dtypes(fake_events_h5: Path) -> None:
    X, y, w = build_features(fake_events_h5)
    assert X.shape == (1000, len(FEATURE_NAMES))
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert w.dtype == np.float64
    # Features should be finite
    assert np.isfinite(X).all()


def _arr(split: Mapping[str, object], key: str) -> np.ndarray:
    """Narrow ``load_split`` return value (which mixes ndarrays and lists) to ndarray."""
    from typing import cast as _cast
    return _cast(np.ndarray, split[key])


def test_build_and_save_split_stratification(fake_events_h5: Path, tmp_path: Path) -> None:
    config = TrainingConfig(
        processed_path=str(fake_events_h5),
        split_path=str(tmp_path / "split.h5"),
        random_seed=42,
    )
    build_and_save_split(config)
    split = load_split(config.split_path)
    y_train = _arr(split, "y_train")
    y_val = _arr(split, "y_val")
    y_test = _arr(split, "y_test")
    overall = (y_train.sum() + y_val.sum() + y_test.sum()) / (len(y_train) + len(y_val) + len(y_test))
    # Stratification should preserve the signal fraction in every split (within tolerance)
    np.testing.assert_allclose(y_train.mean(), overall, atol=0.005)
    np.testing.assert_allclose(y_val.mean(), overall, atol=0.005)
    np.testing.assert_allclose(y_test.mean(), overall, atol=0.005)


def test_load_split_returns_raw_unnormalised_features(fake_events_h5: Path, tmp_path: Path) -> None:
    """X arrays returned by load_split should NOT be standardised - model handles that."""
    config = TrainingConfig(
        processed_path=str(fake_events_h5),
        split_path=str(tmp_path / "split.h5"),
        random_seed=42,
    )
    build_and_save_split(config)
    split = load_split(config.split_path)
    X_train = _arr(split, "X_train")
    # If we'd normalised, mean would be ~= 0. Raw features are well away from 0.
    assert abs(X_train.mean()) > 1.0, "Features look normalised - they should be raw"


def test_scaler_stats_train_only(fake_events_h5: Path, tmp_path: Path) -> None:
    """Scaler stats must match the train split alone, not the full dataset."""
    config = TrainingConfig(
        processed_path=str(fake_events_h5),
        split_path=str(tmp_path / "split.h5"),
        random_seed=42,
    )
    build_and_save_split(config)
    split = load_split(config.split_path)
    median = _arr(split, "scaler_median")
    iqr = _arr(split, "scaler_iqr")
    # Recompute from train slice - must match exactly
    expected_median, expected_iqr = compute_scaler_stats(_arr(split, "X_train"))
    np.testing.assert_allclose(median, expected_median, rtol=1e-12)
    np.testing.assert_allclose(iqr, expected_iqr, rtol=1e-12)


def test_compute_scaler_stats_handles_constant_feature() -> None:
    """If a feature is constant (IQR=0), the IQR floor is 1.0 to avoid div-by-zero."""
    X = np.ones((100, 2), dtype=np.float32)
    X[:, 0] = np.linspace(0, 10, 100)  # variable
    median, iqr = compute_scaler_stats(X)
    assert iqr[1] == 1.0  # constant column floored to 1
    assert iqr[0] > 0.0
