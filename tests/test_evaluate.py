"""Tests for the threshold-selection protocol in src.evaluate.

The operating threshold is a tuned hyperparameter: it must be selected on the
val split, frozen, and evaluated on the test split. These tests use synthetic
arrays where the val-argmax and test-argmax thresholds deliberately differ, so
any regression that re-couples selection and reporting onto the same split
changes the reported numbers and fails.

Hand-computed optima (Asimov Z over the roc_curve threshold grid):
- selection (val) split: argmax at threshold 0.4 (Z=0 at 0.9 since b=0,
  ~5.72 at 0.6, ~6.13 at 0.4, ~5.16 at 0.1)
- reporting (test) split: argmax at threshold 0.5 (~6.63), while the frozen
  val threshold 0.4 gives only ~5.18 there (s=10, b=1.5)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import _report_at, scan_thresholds, select_threshold
from src.utils import asimov_significance, compute_yields

# Selection (val) split: Asimov-optimal threshold is 0.4
SCORES_SEL = np.array([0.1, 0.4, 0.6, 0.9])
Y_SEL = np.array([0, 1, 0, 1])
W_SEL = np.array([1.0, 1.0, 1.0, 10.0])

# Reporting (test) split: its own argmax is 0.5, not 0.4
SCORES_REP = np.array([0.45, 0.5, 0.8, 0.95])
Y_REP = np.array([0, 1, 0, 1])
W_REP = np.array([1.0, 2.0, 0.5, 8.0])


def test_select_threshold_picks_asimov_argmax() -> None:
    assert select_threshold(SCORES_SEL, Y_SEL, W_SEL) == pytest.approx(0.4)
    assert select_threshold(SCORES_REP, Y_REP, W_REP) == pytest.approx(0.5)


def test_frozen_row_uses_val_threshold_evaluated_on_test() -> None:
    """The headline Z must come from the val-selected threshold applied to test."""
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL,
        SCORES_REP, Y_REP, W_REP,
        cut_tpr=0.5, z_cut=1.0,
    )
    assert scan.frozen.threshold == pytest.approx(0.4)
    s, b = compute_yields(SCORES_REP, Y_REP, W_REP, 0.4)
    assert (s, b) == (pytest.approx(10.0), pytest.approx(1.5))
    assert scan.frozen.z == pytest.approx(asimov_significance(s, b))
    # The regression this guards against: frozen must NOT be the test argmax
    assert scan.frozen.threshold != scan.oracle.threshold
    assert scan.frozen.z != pytest.approx(scan.oracle.z)


def test_oracle_row_is_test_argmax_upper_bound() -> None:
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL,
        SCORES_REP, Y_REP, W_REP,
        cut_tpr=0.5, z_cut=1.0,
    )
    assert scan.oracle.threshold == pytest.approx(0.5)
    s, b = compute_yields(SCORES_REP, Y_REP, W_REP, 0.5)
    assert scan.oracle.z == pytest.approx(asimov_significance(s, b))
    assert scan.oracle.z > scan.frozen.z


def test_cut_matched_threshold_selected_on_val() -> None:
    """The cut-TPR-matched threshold is also chosen on val, then frozen."""
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL,
        SCORES_REP, Y_REP, W_REP,
        cut_tpr=0.5, z_cut=1.0,
    )
    # On val, TPR=0.5 (unweighted) is first reached at threshold 0.9
    assert scan.cut_matched.threshold == pytest.approx(0.9)
    # Evaluated on test: only the 0.95 signal event passes
    assert scan.cut_matched.tpr == pytest.approx(0.5)
    assert scan.cut_matched.s == pytest.approx(8.0)
    assert scan.cut_matched.b == pytest.approx(0.0)


def test_report_at_unweighted_rates_weighted_yields() -> None:
    report = _report_at(SCORES_REP, Y_REP, W_REP, threshold=0.5)
    # score >= 0.5: signal {0.5, 0.95}, background {0.8}
    assert report.tpr == pytest.approx(1.0)
    assert report.fpr == pytest.approx(0.5)
    assert report.s == pytest.approx(10.0)
    assert report.b == pytest.approx(0.5)
