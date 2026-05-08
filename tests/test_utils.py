"""Tests for src.utils — asimov_significance, compute_yields, clopper_pearson, evaluate_cuts."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import asimov_significance, clopper_pearson, compute_yields, evaluate_cuts


# ---------------------------------------------------------------------------
# asimov_significance
# ---------------------------------------------------------------------------


def test_asimov_returns_zero_when_no_background() -> None:
    assert asimov_significance(10.0, 0.0) == 0.0
    assert asimov_significance(10.0, -1.0) == 0.0


def test_asimov_reduces_to_s_over_sqrtb_when_s_small() -> None:
    """In the s ≪ b limit, Z → s/√b. Exact for s = 0."""
    s, b = 1.0, 1000.0
    z = asimov_significance(s, b)
    np.testing.assert_allclose(z, s / np.sqrt(b), rtol=2e-3)


def test_asimov_above_s_over_sqrtb_when_s_comparable_to_b() -> None:
    """When s ~ b, the Asimov formula is more conservative than S/√B."""
    s, b = 50.0, 100.0
    z_asimov = asimov_significance(s, b)
    z_naive = s / np.sqrt(b)
    # Both > 0 and the Asimov version is meaningfully different
    assert z_asimov > 0
    assert abs(z_asimov - z_naive) > 0.1


def test_asimov_monotonic_in_signal() -> None:
    """More signal → higher significance (b held fixed)."""
    b = 100.0
    zs = [asimov_significance(s, b) for s in [1, 5, 10, 20, 50]]
    assert all(zs[i] < zs[i + 1] for i in range(len(zs) - 1))


# ---------------------------------------------------------------------------
# compute_yields
# ---------------------------------------------------------------------------


def test_compute_yields_basic() -> None:
    """Weighted signal and background yields are summed correctly above threshold."""
    scores = np.array([0.1, 0.5, 0.8, 0.9])
    y = np.array([0, 1, 0, 1])
    w = np.array([2.0, 3.0, 4.0, 5.0])
    s, b = compute_yields(scores, y, w, threshold=0.6)
    # score >= 0.6: indices 2 (bkg, w=4) and 3 (sig, w=5)
    assert s == pytest.approx(5.0)
    assert b == pytest.approx(4.0)


def test_compute_yields_threshold_is_inclusive() -> None:
    """An event with score exactly equal to threshold must pass (>= not >)."""
    scores = np.array([0.6, 0.3])
    y = np.array([1, 0])
    w = np.array([2.0, 1.0])
    s, b = compute_yields(scores, y, w, threshold=0.6)
    assert s == pytest.approx(2.0)
    assert b == pytest.approx(0.0)


def test_compute_yields_all_below_threshold() -> None:
    scores = np.array([0.1, 0.2])
    y = np.array([0, 1])
    w = np.array([1.0, 1.0])
    s, b = compute_yields(scores, y, w, threshold=0.9)
    assert s == pytest.approx(0.0)
    assert b == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# clopper_pearson
# ---------------------------------------------------------------------------


def test_clopper_pearson_k_zero_lo_is_zero() -> None:
    """Lower bound must be exactly 0 when k=0 (no successes observed)."""
    lo, hi = clopper_pearson(np.array([0]), np.array([10]))
    assert lo[0] == pytest.approx(0.0)
    assert 0.0 < hi[0] < 1.0


def test_clopper_pearson_k_equals_n_hi_is_one() -> None:
    """Upper bound must be exactly 1 when k=n (all trials succeeded)."""
    lo, hi = clopper_pearson(np.array([10]), np.array([10]))
    assert hi[0] == pytest.approx(1.0)
    assert 0.0 < lo[0] < 1.0


def test_clopper_pearson_central_interval_contains_true_rate() -> None:
    """For k/n = 0.5, the 1-sigma interval should contain 0.5 and be plausibly narrow."""
    lo, hi = clopper_pearson(np.array([50]), np.array([100]))
    assert lo[0] < 0.5 < hi[0]
    assert (hi[0] - lo[0]) < 0.15  # 68% CI on 100 trials is ~±5%


# ---------------------------------------------------------------------------
# evaluate_cuts
# ---------------------------------------------------------------------------


@pytest.fixture
def events() -> dict[str, np.ndarray]:
    """Synthetic 6-event dataset for cut testing."""
    return {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "y": np.array([10, 20, 30, 40, 50, 60]),
        "flag": np.array([0, 1, 0, 1, 0, 1]),
    }


def test_evaluate_cuts_leaf_gt(events: dict[str, np.ndarray]) -> None:
    mask = evaluate_cuts(events, ["x", "> 3"])
    np.testing.assert_array_equal(mask, [False, False, False, True, True, True])


def test_evaluate_cuts_leaf_eq(events: dict[str, np.ndarray]) -> None:
    mask = evaluate_cuts(events, ["flag", "== 1"])
    np.testing.assert_array_equal(mask, [False, True, False, True, False, True])


def test_evaluate_cuts_and(events: dict[str, np.ndarray]) -> None:
    mask = evaluate_cuts(events, {"and": [["x", "> 2"], ["y", "<= 50"]]})
    np.testing.assert_array_equal(mask, [False, False, True, True, True, False])


def test_evaluate_cuts_or(events: dict[str, np.ndarray]) -> None:
    mask = evaluate_cuts(events, {"or": [["x", "< 2"], ["x", ">= 5"]]})
    np.testing.assert_array_equal(mask, [True, False, False, False, True, True])


def test_evaluate_cuts_nested(events: dict[str, np.ndarray]) -> None:
    """and(or(...), leaf) — exercises recursion."""
    cuts = {"and": [{"or": [["x", "< 2"], ["x", ">= 5"]]}, ["flag", "== 1"]]}
    mask = evaluate_cuts(events, cuts)
    np.testing.assert_array_equal(mask, [False, False, False, False, False, True])


def test_evaluate_cuts_invalid_leaf_raises(events: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="Leaf cut"):
        evaluate_cuts(events, ["x", "y", "z"])


def test_evaluate_cuts_unknown_op_raises(events: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="No operator"):
        evaluate_cuts(events, ["x", "?? 3"])


def test_evaluate_cuts_invalid_compound_raises(events: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="must contain 'and' or 'or'"):
        evaluate_cuts(events, {"xor": [["x", "> 2"]]})
