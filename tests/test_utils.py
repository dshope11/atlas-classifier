"""Tests for src.utils — asimov_significance, evaluate_cuts."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils import asimov_significance, evaluate_cuts


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
