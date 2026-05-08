"""Shared utilities for the atlas-classifier pipeline.

Six small, decoupled helpers used across data loading, training, and evaluation:

- ``setup_logging``: configure root logger to write to both stdout and a log file.

- ``chronomat``: a timing decorator that accumulates wall-clock times per function
  into an ``OrderedDict`` for end-of-run reporting.

- ``asimov_significance``: discovery significance using Cowan et al. 2011, Eq. 97.
  Reduces to S/sqrt(B) in the s << b limit. Used for both the cut-based baseline and
  the working-point comparison.

- ``compute_yields``: sum of physics weights passing a score threshold, split by
  signal and background label.

- ``clopper_pearson``: 1sigma Clopper-Pearson binomial confidence interval.

- ``evaluate_cuts``: a recursive boolean evaluator for a small YAML cut DSL
  (``{"and": [["var", "> X"], {"or": [...]}]}``).
"""

from __future__ import annotations

import logging
import operator
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy.stats import beta as _beta_dist


# ---------------------------------------------------------------------------
# setup_logging - stdout + file handler on the root logger
# ---------------------------------------------------------------------------


def setup_logging(log_path: str) -> None:
    """Configure logging to both stdout and ``log_path`` (appended)."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )

# ---------------------------------------------------------------------------
# chronomat - wall-clock timing decorator
# ---------------------------------------------------------------------------

_TIMINGS: OrderedDict[str, float] = OrderedDict()

F = TypeVar("F", bound=Callable[..., Any])


def chronomat(func: F) -> F:
    """Decorator that accumulates a function's wall-clock time in ``_TIMINGS``.

    Multiple calls to the same function add to the running total. Use
    ``print_timings()`` at the end of a script to report the breakdown.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            _TIMINGS[func.__name__] = _TIMINGS.get(func.__name__, 0.0) + elapsed

    return wrapper  # type: ignore[return-value]


def print_timings(logger: Any | None = None) -> None:
    """Print accumulated ``@chronomat`` timings via ``logger`` or stdout."""
    if not _TIMINGS:
        return
    lines = ["Wall-clock timings:"]
    for name, total in _TIMINGS.items():
        lines.append(f"  {name}: {total:.2f}s")
    msg = "\n".join(lines)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# asimov_significance - Cowan et al. 2011, Eq. 97
# ---------------------------------------------------------------------------


def asimov_significance(s: float, b: float) -> float:
    """Asimov discovery significance ``Z = sqrt(2 * [(s+b) * ln(1 + s/b) - s])``.

    Source: Cowan, Cranmer, Gross, Vitells, "Asymptotic formulae for likelihood-based
    tests of new physics", Eur. Phys. J. C 71 (2011) 1554, Eq. 97.

    Reduces to S/sqrt(B) in the s << b limit but handles the s ~ b regime correctly.
    Returns 0 if ``b <= 0`` (no background - significance undefined).
    """
    if b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


# ---------------------------------------------------------------------------
# compute_yields - weighted signal/background yields above a score threshold
# ---------------------------------------------------------------------------


def compute_yields(
    scores: np.ndarray, y: np.ndarray, w: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Sum of physics weights for signal/background events with score >= threshold."""
    pass_mask = scores >= threshold
    s = float(w[pass_mask & (y == 1)].sum())
    b = float(w[pass_mask & (y == 0)].sum())
    return s, b


# ---------------------------------------------------------------------------
# clopper_pearson - 1sigma Clopper-Pearson binomial confidence interval
# ---------------------------------------------------------------------------


def clopper_pearson(
    k: np.ndarray, n: np.ndarray, alpha: float = 0.3173
) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson 1sigma binomial confidence interval for k successes in n trials.

    alpha = 1 - 0.6827 ~= 0.3173; returns the (lo, hi) bounds bracketing the
    central 68.27% of the distribution (one Gaussian sigma).
    """
    lo = np.where(k == 0, 0.0, _beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = np.where(k == n, 1.0, _beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


# ---------------------------------------------------------------------------
# evaluate_cuts - recursive YAML cut DSL
# ---------------------------------------------------------------------------

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def _leaf_mask(events: Any, var: str, expr: str) -> np.ndarray:
    """Evaluate a single ``[var, "op value"]`` condition against ``events[var]``.

    Operators tried in length order so that ``>=``/``<=``/``==``/``!=`` are matched
    before their single-character prefixes.
    """
    expr = expr.strip()
    for op_symbol in ("==", "!=", ">=", "<=", ">", "<"):
        if expr.startswith(op_symbol):
            value_str = expr[len(op_symbol):].strip()
            try:
                value: float | int = float(value_str)
                if "." not in value_str and "e" not in value_str.lower():
                    value = int(value)
            except ValueError as err:
                raise ValueError(f"Cannot parse cut value {value_str!r}") from err
            return np.asarray(_OPERATORS[op_symbol](events[var], value))
    raise ValueError(f"No operator found in cut expression: {expr!r}")


def evaluate_cuts(events: Any, cuts: Any) -> np.ndarray:
    """Evaluate a nested cut DSL against ``events`` (any dict-like or structured array).

    A cut is one of:

    * **Leaf**: ``["var_name", "op value"]`` - evaluates ``events[var_name] op value``
    * **And node**: ``{"and": [cut, cut, ...]}`` - element-wise logical AND
    * **Or node**:  ``{"or":  [cut, cut, ...]}`` - element-wise logical OR

    Returns a boolean numpy array of the same length as ``events``.
    """
    if isinstance(cuts, list):
        if len(cuts) != 2 or not isinstance(cuts[0], str):
            raise ValueError(f"Leaf cut must be [var_name, expr], got: {cuts!r}")
        var, expr = cuts
        return _leaf_mask(events, var, expr)

    if isinstance(cuts, dict):
        if "and" in cuts:
            sub_masks = [evaluate_cuts(events, sub) for sub in cuts["and"]]
            return np.asarray(np.logical_and.reduce(sub_masks))
        if "or" in cuts:
            sub_masks = [evaluate_cuts(events, sub) for sub in cuts["or"]]
            return np.asarray(np.logical_or.reduce(sub_masks))
        raise ValueError(f"Compound cut must contain 'and' or 'or', got: {cuts!r}")

    raise TypeError(f"Cuts must be list or dict, got: {type(cuts).__name__}")
