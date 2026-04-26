"""Shared utilities for the atlas-classifier pipeline.

Three small, decoupled helpers used across data loading, training, and evaluation:

- ``chronomat``: a timing decorator that accumulates wall-clock times per function
  into an ``OrderedDict`` for end-of-run reporting.

- ``asimov_significance``: discovery significance using Cowan et al. 2011, Eq. 97.
  Reduces to S/√B in the s ≪ b limit. Used in evaluate.py for both the cut-based
  baseline and the DNN working-point comparison.

- ``evaluate_cuts``: a recursive boolean evaluator for a small YAML cut DSL
  (``{"and": [["var", "> X"], {"or": [...]}]}``).
"""

from __future__ import annotations

import operator
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np

# ---------------------------------------------------------------------------
# chronomat — wall-clock timing decorator
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
# asimov_significance — Cowan et al. 2011, Eq. 97
# ---------------------------------------------------------------------------


def asimov_significance(s: float, b: float) -> float:
    """Asimov discovery significance ``Z = √(2·[(s+b)·ln(1 + s/b) − s])``.

    Source: Cowan, Cranmer, Gross, Vitells, "Asymptotic formulae for likelihood-based
    tests of new physics", Eur. Phys. J. C 71 (2011) 1554, Eq. 97.

    Reduces to S/√B in the s ≪ b limit but handles the s ~ b regime correctly.
    Returns 0 if ``b ≤ 0`` (no background — significance undefined).
    """
    if b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


# ---------------------------------------------------------------------------
# evaluate_cuts — recursive YAML cut DSL
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
            return _OPERATORS[op_symbol](events[var], value)
    raise ValueError(f"No operator found in cut expression: {expr!r}")


def evaluate_cuts(events: Any, cuts: Any) -> np.ndarray:
    """Evaluate a nested cut DSL against ``events`` (any dict-like or structured array).

    A cut is one of:

    * **Leaf**: ``["var_name", "op value"]`` — evaluates ``events[var_name] op value``
    * **And node**: ``{"and": [cut, cut, ...]}`` — element-wise logical AND
    * **Or node**:  ``{"or":  [cut, cut, ...]}`` — element-wise logical OR

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
            return np.logical_and.reduce(sub_masks)
        if "or" in cuts:
            sub_masks = [evaluate_cuts(events, sub) for sub in cuts["or"]]
            return np.logical_or.reduce(sub_masks)
        raise ValueError(f"Compound cut must contain 'and' or 'or', got: {cuts!r}")

    raise TypeError(f"Cuts must be list or dict, got: {type(cuts).__name__}")
