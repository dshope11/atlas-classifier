"""Tests for src.model - forward pass shapes, logit semantics, loss numerics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.config import TrainingConfig
from src.model import HWWClassifier


@pytest.fixture
def model() -> HWWClassifier:
    config = TrainingConfig(hidden_sizes=[16, 8], dropout=0.2, input_noise_std=0.0)
    feature_names = ["a", "b", "c"]
    median = [1.0, 2.0, 3.0]
    iqr = [0.5, 1.0, 1.5]
    return HWWClassifier(config, feature_names, median, iqr)


# ---------------------------------------------------------------------------
# Forward pass - shape, dtype, semantics
# ---------------------------------------------------------------------------


def test_forward_returns_correct_shape(model: HWWClassifier) -> None:
    x = torch.randn(7, 3)
    out = model(x)
    assert out.shape == (7, 1)


def test_forward_returns_raw_logits_not_bounded_to_unit(model: HWWClassifier) -> None:
    """Output should be unbounded (raw logit), not in [0, 1]."""
    model.eval()
    # Construct large inputs to push logits outside [0, 1] in either direction
    x_large = torch.tensor([[100.0, 100.0, 100.0]] * 16)
    x_small = torch.tensor([[-100.0, -100.0, -100.0]] * 16)
    with torch.no_grad():
        out_large = model(x_large)
        out_small = model(x_small)
    # At least one batch should have values outside [0, 1] - confirms no implicit sigmoid
    assert (out_large.abs() > 1).any() or (out_small.abs() > 1).any()


def test_sigmoid_of_logit_is_in_unit_interval(model: HWWClassifier) -> None:
    model.eval()
    x = torch.randn(20, 3) * 5
    with torch.no_grad():
        proba = torch.sigmoid(model(x))
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_predict_proba_returns_unit_interval(model: HWWClassifier) -> None:
    x = torch.randn(20, 3) * 5
    proba = model.predict_proba(x)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


# ---------------------------------------------------------------------------
# Normalisation buffers
# ---------------------------------------------------------------------------


def test_register_buffer_normalisation(model: HWWClassifier) -> None:
    """``normalise(x)`` should produce (x - median) / IQR using the registered buffers."""
    x = torch.tensor([[1.0, 2.0, 3.0]])  # exactly the median
    normed = model.normalise(x)
    np.testing.assert_allclose(normed.numpy(), [[0.0, 0.0, 0.0]], atol=1e-7)


def test_buffers_in_state_dict(model: HWWClassifier) -> None:
    """Norm stats travel with the checkpoint (register_buffer guarantees this)."""
    sd = model.state_dict()
    assert "input_median" in sd
    assert "input_iqr" in sd


# ---------------------------------------------------------------------------
# Loss numerics
# ---------------------------------------------------------------------------


def test_bce_with_pos_weight_no_nan(model: HWWClassifier) -> None:
    """BCEWithLogitsLoss with pos_weight should produce finite loss on random input."""
    pos_weight = torch.tensor([3.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.eval()
    x = torch.randn(64, 3)
    y = torch.randint(0, 2, (64, 1)).float()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
    assert torch.isfinite(loss).all()


def test_bce_with_extreme_logits_does_not_overflow(model: HWWClassifier) -> None:
    """BCEWithLogitsLoss is numerically stable even at very large |logit|."""
    criterion = nn.BCEWithLogitsLoss()
    big_logits = torch.tensor([[100.0], [-100.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = criterion(big_logits, targets)
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_mismatched_median_length() -> None:
    config = TrainingConfig()
    with pytest.raises(ValueError, match="input_median length"):
        HWWClassifier(config, ["a", "b"], [1.0, 2.0, 3.0], [1.0, 1.0])  # 3 medians, 2 features


def test_constructor_rejects_mismatched_iqr_length() -> None:
    config = TrainingConfig()
    with pytest.raises(ValueError, match="input_iqr length"):
        HWWClassifier(config, ["a", "b"], [1.0, 2.0], [1.0, 1.0, 1.0])


def test_n_features_property() -> None:
    config = TrainingConfig()
    m = HWWClassifier(config, ["a", "b", "c", "d"], [0] * 4, [1] * 4)
    assert m.n_features == 4
