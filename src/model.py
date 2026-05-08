"""HWWClassifier - PyTorch DNN for binary signal/background classification.

The model is fully self-contained: feature names, normalisation statistics,
architecture, and weights all live inside the checkpoint. At inference time
no external config or scaler file is needed.

Forward pass:

1. **InputNorm** - ``(x - median) / iqr`` via two ``register_buffer``
   tensors (``input_median``, ``input_iqr``). Buffers move with the
   model on ``.to(device)`` and are saved automatically in the
   ``state_dict``.
2. **GaussianNoise** - adds ``N(0, sigma)`` to the inputs during training
   only. Acts as a small input regulariser.
3. **Hidden layers** - repeating ``Linear -> BatchNorm1d -> ReLU ->
   Dropout`` block, sized per ``config.hidden_sizes``.
4. **Output** - single linear projection to a raw logit. No sigmoid;
   ``BCEWithLogitsLoss`` applies it internally during training, and
   ``torch.sigmoid`` is applied explicitly at inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from src.config import TrainingConfig

# Anything torch.as_tensor accepts - sequences, ndarrays, tensors. Kept loose
# on purpose so callers (preprocessing, train) can pass np arrays directly.
ScalerStats = Sequence[float] | Tensor | np.ndarray


class HWWClassifier(nn.Module):
    """Binary classifier with in-model normalisation and a raw-logit output."""

    # Type hints for register_buffer-created attributes (mypy cannot infer these)
    input_median: Tensor
    input_iqr: Tensor

    def __init__(
        self,
        config: TrainingConfig,
        feature_names: Sequence[str],
        input_median: ScalerStats,
        input_iqr: ScalerStats,
    ) -> None:
        super().__init__()

        self.feature_names: list[str] = list(feature_names)
        self.input_noise_std: float = float(config.input_noise_std)
        n_features = len(self.feature_names)

        median_t = torch.as_tensor(input_median, dtype=torch.float32).reshape(1, -1)
        iqr_t = torch.as_tensor(input_iqr, dtype=torch.float32).reshape(1, -1)
        if median_t.shape[1] != n_features:
            raise ValueError(
                f"input_median length ({median_t.shape[1]}) != n_features ({n_features})"
            )
        if iqr_t.shape[1] != n_features:
            raise ValueError(
                f"input_iqr length ({iqr_t.shape[1]}) != n_features ({n_features})"
            )
        self.register_buffer("input_median", median_t)
        self.register_buffer("input_iqr", iqr_t)

        layers: list[nn.Module] = []
        in_dim = n_features
        for out_dim in config.hidden_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=config.dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.hidden = nn.Sequential(*layers)

        # Kaiming init for ReLU - torch default is Kaiming uniform; we use
        # Kaiming normal explicitly for reproducibility regardless of pytorch version
        for m in self.hidden.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def normalise(self, x: Tensor) -> Tensor:
        """Apply RobustScaler normalisation using the registered buffers."""
        return (x - self.input_median) / self.input_iqr

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logits of shape ``(batch, 1)``."""
        x = self.normalise(x)
        if self.training and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return cast(Tensor, self.hidden(x))

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Inference helper: returns ``sigmoid(logit)`` as p(signal) in (0, 1)."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()

    # -- Checkpoint I/O -----------------------------------------------------
    # We save state_dict + feature_names + config snapshot in one dict so the
    # checkpoint is fully self-describing. weights_only=False is required to
    # restore the feature_names list and the config dict; this is safe because
    # the checkpoint is something we wrote ourselves.

    def make_checkpoint(self, config: TrainingConfig) -> dict[str, Any]:
        """Build the dict to ``torch.save`` (state_dict + reconstruction metadata)."""
        return {
            "state_dict": self.state_dict(),
            "feature_names": list(self.feature_names),
            "input_noise_std": self.input_noise_std,
            "hidden_sizes": list(config.hidden_sizes),
            "dropout": config.dropout,
        }

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict[str, Any], config: TrainingConfig
    ) -> "HWWClassifier":
        """Reconstruct a model from a checkpoint dict produced by ``make_checkpoint``.

        The architecture is re-derived from the checkpoint's ``hidden_sizes`` /
        ``dropout`` (so the current ``config.yaml`` need not match the trained
        model). Scaler stats come from the saved ``state_dict`` buffers.
        """
        # Build a config-shaped object that matches the saved architecture
        rebuilt = TrainingConfig(
            hidden_sizes=list(ckpt["hidden_sizes"]),
            dropout=float(ckpt["dropout"]),
            input_noise_std=float(ckpt["input_noise_std"]),
            # Other fields don't affect the network architecture; copy from runtime config
            lr=config.lr,
            epochs=config.epochs,
            patience=config.patience,
            batch_size=config.batch_size,
            lumi=config.lumi,
            train_val_test_split=config.train_val_test_split,
            random_seed=config.random_seed,
            lr_patience=config.lr_patience,
            lr_factor=config.lr_factor,
            cuts=config.cuts,
        )
        feature_names = list(ckpt["feature_names"])
        # Buffers in state_dict carry the median/IQR - but the constructor
        # also requires explicit values. Read them from the state_dict.
        median = ckpt["state_dict"]["input_median"].squeeze().tolist()
        iqr = ckpt["state_dict"]["input_iqr"].squeeze().tolist()
        model = cls(rebuilt, feature_names, median, iqr)
        model.load_state_dict(ckpt["state_dict"])
        return model
