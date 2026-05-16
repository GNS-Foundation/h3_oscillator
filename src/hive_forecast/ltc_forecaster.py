"""
LTC-based forecaster — H3-Oscillator adapted for Hive load forecasting.

Architectural design notes
==========================
The original H3-Oscillator (Gray-Scott benchmark) had two components:
  1. C6-equivariant convolution on contiguous hex grids (817 cells, k_ring=16)
  2. Liquid Time-Constant (LTC) dynamics with K=4 iterations

For Hive forecasting, the cells are SCATTERED globally (Italy, Bay Area, US
East, viral expansion in 6 regions) with NO meaningful H3 adjacency. The
C6-equivariant convolution requires contiguous hex neighborhoods to operate
meaningfully, so it's removed for this task.

The LTC dynamics component is preserved. Each cell is processed independently
with a *shared* LTC predictor — same parameters across all cells, but each
cell has its own state. Periodicity features (sin/cos hour, sin/cos day-of-
week) are added as auxiliary input channels, broadcasted to all cells, to
provide the global temporal signal that real human-driven cells will exhibit.

Multi-step prediction is autoregressive: predict step t+1, feed back, predict
t+2, etc.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """Closed-form Liquid Time-Constant cell."""

    def __init__(self, feature_dim: int, n_iterations: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_iterations = n_iterations
        self.W = nn.Linear(feature_dim, feature_dim)
        self.log_tau = nn.Parameter(torch.zeros(feature_dim))
        self.A = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        h = h0
        tau = F.softplus(self.log_tau) + 0.1
        dt = 1.0 / self.n_iterations
        for _ in range(self.n_iterations):
            update = -h / tau + self.A * torch.tanh(self.W(h))
            h = h + dt * update
        return h


class LTCForecaster(nn.Module):
    """Per-cell forecaster: shared LTC dynamics across cells, periodicity injection."""

    def __init__(
        self,
        n_channels: int,
        n_periodicity: int = 4,
        feature_dim: int = 32,
        n_ltc_iterations: int = 4,
        history_length: int = 24,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_periodicity = n_periodicity
        self.feature_dim = feature_dim
        self.history_length = history_length

        encoder_input_dim = history_length * (n_channels + n_periodicity)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.ltc = LTCCell(feature_dim, n_iterations=n_ltc_iterations)
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, n_channels),
        )

    def forward_one_step(
        self,
        history: torch.Tensor,
        periodicity_history: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_cells, hist_len, n_channels = history.shape
        perio = periodicity_history.unsqueeze(1).expand(batch, n_cells, hist_len, -1)
        x = torch.cat([history, perio], dim=-1)
        x = x.reshape(batch * n_cells, -1)
        feat = self.encoder(x)
        feat = self.ltc(feat)
        residual = self.decoder(feat)
        residual = residual.view(batch, n_cells, n_channels)
        last = history[:, :, -1, :]
        return last + residual

    def forward(
        self,
        history: torch.Tensor,
        periodicity_history: torch.Tensor,
        periodicity_target: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        batch, n_cells, _, n_channels = history.shape
        preds = []
        rolling_history = history
        rolling_perio = periodicity_history

        for h in range(horizon):
            pred = self.forward_one_step(rolling_history, rolling_perio)
            preds.append(pred)
            rolling_history = torch.cat([
                rolling_history[:, :, 1:, :],
                pred.unsqueeze(2),
            ], dim=2)
            new_perio = periodicity_target[:, h:h+1, :]
            rolling_perio = torch.cat([
                rolling_perio[:, 1:, :],
                new_perio,
            ], dim=1)

        return torch.stack(preds, dim=2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
