"""
Baseline forecasters for Hive load prediction.

Each baseline takes a history window and predicts `horizon` steps ahead.
All baselines work on standardized data (z-scored per cell per channel).

Three baselines:
1. Persistence: y_hat[t+k] = y[t] for all k
2. Daily-average: y_hat[t+k] = mean of y at (time-of-day, day-of-week) in training data
3. AR(k): per-cell linear autoregression, no spatial info

H3-Oscillator must beat all three by ≥20% MAE to justify the architecture.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


# =============================================================================
# Persistence
# =============================================================================

class PersistenceBaseline:
    """y_hat[t+k] = y[t] for all k. The dumbest possible model."""

    def fit(self, *args, **kwargs) -> "PersistenceBaseline":
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Args:
            history: (batch, n_cells, history_length, n_channels)
            horizon: int

        Returns:
            (batch, n_cells, horizon, n_channels)
        """
        last = history[:, :, -1:, :]  # (batch, n_cells, 1, n_channels)
        return np.broadcast_to(
            last, (last.shape[0], last.shape[1], horizon, last.shape[3])
        ).copy()


# =============================================================================
# Daily average
# =============================================================================

class DailyAverageBaseline:
    """
    y_hat[t+k] = average of y observed at the same hour-of-week
    (hour × day-of-week) in the training data.

    Captures the diurnal + weekly pattern with no learning at all.
    """

    def __init__(self):
        # Per-cell × per (hour, weekday) lookup table
        # Shape: (n_cells, 7*24, n_channels)
        self.lookup: Optional[np.ndarray] = None
        self.n_channels: Optional[int] = None

    @staticmethod
    def _hour_of_week(timestamps) -> np.ndarray:
        """timestamps: pd.DatetimeIndex or array; returns int array 0-167."""
        return np.array([t.weekday() * 24 + t.hour for t in timestamps])

    def fit(
        self, data: np.ndarray, mask: np.ndarray, timestamps
    ) -> "DailyAverageBaseline":
        """
        Args:
            data: (n_cells, n_timesteps, n_channels)
            mask: (n_cells, n_timesteps)
            timestamps: pd.DatetimeIndex with length n_timesteps
        """
        n_cells, n_timesteps, n_channels = data.shape
        self.n_channels = n_channels
        how = self._hour_of_week(timestamps)
        self.lookup = np.zeros((n_cells, 168, n_channels), dtype=np.float32)
        counts = np.zeros((n_cells, 168), dtype=np.float32)

        for t in range(n_timesteps):
            h = how[t]
            for c in range(n_cells):
                if mask[c, t] > 0:
                    self.lookup[c, h] += data[c, t]
                    counts[c, h] += 1.0

        # Avoid div by zero — fill missing slots with cell mean
        for c in range(n_cells):
            cell_mean = data[c][mask[c] > 0].mean(axis=0) if mask[c].sum() > 0 else np.zeros(n_channels)
            for h in range(168):
                if counts[c, h] > 0:
                    self.lookup[c, h] /= counts[c, h]
                else:
                    self.lookup[c, h] = cell_mean
        return self

    def predict(
        self, history: np.ndarray, horizon: int, target_timestamps
    ) -> np.ndarray:
        """
        Args:
            history: (batch, n_cells, history_length, n_channels) — unused; just for API
            horizon: int
            target_timestamps: list of pd.DatetimeIndex (length batch),
                               each of length `horizon`, giving the timestamps
                               of the targets to predict.

        Returns:
            (batch, n_cells, horizon, n_channels)
        """
        batch, n_cells, _, n_channels = history.shape
        preds = np.zeros((batch, n_cells, horizon, n_channels), dtype=np.float32)

        for b in range(batch):
            how = self._hour_of_week(target_timestamps[b])  # (horizon,)
            for c in range(n_cells):
                preds[b, c] = self.lookup[c, how]
        return preds


# =============================================================================
# Per-cell AR(k)
# =============================================================================

class ARBaseline:
    """
    Per-cell autoregressive baseline. Each cell has its own linear AR(k) model
    per channel: y[t+1] = a_0 + sum_i a_i * y[t-i+1] + noise.

    For multi-step prediction, applies recursively.
    """

    def __init__(self, k: int = 24):
        self.k = k
        self.coefs: Optional[np.ndarray] = None  # (n_cells, n_channels, k+1)
        self.n_channels: Optional[int] = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> "ARBaseline":
        """Fit per-cell, per-channel AR(k) via least squares."""
        n_cells, n_timesteps, n_channels = data.shape
        self.n_channels = n_channels
        self.coefs = np.zeros((n_cells, n_channels, self.k + 1), dtype=np.float32)

        for c in range(n_cells):
            for ch in range(n_channels):
                y = data[c, :, ch]  # (n_timesteps,)
                # Build (X, y) where X has columns [1, y[t-1], y[t-2], ..., y[t-k]]
                # and target is y[t]
                if n_timesteps <= self.k:
                    continue
                X_rows, y_rows = [], []
                for t in range(self.k, n_timesteps):
                    if mask[c, t] == 0:
                        continue
                    history = y[t - self.k:t][::-1]  # most recent first
                    X_rows.append(np.concatenate([[1.0], history]))
                    y_rows.append(y[t])
                if len(X_rows) < self.k + 5:
                    continue
                X = np.stack(X_rows)
                y_vec = np.array(y_rows)
                # Skip if all-zero / constant (no variance to fit)
                if np.std(y_vec) < 1e-6 or np.allclose(X[:, 1:], X[:, 1:2]):
                    self.coefs[c, ch, 0] = float(np.mean(y_vec))
                    continue
                # Ridge-regularized least squares
                lam = 0.01
                A = X.T @ X + lam * np.eye(self.k + 1)
                b = X.T @ y_vec
                try:
                    self.coefs[c, ch] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.coefs[c, ch, 0] = float(np.mean(y_vec))
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Args:
            history: (batch, n_cells, history_length, n_channels)
            horizon: int

        Returns:
            (batch, n_cells, horizon, n_channels)
        """
        batch, n_cells, hist_len, n_channels = history.shape
        if hist_len < self.k:
            # Pad with first value
            pad = np.repeat(history[:, :, :1], self.k - hist_len, axis=2)
            history = np.concatenate([pad, history], axis=2)
            hist_len = self.k

        preds = np.zeros((batch, n_cells, horizon, n_channels), dtype=np.float32)
        # Roll forward step by step
        rolling = history[:, :, -self.k:, :].copy()  # (batch, n_cells, k, n_channels)

        for h in range(horizon):
            for c in range(n_cells):
                for ch in range(n_channels):
                    coef = self.coefs[c, ch]  # (k+1,)
                    # rolling[:, c, :, ch] is (batch, k); flip to most-recent-first
                    feats = rolling[:, c, ::-1, ch]  # (batch, k)
                    X = np.concatenate(
                        [np.ones((batch, 1), dtype=np.float32), feats], axis=1
                    )  # (batch, k+1)
                    preds[:, c, h, ch] = X @ coef
            # Roll: drop oldest, append predicted
            rolling = np.concatenate(
                [rolling[:, :, 1:, :], preds[:, :, h:h+1, :]], axis=2
            )
        return preds


# =============================================================================
# Evaluation
# =============================================================================

def compute_mae(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict:
    """
    Compute MAE per channel, plus overall, on masked predictions.

    Args:
        pred: (batch, n_cells, horizon, n_channels)
        target: same shape
        mask: (batch, n_cells, horizon) — 1 where target is valid

    Returns:
        dict with per-channel and overall MAE
    """
    err = np.abs(pred - target)  # (batch, n_cells, horizon, n_channels)
    mask_b = mask[..., None]  # (batch, n_cells, horizon, 1)
    n = mask_b.sum().clip(min=1)
    result = {
        "overall": float((err * mask_b).sum() / (n * pred.shape[-1])),
    }
    for ch in range(pred.shape[-1]):
        result[f"ch{ch}"] = float((err[..., ch] * mask).sum() / n)
    return result


def compute_per_horizon_mae(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    MAE broken down by horizon step.

    Returns: (horizon,) array of MAEs averaged over batch, cells, channels.
    """
    err = np.abs(pred - target)  # (batch, n_cells, horizon, n_channels)
    mask_b = mask[..., None]
    horizon = err.shape[2]
    out = np.zeros(horizon)
    for h in range(horizon):
        e = err[:, :, h, :]
        m = mask[:, :, h:h+1]
        n = m.sum().clip(min=1)
        out[h] = float((e * m[..., 0:1]).sum() / (n * err.shape[-1]))
    return out
