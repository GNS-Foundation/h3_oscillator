"""
Data loader for Hive load forecasting.

Reads Parquet files matching the production hive_inference_log schema
(works identically for synthetic and real Hive data), aggregates to
fixed time buckets per cell, adds periodicity features, and returns
tensors ready for training.

Design principle: the SAME loading code must work for synthetic
(/data/synthetic_hive/) and real (/data/real_hive/) inputs. Only the
source path changes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Channel definitions
# =============================================================================

CHANNEL_NAMES = [
    "n_jobs",              # count of jobs in bucket
    "n_workers_online",    # unique worker_pks seen in bucket
    "mean_latency_ms",     # mean latency over jobs in bucket
    "user_tps",            # user-experienced tokens/sec: total_tokens_out / total_latency_s
]

# Number of periodicity features added (sin/cos hour, sin/cos day-of-week)
N_PERIODICITY = 4


# =============================================================================
# Loaded dataset
# =============================================================================

@dataclass
class HiveDataset:
    """In-memory representation of a Hive forecasting dataset."""
    timestamps: pd.DatetimeIndex      # bucket start times (UTC)
    cells: list                       # list of h3_cell IDs in row order
    channel_names: list               # ordered channel names
    data: np.ndarray                  # (n_cells, n_timesteps, n_channels)
    mask: np.ndarray                  # (n_cells, n_timesteps) — 1 if cell active
    periodicity: np.ndarray           # (n_timesteps, N_PERIODICITY)
    bucket_minutes: int               # time bucket size in minutes

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_timesteps(self) -> int:
        return len(self.timestamps)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def to_torch(self, device: str = "cpu") -> dict:
        """Convert to PyTorch tensors."""
        return {
            "data": torch.from_numpy(self.data).float().to(device),
            "mask": torch.from_numpy(self.mask).float().to(device),
            "periodicity": torch.from_numpy(self.periodicity).float().to(device),
        }

    def temporal_split(
        self,
        train_frac: float = 0.667,  # 8 weeks of 12
        val_frac: float = 0.167,    # 2 weeks of 12
    ) -> tuple:
        """
        Split this dataset temporally into (train, val, test).
        train_frac + val_frac determines train+val boundary; test is the rest.
        """
        n = self.n_timesteps
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        def _slice(start, stop):
            return HiveDataset(
                timestamps=self.timestamps[start:stop],
                cells=self.cells,
                channel_names=self.channel_names,
                data=self.data[:, start:stop, :],
                mask=self.mask[:, start:stop],
                periodicity=self.periodicity[start:stop],
                bucket_minutes=self.bucket_minutes,
            )

        return _slice(0, train_end), _slice(train_end, val_end), _slice(val_end, n)


# =============================================================================
# Main loader
# =============================================================================

def load_hive_data(
    data_dir: Path,
    bucket_minutes: int = 60,
    min_cell_activity_hours: int = 24,
    train_frac: float = 0.667,
    min_train_activity_hours: int = 100,
) -> HiveDataset:
    """
    Load Hive telemetry from Parquet files and aggregate to time buckets.

    Args:
        data_dir: directory containing inference_log/ partitions and
                  inference_stats.parquet
        bucket_minutes: time bucket size for aggregation (default 60 = hourly)
        min_cell_activity_hours: skip cells with fewer than this many hours
                  of activity overall (drops dev/test residue cells)
        train_frac: fraction of data that will be used for training (default 0.667
                  = 8 weeks of 12). Used to determine the training window.
        min_train_activity_hours: skip cells with fewer than this many hours
                  of activity within the training window. Ensures cells have
                  enough training data to compute reasonable per-cell statistics.

    Returns:
        HiveDataset with shape (n_cells, n_timesteps, n_channels)
    """
    data_dir = Path(data_dir)
    log_path = data_dir / "inference_log"

    print(f"Loading inference_log from {log_path}...")
    log = pd.read_parquet(log_path)
    # Drop partition column if present
    if "dt" in log.columns:
        log = log.drop(columns=["dt"])
    print(f"  → {len(log):,} events, {log['h3_cell'].nunique()} cells, "
          f"{log['created_at'].min()} to {log['created_at'].max()}")

    # Bucket timestamp
    bucket_freq = f"{bucket_minutes}min"
    log["bucket"] = log["created_at"].dt.floor(bucket_freq)

    # Aggregate per (cell, bucket)
    print(f"Aggregating to {bucket_minutes}-min buckets per cell...")
    agg = (
        log.groupby(["h3_cell", "bucket"])
        .agg(
            n_jobs=("id", "count"),
            n_workers_online=("worker_pk", "nunique"),
            mean_latency_ms=("latency_ms", "mean"),
            total_tokens_out=("tokens_out", "sum"),
            total_latency_ms=("latency_ms", "sum"),
        )
        .reset_index()
    )
    # user_tps: total output tokens / total wall-clock seconds
    agg["user_tps"] = (
        agg["total_tokens_out"] / (agg["total_latency_ms"] / 1000.0).clip(lower=0.001)
    )
    agg = agg.drop(columns=["total_tokens_out", "total_latency_ms"])

    # Drop cells with too little overall activity
    activity = agg.groupby("h3_cell").size()
    active_cells = activity[activity >= min_cell_activity_hours].index.tolist()
    print(f"  → {len(active_cells)} cells with ≥{min_cell_activity_hours} "
          f"active buckets overall (filtered from {agg['h3_cell'].nunique()})")
    agg = agg[agg["h3_cell"].isin(active_cells)].copy()

    # Additionally: filter cells that don't have enough activity in training window
    # Training window = first `train_frac` of the time range
    all_buckets_pre = pd.date_range(
        start=log["bucket"].min(),
        end=log["bucket"].max(),
        freq=bucket_freq,
        tz="UTC",
    )
    train_end_time = all_buckets_pre[int(len(all_buckets_pre) * train_frac)]
    train_activity = (
        agg[agg["bucket"] < train_end_time]
        .groupby("h3_cell").size()
    )
    train_active_cells = train_activity[
        train_activity >= min_train_activity_hours
    ].index.tolist()
    print(f"  → {len(train_active_cells)} cells with ≥{min_train_activity_hours} "
          f"active buckets within training window (cells activating too late dropped)")
    active_cells = sorted(set(active_cells) & set(train_active_cells))
    agg = agg[agg["h3_cell"].isin(active_cells)].copy()

    # Build the full time grid
    all_buckets = pd.date_range(
        start=log["bucket"].min(),
        end=log["bucket"].max(),
        freq=bucket_freq,
        tz="UTC",
    )
    cells_sorted = sorted(active_cells)
    n_cells, n_timesteps, n_channels = len(cells_sorted), len(all_buckets), len(CHANNEL_NAMES)

    print(f"Building dense tensor ({n_cells} cells × {n_timesteps} timesteps × "
          f"{n_channels} channels)...")

    # Pivot to wide format then to tensor
    data = np.zeros((n_cells, n_timesteps, n_channels), dtype=np.float32)
    mask = np.zeros((n_cells, n_timesteps), dtype=np.float32)

    bucket_to_idx = {b: i for i, b in enumerate(all_buckets)}
    cell_to_idx = {c: i for i, c in enumerate(cells_sorted)}

    for _, row in agg.iterrows():
        ci = cell_to_idx[row["h3_cell"]]
        ti = bucket_to_idx.get(row["bucket"])
        if ti is None:
            continue
        data[ci, ti, 0] = row["n_jobs"]
        data[ci, ti, 1] = row["n_workers_online"]
        data[ci, ti, 2] = row["mean_latency_ms"]
        data[ci, ti, 3] = row["user_tps"]
        mask[ci, ti] = 1.0

    # For inactive buckets within a cell's active range, latency/tps NaN
    # → fill with cell mean (so model doesn't see crazy values)
    for ci in range(n_cells):
        active_t = np.where(mask[ci] > 0)[0]
        if len(active_t) == 0:
            continue
        first_active = active_t[0]
        # Channels 2 and 3 (latency, tps) need fill for buckets where mask=0
        for ch in [2, 3]:
            cell_mean = data[ci, active_t, ch].mean()
            inactive_in_range = (mask[ci] == 0) & (
                np.arange(n_timesteps) >= first_active
            )
            data[ci, inactive_in_range, ch] = cell_mean

    # Periodicity features (shared across cells)
    print("Computing periodicity features (sin/cos hour, sin/cos day-of-week)...")
    hours = np.array([t.hour + t.minute / 60.0 for t in all_buckets])
    days = np.array([t.weekday() for t in all_buckets])
    periodicity = np.stack([
        np.sin(2 * np.pi * hours / 24.0),
        np.cos(2 * np.pi * hours / 24.0),
        np.sin(2 * np.pi * days / 7.0),
        np.cos(2 * np.pi * days / 7.0),
    ], axis=1).astype(np.float32)

    print(f"Done. Final shape: data {data.shape}, mask {mask.shape}, "
          f"periodicity {periodicity.shape}")
    return HiveDataset(
        timestamps=all_buckets,
        cells=cells_sorted,
        channel_names=list(CHANNEL_NAMES),
        data=data,
        mask=mask,
        periodicity=periodicity,
        bucket_minutes=bucket_minutes,
    )


# =============================================================================
# Channel scaling
# =============================================================================

class ChannelScaler:
    """
    Per-channel z-score normalization fit on training data.

    n_jobs and n_workers can be very different in magnitude across cells
    (anchor has 60, cohort has 5). We use per-cell + per-channel statistics
    for the jobs/workers channels, and global statistics for latency/tps.
    """

    def __init__(self):
        self.cell_means: Optional[np.ndarray] = None  # (n_cells, n_channels)
        self.cell_stds: Optional[np.ndarray] = None   # (n_cells, n_channels)
        # Channels 0, 1 (jobs, workers) use per-cell scaling
        # Channels 2, 3 (latency, tps) use per-cell mean but global std
        self.per_cell_channels = [0, 1, 2, 3]  # all per-cell for simplicity

    def fit(self, dataset: HiveDataset) -> "ChannelScaler":
        d = dataset.data  # (n_cells, n_timesteps, n_channels)
        m = dataset.mask[:, :, None]  # (n_cells, n_timesteps, 1)
        # Per-cell stats over time, ignoring inactive buckets
        active_counts = m.sum(axis=1).clip(min=1)
        self.cell_means = (d * m).sum(axis=1) / active_counts
        # Variance: E[X^2] - E[X]^2 over active buckets
        sq_means = ((d * m) ** 2).sum(axis=1) / active_counts
        raw_std = np.sqrt(np.clip(sq_means - self.cell_means ** 2, 0.0, None))
        # Sensible floor: at minimum 1.0 OR 1% of mean magnitude, whichever is
        # larger. This prevents division explosion for inactive cells (mean=0,
        # std=0) and constant channels (mean=k, std=0).
        floor = np.maximum(np.abs(self.cell_means) * 0.01, 1.0)
        self.cell_stds = np.maximum(raw_std, floor)
        return self

    def transform(self, dataset: HiveDataset) -> HiveDataset:
        d = (dataset.data - self.cell_means[:, None, :]) / self.cell_stds[:, None, :]
        return HiveDataset(
            timestamps=dataset.timestamps,
            cells=dataset.cells,
            channel_names=dataset.channel_names,
            data=d.astype(np.float32),
            mask=dataset.mask,
            periodicity=dataset.periodicity,
            bucket_minutes=dataset.bucket_minutes,
        )

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """scaled_data shape: (n_cells, n_timesteps, n_channels)"""
        return scaled_data * self.cell_stds[:, None, :] + self.cell_means[:, None, :]


# =============================================================================
# Windowing helpers
# =============================================================================

def make_windows(
    dataset: HiveDataset,
    history_length: int,
    horizon: int,
    stride: int = 1,
    drop_inactive: bool = True,
) -> dict:
    """
    Slice (n_cells, n_timesteps, ...) into (n_windows, n_cells, history_length, ...)
    and corresponding targets (n_windows, n_cells, horizon, n_channels).

    Args:
        dataset: HiveDataset
        history_length: number of past timesteps used as input
        horizon: number of future timesteps to predict
        stride: step between windows
        drop_inactive: if True, drop windows where ALL cells were inactive
                       at any point in the history

    Returns:
        dict with:
            'history': (n_windows, n_cells, history_length, n_channels)
            'history_mask': (n_windows, n_cells, history_length)
            'periodicity_history': (n_windows, history_length, N_PERIODICITY)
            'target': (n_windows, n_cells, horizon, n_channels)
            'target_mask': (n_windows, n_cells, horizon)
            'periodicity_target': (n_windows, horizon, N_PERIODICITY)
            't_start': (n_windows,) timestamps of window start (history begin)
    """
    n_cells, n_timesteps, n_channels = dataset.data.shape
    window_size = history_length + horizon
    starts = np.arange(0, n_timesteps - window_size + 1, stride)

    histories, history_masks, perio_hist = [], [], []
    targets, target_masks, perio_tgt = [], [], []
    t_starts = []

    for s in starts:
        h_end = s + history_length
        t_end = h_end + horizon
        hist = dataset.data[:, s:h_end, :]
        hist_mask = dataset.mask[:, s:h_end]
        tgt = dataset.data[:, h_end:t_end, :]
        tgt_mask = dataset.mask[:, h_end:t_end]

        if drop_inactive and hist_mask.sum() == 0:
            continue

        histories.append(hist)
        history_masks.append(hist_mask)
        perio_hist.append(dataset.periodicity[s:h_end])
        targets.append(tgt)
        target_masks.append(tgt_mask)
        perio_tgt.append(dataset.periodicity[h_end:t_end])
        t_starts.append(dataset.timestamps[s])

    return {
        "history": np.stack(histories),
        "history_mask": np.stack(history_masks),
        "periodicity_history": np.stack(perio_hist),
        "target": np.stack(targets),
        "target_mask": np.stack(target_masks),
        "periodicity_target": np.stack(perio_tgt),
        "t_start": np.array(t_starts),
    }


if __name__ == "__main__":
    # Smoke test
    from pathlib import Path
    data = load_hive_data(
        Path("data/synthetic_hive"),
        bucket_minutes=60,
        min_cell_activity_hours=48,
    )
    print(f"\n=== Loaded dataset ===")
    print(f"Cells: {data.n_cells}")
    print(f"Timesteps: {data.n_timesteps}")
    print(f"Channels: {data.channel_names}")
    print(f"Data range: {data.timestamps[0]} → {data.timestamps[-1]}")
    print(f"\nChannel stats (mean / std over active buckets):")
    for ci, name in enumerate(data.channel_names):
        active = data.mask > 0
        vals = data.data[:, :, ci][active]
        print(f"  {name:<20} mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")
