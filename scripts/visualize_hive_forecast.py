#!/usr/bin/env python3
"""Visualize H3-Oscillator-derived LTC forecaster predictions vs baselines."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hive_forecast.data_loader import load_hive_data, ChannelScaler, make_windows
from hive_forecast.baselines import (
    PersistenceBaseline, DailyAverageBaseline,
)
from hive_forecast.ltc_forecaster import LTCForecaster


def main():
    HISTORY = 24
    HORIZON = 6
    FEATURE_DIM = 32
    EPOCHS = 25
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading...")
    ds = load_hive_data(ROOT / "data" / "synthetic_hive", bucket_minutes=60)
    train_ds, val_ds, test_ds = ds.temporal_split(0.667, 0.167)
    scaler = ChannelScaler().fit(train_ds)
    train_s = scaler.transform(train_ds)
    val_s   = scaler.transform(val_ds)
    test_s  = scaler.transform(test_ds)

    train_w = make_windows(train_s, HISTORY, HORIZON, stride=1)
    val_w   = make_windows(val_s,   HISTORY, HORIZON, stride=1)
    test_w  = make_windows(test_s,  HISTORY, HORIZON, stride=1)

    print("Training quick LTC model...")
    model = LTCForecaster(
        n_channels=ds.n_channels,
        n_periodicity=train_s.periodicity.shape[1],
        feature_dim=FEATURE_DIM,
        history_length=HISTORY,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_val = None, float("inf")

    for epoch in range(EPOCHS):
        model.train()
        idx = np.random.permutation(train_w["history"].shape[0])
        for i in range(0, len(idx), 64):
            bi = idx[i:i+64]
            hist  = torch.from_numpy(train_w["history"][bi]).float()
            perh  = torch.from_numpy(train_w["periodicity_history"][bi]).float()
            pert  = torch.from_numpy(train_w["periodicity_target"][bi]).float()
            tgt   = torch.from_numpy(train_w["target"][bi]).float()
            tgt_m = torch.from_numpy(train_w["target_mask"][bi]).float()
            optimizer.zero_grad()
            pred = model(hist, perh, pert, HORIZON)
            loss = ((pred - tgt) ** 2 * tgt_m.unsqueeze(-1)).sum() / \
                   (tgt_m.sum() * pred.shape[-1] + 1e-6)
            loss.backward()
            optimizer.step()
        # Val
        model.eval()
        with torch.no_grad():
            v_hist = torch.from_numpy(val_w["history"]).float()
            v_perh = torch.from_numpy(val_w["periodicity_history"]).float()
            v_pert = torch.from_numpy(val_w["periodicity_target"]).float()
            v_pred = model(v_hist, v_perh, v_pert, HORIZON).numpy()
            v_mae = np.abs(v_pred - val_w["target"]).mean()
        if v_mae < best_val:
            best_val = v_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best val MAE: {best_val:.4f}")

    # Run all models on test
    print("Running all models on test set...")
    model.eval()
    with torch.no_grad():
        t_hist = torch.from_numpy(test_w["history"]).float()
        t_perh = torch.from_numpy(test_w["periodicity_history"]).float()
        t_pert = torch.from_numpy(test_w["periodicity_target"]).float()
        ltc_pred = model(t_hist, t_perh, t_pert, HORIZON).numpy()

    persist_pred = PersistenceBaseline().predict(test_w["history"], HORIZON)
    daily = DailyAverageBaseline()
    daily.fit(train_s.data, train_s.mask, train_s.timestamps)
    target_ts = []
    for ws in test_w["t_start"]:
        ws_idx = list(test_s.timestamps).index(ws)
        target_start_idx = ws_idx + HISTORY
        target_ts.append(test_s.timestamps[target_start_idx:target_start_idx + HORIZON])
    daily_pred = daily.predict(test_w["history"], HORIZON, target_ts)

    # Inverse-transform back to original units for plotting
    def inv(arr):
        # arr: (n_windows, n_cells, horizon, n_channels) — apply per-cell scaling
        return arr * scaler.cell_stds[None, :, None, :] + scaler.cell_means[None, :, None, :]

    test_tgt_orig = inv(test_w["target"])
    ltc_pred_orig = inv(ltc_pred)
    persist_pred_orig = inv(persist_pred)
    daily_pred_orig = inv(daily_pred)

    # Pick a few representative cells: anchor + a few cohort cells
    cells_to_plot = [
        (0, "Cell 0 (anchor or first sorted)"),
        (5, "Cell 5"),
        (10, "Cell 10"),
    ]

    # For each cell, plot t+1 predictions vs actuals over the test period
    fig, axes = plt.subplots(len(cells_to_plot), 1, figsize=(14, 10), sharex=False)
    if len(cells_to_plot) == 1:
        axes = [axes]

    for ax, (ci, label) in zip(axes, cells_to_plot):
        if ci >= ds.n_cells:
            continue
        n_jobs_target = test_tgt_orig[:, ci, 0, 0]  # (n_windows,) — t+1, channel 0 (n_jobs)
        n_jobs_ltc = ltc_pred_orig[:, ci, 0, 0]
        n_jobs_persist = persist_pred_orig[:, ci, 0, 0]
        n_jobs_daily = daily_pred_orig[:, ci, 0, 0]

        x = np.arange(len(n_jobs_target))
        ax.plot(x, n_jobs_target, label="Actual", color="black", linewidth=1.2, alpha=0.85)
        ax.plot(x, n_jobs_ltc, label="LTC forecaster", color="C0", linewidth=1.0, alpha=0.85)
        ax.plot(x, n_jobs_persist, label="Persistence", color="C1", linewidth=0.7, alpha=0.6)
        ax.plot(x, n_jobs_daily, label="Daily-average", color="C2", linewidth=0.7, alpha=0.6)
        ax.set_title(f"{label}: {ds.cells[ci]} — predicting n_jobs/hour 1-step-ahead on test set")
        ax.set_xlabel("Test-set timestep (hourly)")
        ax.set_ylabel("Jobs/hour")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ROOT / "figures" / "hive_forecast_predictions_vs_actual.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # Also: per-horizon error plot
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = np.arange(1, HORIZON + 1)
    ltc_h = np.abs(ltc_pred - test_w["target"]).mean(axis=(0, 1, 3))
    persist_h = np.abs(persist_pred - test_w["target"]).mean(axis=(0, 1, 3))
    daily_h = np.abs(daily_pred - test_w["target"]).mean(axis=(0, 1, 3))
    ax.plot(horizons, ltc_h, "o-", label=f"LTC forecaster (test MAE={ltc_h.mean():.3f})", linewidth=2)
    ax.plot(horizons, persist_h, "s-", label=f"Persistence (test MAE={persist_h.mean():.3f})", linewidth=1.5)
    ax.plot(horizons, daily_h, "^-", label=f"Daily-average (test MAE={daily_h.mean():.3f})", linewidth=1.5)
    ax.set_xlabel("Prediction horizon (hours ahead)")
    ax.set_ylabel("Test MAE (standardized units)")
    ax.set_title("Prediction error vs horizon — all channels combined")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = ROOT / "figures" / "hive_forecast_per_horizon_mae.png"
    plt.savefig(out2, dpi=130)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
