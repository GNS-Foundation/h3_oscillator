#!/usr/bin/env python3
"""
Train H3-Oscillator-derived LTC forecaster on synthetic Hive data,
compare against baselines (persistence, daily-average, AR(k)).

Usage:
    python scripts/train_hive_forecaster.py [options]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hive_forecast.data_loader import (
    load_hive_data, ChannelScaler, make_windows
)
from hive_forecast.baselines import (
    PersistenceBaseline, DailyAverageBaseline, ARBaseline,
    compute_mae, compute_per_horizon_mae,
)
from hive_forecast.ltc_forecaster import LTCForecaster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic_hive")
    parser.add_argument("--bucket-minutes", type=int, default=60)
    parser.add_argument("--history-length", type=int, default=24)  # 24 hours
    parser.add_argument("--horizon", type=int, default=6)           # 6 hours ahead
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--n-ltc-iter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "hive_forecast")
    parser.add_argument("--skip-ar", action="store_true",
                        help="Skip the AR(k) baseline (slow for k=24 × 25 cells × 4 channels)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Hive Load Forecasting — LTC Forecaster vs Baselines")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Bucket: {args.bucket_minutes} min, History: {args.history_length}, "
          f"Horizon: {args.horizon}")
    print()

    # ---- Load and scale ----
    ds = load_hive_data(args.data_dir, bucket_minutes=args.bucket_minutes,
                       min_cell_activity_hours=48)
    print(f"Dataset: {ds.n_cells} cells × {ds.n_timesteps} timesteps × "
          f"{ds.n_channels} channels")

    train_ds, val_ds, test_ds = ds.temporal_split(train_frac=0.667, val_frac=0.167)
    print(f"  train: {train_ds.n_timesteps} timesteps "
          f"({train_ds.timestamps[0].date()} → {train_ds.timestamps[-1].date()})")
    print(f"  val:   {val_ds.n_timesteps} timesteps "
          f"({val_ds.timestamps[0].date()} → {val_ds.timestamps[-1].date()})")
    print(f"  test:  {test_ds.n_timesteps} timesteps "
          f"({test_ds.timestamps[0].date()} → {test_ds.timestamps[-1].date()})")

    scaler = ChannelScaler().fit(train_ds)
    train_s = scaler.transform(train_ds)
    val_s   = scaler.transform(val_ds)
    test_s  = scaler.transform(test_ds)
    print()

    # ---- Window the data ----
    print("Windowing...")
    train_w = make_windows(train_s, args.history_length, args.horizon, stride=1)
    val_w   = make_windows(val_s,   args.history_length, args.horizon, stride=1)
    test_w  = make_windows(test_s,  args.history_length, args.horizon, stride=1)
    print(f"  train windows: {train_w['history'].shape[0]}")
    print(f"  val windows:   {val_w['history'].shape[0]}")
    print(f"  test windows:  {test_w['history'].shape[0]}")
    print()

    # =========================================================================
    # Baselines
    # =========================================================================
    print("=" * 70)
    print("Computing baselines on TEST set")
    print("=" * 70)

    results = {}

    # Persistence
    print("[1/3] Persistence...")
    persist = PersistenceBaseline()
    persist_pred = persist.predict(test_w["history"], args.horizon)
    results["persistence"] = {
        "mae": compute_mae(persist_pred, test_w["target"], test_w["target_mask"]),
        "per_horizon": compute_per_horizon_mae(
            persist_pred, test_w["target"], test_w["target_mask"]
        ).tolist(),
    }
    print(f"  overall MAE: {results['persistence']['mae']['overall']:.4f}")

    # Daily average
    print("[2/3] Daily-average...")
    daily = DailyAverageBaseline()
    daily.fit(train_s.data, train_s.mask, train_s.timestamps)
    # Need target timestamps for each window
    target_ts = []
    for ws in test_w["t_start"]:
        ws_idx = list(test_s.timestamps).index(ws)
        target_start_idx = ws_idx + args.history_length
        target_ts.append(test_s.timestamps[target_start_idx:target_start_idx + args.horizon])
    daily_pred = daily.predict(test_w["history"], args.horizon, target_ts)
    results["daily_average"] = {
        "mae": compute_mae(daily_pred, test_w["target"], test_w["target_mask"]),
        "per_horizon": compute_per_horizon_mae(
            daily_pred, test_w["target"], test_w["target_mask"]
        ).tolist(),
    }
    print(f"  overall MAE: {results['daily_average']['mae']['overall']:.4f}")

    # AR(k)
    if not args.skip_ar:
        print("[3/3] AR(24)...")
        t0 = time.time()
        ar = ARBaseline(k=min(24, args.history_length))
        ar.fit(train_s.data, train_s.mask)
        ar_pred = ar.predict(test_w["history"], args.horizon)
        results["ar_k24"] = {
            "mae": compute_mae(ar_pred, test_w["target"], test_w["target_mask"]),
            "per_horizon": compute_per_horizon_mae(
                ar_pred, test_w["target"], test_w["target_mask"]
            ).tolist(),
        }
        print(f"  overall MAE: {results['ar_k24']['mae']['overall']:.4f}  "
              f"(fit+predict in {time.time()-t0:.1f}s)")
    print()

    # =========================================================================
    # LTC Forecaster
    # =========================================================================
    print("=" * 70)
    print("Training LTC Forecaster")
    print("=" * 70)

    model = LTCForecaster(
        n_channels=ds.n_channels,
        n_periodicity=train_s.periodicity.shape[1],
        feature_dim=args.feature_dim,
        n_ltc_iterations=args.n_ltc_iter,
        history_length=args.history_length,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Build PyTorch tensors
    def _to_tensors(w):
        return {
            "history": torch.from_numpy(w["history"]).float(),
            "perio_h": torch.from_numpy(w["periodicity_history"]).float(),
            "target": torch.from_numpy(w["target"]).float(),
            "perio_t": torch.from_numpy(w["periodicity_target"]).float(),
            "target_mask": torch.from_numpy(w["target_mask"]).float(),
        }
    train_t = _to_tensors(train_w)
    val_t = _to_tensors(val_w)
    test_t = _to_tensors(test_w)

    # Batched training
    def _train_epoch():
        model.train()
        n = train_t["history"].shape[0]
        idx = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            batch_idx = idx[i:i+args.batch_size]
            hist  = train_t["history"][batch_idx].to(device)
            perh  = train_t["perio_h"][batch_idx].to(device)
            pert  = train_t["perio_t"][batch_idx].to(device)
            tgt   = train_t["target"][batch_idx].to(device)
            tgt_m = train_t["target_mask"][batch_idx].to(device)
            optimizer.zero_grad()
            pred = model(hist, perh, pert, args.horizon)
            # Masked MSE
            loss = ((pred - tgt) ** 2 * tgt_m.unsqueeze(-1)).sum() / \
                   (tgt_m.sum() * pred.shape[-1] + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        return total_loss / n

    @torch.no_grad()
    def _eval(tensors, batch_size=128):
        model.eval()
        n = tensors["history"].shape[0]
        preds_list, targets_list, masks_list = [], [], []
        for i in range(0, n, batch_size):
            hist  = tensors["history"][i:i+batch_size].to(device)
            perh  = tensors["perio_h"][i:i+batch_size].to(device)
            pert  = tensors["perio_t"][i:i+batch_size].to(device)
            pred = model(hist, perh, pert, args.horizon)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(tensors["target"][i:i+batch_size].numpy())
            masks_list.append(tensors["target_mask"][i:i+batch_size].numpy())
        return (
            np.concatenate(preds_list, axis=0),
            np.concatenate(targets_list, axis=0),
            np.concatenate(masks_list, axis=0),
        )

    best_val_mae = float("inf")
    best_state = None
    history_log = []
    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val MAE':>10} {'Time':>7}")
    print("-" * 40)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = _train_epoch()
        val_pred, val_tgt, val_mask = _eval(val_t)
        val_mae = compute_mae(val_pred, val_tgt, val_mask)["overall"]
        elapsed = time.time() - t0
        print(f"{epoch+1:>5} {train_loss:>12.4f} {val_mae:>10.4f} {elapsed:>6.1f}s")
        history_log.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_mae": float(val_mae),
        })
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val MAE: {best_val_mae:.4f} (epoch "
          f"{1 + min(range(len(history_log)), key=lambda i: history_log[i]['val_mae'])})")
    model.load_state_dict(best_state)
    test_pred, test_tgt, test_mask = _eval(test_t)
    results["ltc_forecaster"] = {
        "mae": compute_mae(test_pred, test_tgt, test_mask),
        "per_horizon": compute_per_horizon_mae(test_pred, test_tgt, test_mask).tolist(),
        "n_parameters": model.count_parameters(),
        "best_val_mae": float(best_val_mae),
        "training_history": history_log,
    }
    print(f"Test MAE: {results['ltc_forecaster']['mae']['overall']:.4f}")
    print()

    # =========================================================================
    # Comparison report
    # =========================================================================
    print("=" * 70)
    print("FINAL COMPARISON (test-set MAE, standardized units)")
    print("=" * 70)
    print(f"{'Model':<25} {'Overall MAE':>12} {'vs Persistence':>16}")
    print("-" * 60)
    persist_overall = results["persistence"]["mae"]["overall"]
    for name in ["persistence", "daily_average", "ar_k24", "ltc_forecaster"]:
        if name not in results:
            continue
        mae = results[name]["mae"]["overall"]
        if name == "persistence":
            label = "(baseline)"
        else:
            improvement = (persist_overall - mae) / persist_overall * 100
            label = f"{improvement:+.1f}%"
        print(f"{name:<25} {mae:>12.4f} {label:>16}")

    print()
    print("Per-horizon MAE breakdown (LTC vs daily-average):")
    if "ltc_forecaster" in results and "daily_average" in results:
        ltc_h = results["ltc_forecaster"]["per_horizon"]
        daily_h = results["daily_average"]["per_horizon"]
        print(f"{'Step':>6} {'LTC':>10} {'Daily-avg':>12} {'LTC/Daily':>12}")
        for h in range(args.horizon):
            ratio = ltc_h[h] / daily_h[h] if daily_h[h] > 0 else float("nan")
            print(f"  t+{h+1:>2}  {ltc_h[h]:>10.4f} {daily_h[h]:>12.4f} {ratio:>12.3f}")

    print()
    print("Per-channel MAE breakdown (LTC test set):")
    if "ltc_forecaster" in results:
        for ch_idx in range(ds.n_channels):
            name = ds.channel_names[ch_idx]
            mae = results["ltc_forecaster"]["mae"].get(f"ch{ch_idx}", float("nan"))
            print(f"  {name:<20} {mae:.4f}")

    # Save results
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
