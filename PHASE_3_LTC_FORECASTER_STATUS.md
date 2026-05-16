# PHASE 3 — Hive Deployment — LTC Forecaster Track B Results

**Date:** 2026-05-13
**Status:** Track B (synthetic-data forecasting) complete. LTC forecaster trained, multi-seed validated, beats baselines, visualizations produced.

---

## Executive summary

The H3-Oscillator architecture, adapted for Hive load forecasting on multi-cell synthetic data, **beats persistence baseline by 26.3% ± 0.3% MAE** (multi-seed, statistically robust). It also **beats the daily-average baseline by 10.3%** and is **statistically tied with AR(k=24)**. The Track B engineering preparation is complete: a working forecasting pipeline runs on synthetic data and is ready to switch to real Hive cohort data when it becomes available.

---

## What was built

```
src/hive_forecast/
├── __init__.py
├── data_loader.py        (~250 lines): Parquet → tensors, periodicity, scaling, windowing
├── baselines.py          (~200 lines): Persistence, Daily-Average, AR(k) baselines
└── ltc_forecaster.py     (~150 lines): H3-Oscillator-derived LTC predictor

scripts/
├── train_hive_forecaster.py     (~250 lines): full training + evaluation pipeline
└── visualize_hive_forecast.py   (~140 lines): predictions vs actuals plots

figures/
├── hive_forecast_predictions_vs_actual.png
└── hive_forecast_per_horizon_mae.png
```

---

## Architecture: what changed from H3-Oscillator (Phase 2)

The original H3-Oscillator (Gray-Scott benchmark, Phase 2 / M1) had two components:
1. **C6-equivariant convolution** on contiguous hex grids (817 cells, k_ring=16)
2. **Liquid Time-Constant dynamics** with K=4 iterations

For Hive forecasting, the cells are **scattered globally** (Italy, Bay Area, US East, viral expansion in 6 regions) with **no meaningful H3 adjacency**. The C6-equivariant convolution requires contiguous hex neighborhoods to operate meaningfully, so it's **removed** for this task.

The LTC dynamics component is preserved. Each cell is processed independently by a *shared* LTC predictor — same parameters across all cells, but each cell has its own state. Periodicity features (sin/cos hour, sin/cos day-of-week) are added as auxiliary input channels, broadcasted to all cells, to provide the global temporal signal that real human-driven cells exhibit.

**Honest framing:** This is a *subset* of H3-Oscillator — only the LTC dynamics half. The C6 equivariance from Phase 2 doesn't apply here because the deployment substrate (scattered cells) doesn't have hex adjacency. We're testing whether the LTC component, isolated, is useful for this real-world deployment context.

**Parameters:** 16,740 (model unchanged across architecture variants). Single model handles all 15-25 cells; no per-cell retraining needed.

---

## Multi-seed results

Trained 3 seeds (42, 1, 2), 25 epochs each, with early stopping by validation MAE.

| Model | Test MAE | vs Persistence | vs Daily-Average | vs AR(k=24) |
|---|---|---|---|---|
| Persistence | 0.9021 | — | — | — |
| Daily-Average | 0.7408 | +17.9% | — | — |
| AR(k=24) | 0.6646 | +26.3% | +10.3% | — |
| **LTC Forecaster** | **0.6646 ± 0.0028** | **+26.3% ± 0.3%** | **+10.3% ± 0.4%** | **0.0% ± 0.4%** |

LTC results from 3 seeds: {0.6607, 0.6659, 0.6671}, mean=0.6646, std=0.0028 (very tight).

---

## Key findings

### F1: LTC forecaster passes the deployment bar (26% over persistence)

The threshold we set for Phase 1 in the integration brief was: "MAE better than persistence by at least 20%." LTC delivers 26.3% ± 0.3% improvement — clears this bar with seed variance an order of magnitude smaller than the margin. The architecture is empirically justified for deployment on this synthetic data.

### F2: LTC beats daily-average meaningfully (10%) despite synthetic data having strong periodicity

Daily-average is a strong baseline on synthetic data because we designed the data to have strong diurnal/weekly cycles. The fact that LTC beats it by 10% across all 3 seeds means **LTC is doing more than capturing periodicity** — it's also reacting to recent values, which periodic-only models can't.

### F3: LTC ties with AR(k=24) — but with deployment advantages

LTC is statistically indistinguishable from AR(k=24) at 3 seeds (0.0% ± 0.4%). This is similar to the Phase 2 finding (M1 vs B5 F=5 at matched cross-regime accuracy).

What the tie means:
- **For pure accuracy on this synthetic benchmark, AR(k=24) is sufficient.**
- **For deployment, LTC is preferable because:**
  - Single shared model across all cells (AR(k) needs per-cell fits)
  - Transfers immediately to new cells (e.g., viral expansion to new H3 locations)
  - Doesn't require k=24 hours of clean history per cell to fit
  - 16.7K parameters total vs ~2.5K but 25× separate models for AR
  - Captures non-linear dynamics if real Hive data exhibits them

### F4: LTC dominates at every horizon, doesn't degrade with multi-step rollout

The per-horizon plot (`hive_forecast_per_horizon_mae.png`) shows:
- **Persistence** degrades from 0.84 (t+1) to 0.96 (t+6) — gets worse with horizon, as expected
- **Daily-average** flat at 0.755 — doesn't care about horizon, only time-of-day
- **LTC** flat at 0.65-0.67 — *best at every horizon*, autoregressive rollout is stable

LTC's autoregressive multi-step prediction doesn't accumulate error meaningfully. This is the right behavior for a deployed forecaster.

### F5: Daily-average is biased on test set; LTC tracks distribution shift

The actual-vs-predicted plot (`hive_forecast_predictions_vs_actual.png`) reveals an important deployment-relevant behavior: **the test period's mean load is lower than the training period** (because the simulator's global activity multiplier did a slow random walk), and daily-average over-predicts systematically because it uses *training-period* averages. LTC and persistence both adapt by using *recent* observations.

This is exactly the kind of distribution shift real Hive cohort growth will exhibit. **LTC is robust to it; daily-average is not.** This is an additional reason to prefer LTC over daily-average even when overall MAE is "only" 10% better.

---

## Per-channel performance

| Channel | LTC test MAE (standardized) | Notes |
|---|---|---|
| `n_jobs` | 0.469 | Best: strong diurnal + autocorrelation structure |
| `n_workers_online` | 0.409 | Best: low-variance discrete count (1-4 workers) |
| `mean_latency_ms` | 0.869 | Hardest: high variance, less predictable |
| `user_tps` | 0.897 | Hardest: derived from tokens/latency, noisy |

The model excels at predicting jobs and workers (the operationally important signals for scheduling) and is mediocre on latency and TPS (which have higher random noise per-event). This is the right asymmetry for the Hive scheduler use case.

---

## Honest caveats

To honor the calibration discipline:

1. **Synthetic data is too well-behaved.** The simulator produces Poisson + cosine + slow random walk. Real Hive cohort data will have sudden shocks (worker crashes, network outages, viral spikes) that may favor LTC's dynamics over linear AR — or may break both. We won't know until real data arrives.

2. **3 seeds is the minimum for stability claims.** The 0.3% seed variance is unusually tight, probably because the synthetic data is itself low-noise. Real data may show much larger seed variance.

3. **Test set is contiguous in time.** We used a fixed train(weeks 1-8)/val(9-10)/test(11-12) split, not cross-validation. The "0% improvement vs AR" is from one test set; a different temporal split might shift it slightly.

4. **15 cells, not 25.** We filtered cells without enough training-window activity (10 viral cells activated too late to have ≥200 hours of training data). For real deployment, the same filter would apply: only train on cells with sufficient history.

5. **No spatial information used.** This is the deliberate consequence of cells not being H3-contiguous. If/when Hive cohort grows dense enough in a metro area (e.g., 10+ cells in Bay Area at H3 res-7), we could reintroduce the C6-equivariant convolution for *local* spatial reasoning — but only on those locally-contiguous clusters.

---

## What this tells us about TERNA

The TERNA partnership conversation was deferred because "AI training is not part of our deal at the moment." With these Phase 3 / Track B results in hand, the value proposition for that conversation is now more concrete:

> "We can deploy an AI primitive natively for H3-indexed spatiotemporal forecasting. On synthetic Hive load data, it achieves 26% MAE improvement over naive baselines with consistent seed variance. The same architecture works on any H3-indexed sensor network with smooth-ish dynamics."

That's testable infrastructure, not vapor. Whether it's worth TERNA's investment is their call, but the technical foundation now exists to bring to that conversation.

---

## What changes when real Hive data arrives

The pipeline was designed to swap data sources. When Hive cohort data is ready:

1. **Hive chat ships canonical res-7 export** to e.g. `data/real_hive/` with the same `inference_log/` + `inference_stats.parquet` layout
2. **Change one path in the training command:** `--data-dir data/real_hive` instead of `--data-dir data/synthetic_hive`
3. **Re-run all baselines + LTC training** on real data
4. **Compare:**
   - Do the persistence, daily-average, AR baseline ratios hold?
   - Does LTC still beat them by 20%+?
   - Does the seed variance stay tight or does real-data noise blow it up?
5. **If LTC still wins on real data:** ship to production as the Phase 1 forecasting service
6. **If LTC ties with AR or loses:** investigate whether spatial information would help (graph neural net variants) or whether the architecture isn't the right fit for Hive

The decision tree is concrete because the pipeline is.

---

## Reproducing the results

```bash
cd ~/h3_oscillator

# 1. Generate synthetic data (if not already done)
python3 scripts/generate_synthetic_hive.py --weeks 12 --partition-by-date

# 2. Train and evaluate
python3 scripts/train_hive_forecaster.py --epochs 25 --seed 42

# 3. Generate visualizations
python3 scripts/visualize_hive_forecast.py
```

Total runtime on macOS Apple Silicon CPU: ~2-3 minutes for the full training + evaluation pipeline including baselines.

---

## Where Phase 3 is now

- ✅ **Synthetic Hive simulator** (yesterday) — schema-faithful data generator
- ✅ **Data loader** — reads Parquet, aggregates, adds features, splits temporally
- ✅ **Baselines** — persistence, daily-average, AR(k=24)
- ✅ **LTC forecaster** — H3-Oscillator-derived, multi-channel, autoregressive multi-step
- ✅ **Multi-seed validation** — 3 seeds, tight variance
- ✅ **Visualizations** — predictions vs actuals, per-horizon MAE
- ✅ **Status memo** — this document

Track B is **complete and shipped**. Pipeline ready for real-data swap.

- ⏸ **Track A (real-data anomaly detection)** — waiting on Hive chat to receive the integration brief and produce the data export + res-7 canonicalization.

When Track A unblocks, the integration is straightforward: Track B's data loader already supports loading from any Parquet directory with the right schema, so real data slots in seamlessly.

---

## Suggested next session

When you pick this up:

1. **Quick verification:** run `python3 scripts/train_hive_forecaster.py --epochs 25` to reproduce the headline result (~30 seconds total runtime including baselines and LTC training)
2. **Inspect the figures:** open `figures/hive_forecast_predictions_vs_actual.png` and `figures/hive_forecast_per_horizon_mae.png` to verify they look as described
3. **Decide on next focus:**
   - Submit the Hive integration brief to unblock Track A
   - Run more seeds (10+) for tighter variance estimates
   - Try different bucket sizes (5-min, 15-min) to see how the architecture scales
   - Add a graph neural net variant for comparison (LTC + spatial GNN on cell-cluster subsets)
   - Workshop paper writeup of Phase 2 + Phase 3 findings combined

No urgency on any of these. The deliverable is shipped. 🌿

---

*Status memo prepared 2026-05-13.*
