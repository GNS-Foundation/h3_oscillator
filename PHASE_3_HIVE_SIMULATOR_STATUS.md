# PHASE 3 — Hive Deployment Prep — Synthetic Simulator Status

**Date:** 2026-05-13
**Status:** v1 simulator complete, calibrated, tested. Ready for use as data substrate for multi-cell forecasting development.

---

## What was built

A synthetic Hive telemetry simulator that produces data matching the **exact production schema** of `hive_inference_log` (17 columns) and `hive_inference_stats` (9 columns), calibrated to the real anchor's observed statistics.

### Files

```
src/synthetic/
├── __init__.py
└── hive_simulator.py          (~650 lines, no PyTorch dep, numpy + pandas only)

scripts/
└── generate_synthetic_hive.py (CLI runner with argparse)
```

### Architecture

- **`CellConfig`** dataclass — per-cell parameters: H3 ID, timezone, arrival rate, diurnal amplitude, peak hour, worker count, latency/TPS distributions
- **`SimulatorConfig`** dataclass — global parameters: time range, cells list, seed, cross-cell correlation strength
- **`HiveSimulator`** class — discrete-event generation per timestep (60s default), per-cell Poisson arrivals modulated by diurnal/weekly cycles and a global cross-cell multiplier
- **`make_default_cohort_config(weeks=N)`** — builds the realistic cohort scenario (Italy anchor + Bay Area + US East/Midwest + viral expansion)
- **Parquet export** with optional date partitioning, matching the layout Hive's eventual export pipeline will use

---

## Calibration to real anchor

Target stats from Hive chat response (real `@hive-anchor-eu`):

| Metric | Real anchor | Synthetic anchor | Status |
|---|---|---|---|
| Jobs/hour | ~60 | 59.6 | ✓ within 1% |
| Tokens/sec (user-experienced) | ~109 | 94.2 | ⚠ within 15%, slightly low |
| Latency (ms) | ~550 | 496 | ⚠ within 10%, slightly low |
| Worker count | 1 | 1 | ✓ exact |
| Diurnal cycle | none (machine-driven) | none (flat) | ✓ exact |
| Provider mix | hive-dominant | 95% hive / 5% groq | ✓ matches |

The tok/s and latency are slightly under target. Cause: my physics has `latency = tokens_out/effective_tps + overhead`, so user-experienced TPS is necessarily a bit lower than the worker's raw TPS due to overhead. To hit exactly 109 tok/s user-experienced, the anchor's `mean_tokens_per_second` parameter should be ~120 (raw worker rate) instead of 109. Easy adjustment if needed; v1 is "close enough" for forecasting development.

### Cohort cells (synthetic, not calibrated to anything real yet)

Cohort cells produce 4-15 jobs/hour with realistic diurnal cycles (peak at 14:00-16:00 local, weekend factor 0.4-0.7), 2-4 workers per cell, latency 450-620ms, tok/s 78-103. These numbers are reasonable for human-driven chat workloads but will be re-calibrated once real cohort data is available.

---

## Cohort growth trajectory (default config, 12 weeks)

| Week | Active cells | Jobs in week | Notes |
|---|---|---|---|
| 1 | 1 | 10,379 | Anchor only |
| 2 | 4 | 14,381 | + Bay Area (Mauricio cluster) |
| 3 | 9 | 22,071 | + US East/Midwest |
| 4 | 9 | 26,851 | Cells warming up |
| 5-11 | 11-23 | ~27,000/wk | Viral expansion (~2/wk) |
| 12 | 25 | 28,700 | Target state |

By week 12: ~292,000 events across 25 cells. Plenty for forecasting training.

---

## Schema match

**`inference_log`** (17 cols, exact match to production):
```
id, h3_cell, epoch, worker_pk, job_id, requester_pk, model, provider,
tokens_in, tokens_out, latency_ms, prompt_hash, response_hash, job_hash,
stellar_tx, cost_gns, created_at
```

**`inference_stats`** (9 cols, exact match):
```
h3_cell, hour, model, provider, total_jobs, total_tokens_in,
total_tokens_out, avg_latency_ms, total_cost_gns
```

All H3 cell IDs are valid 15-char lowercase hex strings starting with `87` (res-7), matching the canonicalization the Hive chat recommended. Hashes are 64-char SHA256-format hex. Worker/requester pks have the `WK_`/`RQ_` prefix matching production conventions.

---

## Usage

### Quick test (no files written)

```bash
python scripts/generate_synthetic_hive.py --weeks 2 --summary-only
```

### Full 12-week dataset, date-partitioned (recommended)

```bash
python scripts/generate_synthetic_hive.py \
    --weeks 12 \
    --seed 42 \
    --output-dir data/synthetic_hive \
    --partition-by-date
```

Output layout:
```
data/synthetic_hive/
├── inference_log/
│   ├── dt=2026-05-13/part.parquet
│   ├── dt=2026-05-14/part.parquet
│   └── ... (one per day)
├── inference_stats.parquet
└── meta.json
```

### Loading in training code

```python
import pandas as pd

# Partitioned per-event log (large, use only if you need per-event grain)
log = pd.read_parquet('data/synthetic_hive/inference_log')

# Pre-aggregated hourly stats (smaller, right grain for forecasting)
stats = pd.read_parquet('data/synthetic_hive/inference_stats.parquet')

# Same call works for real Hive Parquet exports — only the source dir changes
```

---

## What it explicitly is NOT

To honor the calibration discipline established in Phase 2:

- **NOT a model of real Hive user behavior.** Cohort cells follow a *plausible* diurnal/weekly pattern based on chat-app heuristics, not measured user data. Real cohort data may surprise us.
- **NOT a real-data substitute.** Models trained on synthetic data must be retrained on real data before any production claim can be made. The synthetic phase tests *infrastructure*, not *the architecture's real-world performance*.
- **NOT validated against real multi-cell statistics.** The 1-cell anchor calibration is the only real anchor point. Everything else is forward-looking modeling.

---

## What it IS

- **A schema-faithful data source** for developing the pipeline (data loading → preprocessing → training → inference → integration) at realistic scale (~25 cells, ~290k events) before real multi-cell data exists.
- **A controlled-experiment substrate** for testing architectural variants (does H3-Oscillator's spatial structure help on plausible patterns? does periodicity injection improve results?) with known ground truth.
- **A pipeline that drop-in replaces with real data** once it's available — same schema, same loading code, only the data source changes.

---

## Performance

- Generating 2 weeks (~24k events, 4 cells): **3 seconds** on sandbox CPU
- Generating 12 weeks (~290k events, 25 cells): **~30 seconds** estimated (scales roughly linearly)
- Memory: events held in Python list until DataFrame assembly; 290k events ≈ 200 MB peak. Fine for development; if scaling to 100k+ cells we'd want chunked output.

---

## Next steps

In priority order:

1. **Submit Hive integration brief** to the Hive chat (`08_hive_integration_brief.md`). Wait for their data export + res 7 canonicalization. Use their answers to fine-tune simulator parameters (especially: model distribution, requester patterns, real anchor's tok/s and latency to 2 decimal places).

2. **Build the unified data loader** that reads from either real Parquet or synthetic Parquet via the same interface. Goal: training code is identical whether data comes from `/data/synthetic_hive/` or `/data/real_hive/`. This is the *pipeline-validation* deliverable.

3. **Adapt H3-Oscillator architecture for the Hive forecasting task:**
   - Multi-channel input (4 channels recommended by Hive chat: `n_jobs_completed`, `n_workers_online`, `mean_latency_ms`, `mean_tokens_per_second`)
   - Periodicity injection (time-of-day, day-of-week as sin/cos features added to input channels)
   - Multi-step rollout for 30-min ahead at 5-min resolution = 6 steps
   - Adapt H3-cell adjacency structure for 25 cells (vs 817 we used in Gray-Scott — much sparser; need to handle disconnected cell components, since synthetic cohort isn't geographically contiguous)

4. **Train on synthetic, validate against baselines:**
   - Persistence baseline (predict next = current)
   - Daily-average baseline (predict next = average for this time-of-day)
   - Linear AR(k) per cell + spatial neighbors
   - H3-Oscillator must beat (3) by ≥20% to justify the architecture for the Hive use case

5. **In parallel: Phase 0 anomaly detection on real 1-cell data** per the Hive chat's plan. This is the *real production deployment* track. Synthetic work is preparation for Phase 1.

---

## Honest caveats for the eventual writeup

When writing this work up, the framing should be:

> "We developed a synthetic Hive load simulator calibrated to the production anchor's observed statistics, used to develop and stress-test the H3-Oscillator multi-cell forecasting pipeline before real multi-cell data was available. Models trained on synthetic data were re-trained on real data once the cohort provided sufficient spatial coverage. We report results from real-data training; the synthetic phase was infrastructure preparation."

The synthetic phase is *engineering* preparation. The real-data phase is *scientific* validation. Don't conflate them.

---

*Status memo prepared 2026-05-13.*
