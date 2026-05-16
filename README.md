# H3-Oscillator

A gauge-equivariant convolutional network with continuous-time Liquid Time-Constant dynamics, operating natively on Uber's H3 hexagonal grid.

The architecture combines two complementary inductive biases — C6 hex equivariance (from Cohen-style gauge-equivariant convolution) and bounded oscillatory dynamics (from Hasani-style Liquid Time-Constant cells) — and is designed for spatial and spatiotemporal prediction tasks on H3-indexed substrates.

## Current status

**Phase 2 complete** — architectural validation on synthetic PDE benchmark (Gray-Scott reaction-diffusion on H3 hex grids). 36 multi-seed training runs across six architecture configurations.

**Phase 3 complete** — deployment validation on synthetic distributed inference network telemetry. LTC-based forecaster beats persistence baseline by 26.3% ± 0.3% MAE on multi-cell hourly load forecasting.

Pipeline is ready to ingest real production data from H3-indexed inference networks as cohorts grow to multi-cell coverage.

## Key findings (Phase 2)

Five findings emerged from the multi-seed study:

- **F1.** C6-equivariance placed in encoder/decoder layers is a genuine generalization prior — 25-63% cross-distribution improvement over unconstrained baselines.
- **F2.** LTC dynamics recover in-distribution capacity and reduce seed-to-seed variance by 1.5-60×.
- **F3.** The same C6 constraint applied to the recurrent dynamics operator is parameter efficiency rather than generalization — Schur's lemma made visible in empirical data.
- **F4.** Capacity is non-monotonic with cross-regime performance; more parameters do not uniformly help.
- **F5.** Single-seed runs can mask bimodal training failure modes — multi-seed evaluation is non-negotiable.

Per-phase implementation memos document the experimental setup, results, and calibration moments for each step. See `PHASE_*_STATUS.md` at the repository root.

## Phase 3 deployment validation

After completing the architectural study, the architecture was applied to a deployment context: forecasting load across an H3-indexed distributed inference network. Because production cells are not geographically contiguous (Italy anchor + scattered cohort), the spatial equivariance machinery does not operate in this setting — only the LTC dynamics component does. The forecaster operates at the per-cell level with shared parameters across cells (~16.7K parameters total).

Multi-seed results (3 seeds) on synthetic 17-cell hourly load forecasting:

| Model | Test MAE | vs Persistence |
|---|---|---|
| Persistence | 0.9021 | baseline |
| Daily-average | 0.7408 | +17.9% |
| AR(k=24) | 0.6646 | +26.3% |
| **LTC forecaster** | **0.6646 ± 0.0028** | **+26.3% ± 0.3%** |

The LTC forecaster cleanly clears the 20% threshold required for deployment over the persistence baseline. It ties statistically with the AR(k=24) baseline (0.0% ± 0.4%) but dominates at every forecast horizon from t+1 through t+6, while persistence error climbs from 0.84 to 0.96.

See `PHASE_3_HIVE_SIMULATOR_STATUS.md` and `PHASE_3_LTC_FORECASTER_STATUS.md` for the full setup, calibration to production anchor cell, and per-channel breakdowns.

## Architecture summary

The full H3-Oscillator pipeline (Phase 2 configuration):

```
Input field (H3 cells, C channels)
    -> S2R encoder (scalar -> regular feature, C6-equivariant)
    -> R2R recurrent dynamics (LTC-modulated, regular -> regular)
    -> R2S decoder (regular -> scalar)
Output field (H3 cells, C channels)
```

All convolution kernels are C6-equivariant by construction — the parameterization itself enforces the symmetry, not training or regularization. Equivariance is verified numerically to ~2e-7 (float32 precision) across 5 seeds × 3 layer types.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Phase 0 sanity checks (Gray-Scott data generator + B1 baseline)
python -m scripts.phase0_sanity_check
python -m scripts.phase0_b1_verification

# Phase 2 training (single seed; for multi-seed, vary --seed)
python -m scripts.train_m1_static       # equivariant static (no dynamics)
python -m scripts.train_m1              # full H3-Oscillator (equivariance + LTC)
python -m scripts.train_b5              # unconstrained ablation

# Phase 3 synthetic Hive forecasting pipeline
python -m scripts.generate_synthetic_hive    # generate synthetic dataset
python -m scripts.train_hive_forecaster      # train LTC forecaster + baselines
python -m scripts.visualize_hive_forecast    # produce comparison figures
```

## Project structure

| Path | Purpose |
|---|---|
| `src/h3_region.py` | H3 region construction and adjacency tensors |
| `src/gray_scott_h3.py` | Gray-Scott reaction-diffusion dynamics (PDE benchmark) |
| `src/data_loader.py` | Dataset loading and batching |
| `src/training.py` | Shared training loop and evaluation harness |
| `src/visualization.py` | Plotting helpers used across phases |
| `src/models/equivariant_conv.py` | S2R, R2R, R2S C6-equivariant primitives |
| `src/models/equivariant_static.py` | Static equivariant encoder/decoder (Phase 2) |
| `src/models/oscillator_dynamics.py` | LTC dynamics layer |
| `src/models/h3_oscillator.py` | Full H3-Oscillator (equivariance + LTC) |
| `src/models/h3_oscillator_b5.py` | B5 unconstrained ablation |
| `src/models/unconstrained_conv.py` | Free directional kernels |
| `src/models/gnn_baseline.py` | B3 graph neural network baseline |
| `src/models/hex_cnn_baseline.py` | B4 hex CNN baseline |
| `src/synthetic/hive_simulator.py` | Synthetic Hive inference network simulator (Phase 3) |
| `src/hive_forecast/` | Phase 3 forecasting pipeline (data loader, baselines, LTC forecaster) |
| `scripts/` | Phase-specific runner scripts (build_dataset, training, visualization) |
| `figures/` | Generated visualizations |
| `data/`, `logs/` | Generated outputs (gitignored; regenerable) |
| `PHASE_*_STATUS.md` | Per-phase implementation memos |

## Foundational papers

- Cohen, T. et al. 2019. *Gauge Equivariant Convolutional Networks and the Icosahedral CNN.*
- Hasani, R. et al. 2021. *Liquid Time-Constant Networks.* AAAI.
- Hoogeboom, E. et al. 2018. *HexaConv.* ICLR.

## What's not yet done

For full intellectual honesty:

- **No comparison against Neural Operators** (FNO, DeepONet). These are the SOTA academic baselines for PDE prediction and the comparison is deferred but worth running.
- **No comparison against Transformers** on the Gray-Scott benchmark.
- **Real production data validation is pending.** The Phase 3 LTC forecaster has been validated only on synthetic multi-cell Hive telemetry calibrated to a single production anchor cell. Real cohort data will arrive as the H3-indexed inference network grows to multi-cell coverage.

## Relationship to companion repository

This repository sits alongside [`smm-sandbox`](https://github.com/GNS-Foundation/smm-sandbox), which explores helical and oscillatory inductive biases for reasoning tasks (Sudoku, ARC-AGI) using AKOrN-style Kuramoto networks. The two projects share an intellectual thesis — *encode geometric and resonant structure as inductive bias rather than treating it as coordinate convention* — but operate on different substrates, problem domains, and theoretical foundations.

## License

MIT. See `LICENSE`.
