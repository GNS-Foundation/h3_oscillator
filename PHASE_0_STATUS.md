# Phase 0 Status — H3-Oscillator Implementation

**Date:** Phase 0 / Week 1 completion.
**Context:** First implementation work for `04_h3_oscillator_architecture_v0.2.md` and `05_h3_oscillator_experiment_v0.2.md`. The user is reading HexaConv (Hoogeboom et al. 2018) while this work proceeds.

---

## What Phase 0 covers

Per `05 §10`, Phase 0 / Week 1 milestone is:

> *"Can generate Gray-Scott trajectories, B1 reconstructs them perfectly, hex convolution library works."*

This document confirms the first two are done. The third (hex convolution library check) is deferred to the start of Phase 2 since it's only relevant when implementing the H3-Oscillator architecture itself.

---

## What was built

### 1. `src/h3_region.py` — H3 region setup module

`H3Region` class encapsulates a region of H3 cells with precomputed adjacency. Key features:
- Center, resolution, and k-ring define the region
- Detects and rejects pentagonal cells (per v0.1 single-chart commitment)
- Precomputes `neighbor_indices: (n_cells, 6)` for fast vectorized Laplacian
- Caches cell-center lat/lons for visualization
- Boundary-aware: cells at the region edge use only their valid (in-region) neighbors

`laplacian(field, region)` function:
- Vectorized discrete hex Laplacian using neighbor lookup
- Boundary condition: no-flux (boundary cells use only valid neighbors)
- Verified: returns 0 for constant fields, expected statistics for random fields

**Verification:** With doc 05 settings (center 45.0, 0.0, resolution 5, k-ring 16):
- 817 cells exactly (matches doc 05 spec)
- 0 pentagons
- 721 interior cells (all 6 neighbors), remainder on boundary

### 2. `src/gray_scott_h3.py` — Gray-Scott dynamics + B1 baseline

`GrayScottParams` dataclass with the four standard regimes (α, β, γ, δ) from Pearson 1993.

`gray_scott_step(u, v, region, params)` — one Forward Euler step of:
```
du/dt = D_u * Lap(u) - u*v² + F*(1-u)
dv/dt = D_v * Lap(v) + u*v² - (F+k)*v
```

`initialize_gs(...)` — creates initial conditions with `n_seeds` scattered perturbation locations:
- `n_seeds=1` for γ (mazes): patterns propagate from a single seed
- `n_seeds=12` for α (spots): spots don't propagate, need scattered seeds for region coverage
- Steady state u=1, v=0 with localized perturbations + small noise

`integrate_trajectory(...)` — generates the full ground-truth trajectory.

`predict_trajectory_handcrafted(...)` — **this is baseline B1.** Given an initial state, integrates forward using the known dynamics. Used as the upper bound on prediction accuracy.

### 3. `src/visualization.py` — H3 cell field visualization

`plot_field(field, region, ...)` — renders cells as colored polygons with matplotlib.
`plot_trajectory_panels(...)` — multi-frame grid visualization of pattern emergence.

### 4. `scripts/phase0_sanity_check.py` — pattern formation verification

Generates 5000-step trajectories in α and γ regimes, verifies pattern statistics, saves figures.

### 5. `scripts/phase0_b1_verification.py` — B1 correctness test

Confirms B1 reconstructs trajectory continuation with **zero error** (within float precision) when given the true intermediate state. This is expected since B1 IS the dynamics; the test verifies the data pipeline is deterministic and consistent.

---

## Verification results

### Gray-Scott pattern formation

**α regime (spots):** 12 stable spots distributed across the region after 5000 steps. Final v statistics: mean=0.077, std=0.119, range=[0.000, 0.397]. Spots are localized, non-propagating, stable — matches Pearson 1993 expectations.

**γ regime (mazes):** Maze-like pattern fills the entire region from a single seed. Final v statistics: mean=0.161, std=0.099, range=[0.015, 0.339]. Pattern emergence visible across the trajectory: localized seed → expanding fronts → stable maze structure. Matches expected Gray-Scott γ behavior.

See:
- `figures/gs_alpha_final.png` — final state of α regime (12 spots distributed across region)
- `figures/gs_alpha_panels.png` — α temporal evolution
- `figures/gs_gamma_final.png` — final state of γ regime (mazes filling region)
- `figures/gs_gamma_panels.png` — γ temporal evolution from single seed

### B1 verification

For both regimes, B1 prediction RMSE = 0.00e+00 across all 8 forecast frames. This confirms the data pipeline is deterministic and B1 is correctly implemented as the upper bound.

---

## Performance characteristics

On the implementation server (Linux x86_64 CPU; user's actual machine is Apple Silicon, expected similar or faster):

- Region setup (817 cells): < 1 second
- Single Gray-Scott step (817 cells, vectorized NumPy): ~0.1 ms
- 5000-step trajectory generation: ~5 seconds
- Visualization rendering: ~1 second per panel

Implication: generating the full dataset (1100 trajectories × 32 frames × 256 substeps × 2 regimes) will take approximately **5-10 minutes**. Comfortably tractable.

---

## What's next

Per `05 §10`, the immediate next milestones:

### Phase 0 remaining (this week)
- **Dataset construction script.** Generate the train/val/test datasets per `05 §4`:
  - 1000 train + 100 val + 100 test trajectories per regime (α, γ)
  - 32 frames each, sampled every 8 generator steps after 1000-step spin-up
  - Save as `.npz` files in `data/`
  - Include the OOD splits: temporal (longer horizon), parameter (different F, k), spatial (held-out region)

This should be straightforward given the working generator. Estimated: 2-3 hours of work.

### Phase 1 (next week)
- **B2: Transformer with H3 positional encoding** (~2 days)
- **B3: Standard GNN on H3 adjacency** (~1 day, PyTorch Geometric)
- **B4: Static hex CNN** (~2 days, requires hex convolution wiring)

### Phase 2 (weeks 3-4)
- **M1: H3-Oscillator architecture** — encoder, dynamics, decoder, DFT diagnostic head
- **B5: Unconstrained LTC ablation** (modification of M1)

### Phase 3 (week 5)
- Hyperparameter scan + main comparison sweep

### Phase 4 (week 6)
- Ablation analysis, DFT diagnostics, results writeup → `06_h3_oscillator_findings.md`

---

## Decisions made during Phase 0

1. **`n_seeds` per regime.** The doc 05 spec didn't differentiate; I added per-regime `init_configs` because α and γ have different propagation behavior. α needs scattered seeds; γ propagates from one. This is an implementation detail that doesn't change the experimental claim — both regimes still produce the expected pattern types.

2. **Boundary handling for Laplacian.** No-flux (boundary cells use only their in-region neighbors). This is the natural choice; alternative would be periodic BC, which doesn't fit the H3 region geometry.

3. **h3-py v4 API.** Library uses `latlng_to_cell` / `grid_disk` (v4) rather than `geo_to_h3` / `k_ring` (v3). Code is v4-native. If the user is on h3 v3, they'll need to update or rewrite the API calls.

4. **Forward Euler with dt=1.0** (matching doc 05 §2.3). Stable for the parameter regimes used. If higher accuracy is needed, RK4 is a drop-in replacement; not necessary for v0.1.

---

## Risks and open questions

1. **Trajectory diversity.** Each trajectory uses a different `seed` for initialization, but the dynamics are deterministic. We're not currently varying the perturbation pattern style across trajectories. For a robust dataset, we may want to vary `n_seeds`, `seed_radius`, and `perturbation_fraction` per trajectory. Decision deferred to Phase 0 dataset construction.

2. **Lat/lon projection for visualization.** I plot lon on x, lat on y. For our small region (~5° × 5° around 45°N), this looks fine. For larger regions or higher latitudes, equirectangular distortion would matter — irrelevant for v0.1 but worth noting for Phase 2b.

3. **Trajectory length sufficiency.** I use 5000 steps for sanity-check; doc 05 specifies 1000 spin-up + 256 recorded. The 5000-step result confirms patterns are mature; the doc 05 spec should also work but was untested. Will verify when building the dataset script.

---

## How to use these files

```bash
cd h3_oscillator
python -m scripts.phase0_sanity_check    # generates pattern figures
python -m scripts.phase0_b1_verification  # verifies B1 reconstructs exactly
```

Dependencies: `numpy`, `matplotlib`, `h3` (v4.x). All Apple Silicon-compatible.

To proceed to Phase 0 dataset construction: I'll write `scripts/build_dataset.py` that uses the working generator to produce the train/val/test splits.

---

## Calibration note

This is implementation work, not yet experimental work. No claims about the H3-Oscillator architecture's performance can be made until M1 is built and trained. The progress here is **infrastructure for the experiment**, not the experiment itself. Equivalent to having built the dataset/scoring pipeline for Phase 1 (Sudoku) before any model was trained.

The architectural Acid Test (M1 vs B5) is still 4-5 weeks of work away. Patience and discipline.
