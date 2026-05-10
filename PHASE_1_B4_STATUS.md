# Phase 1 — B4 Hex CNN Baseline Status

**Date:** Phase 1 / Week 2 — second baseline complete.
**Context:** Continuation after `PHASE_1_B3_STATUS.md`. B4 is the second of three planned baselines (B2 transformer still TODO). Builds on Phase 0 dataset infrastructure and Phase 1 / B3 training infrastructure.

---

## What's new in this session

### 1. `src/models/hex_cnn_baseline.py` — B4 model

Static hex CNN with **direction-aware kernels** (HexagDLy-style). Each layer:
- 7 free weights per (in_channel, out_channel) pair: 1 center + 6 directional
- Each directional weight applies to the cell's neighbor at a specific compass direction (0°, 60°, 120°, 180°, 240°, 300°)
- For boundary cells with fewer than 6 valid neighbors, missing directions contribute zero

**Default config:** hidden_dim=20, n_layers=3 → **8,562 parameters** (in doc 05 §5.1.4 spec range of 5K-10K). Slightly larger than B3 (6,402) but same ballpark.

**Architectural difference from B3:** B3 (GNN) uses mean-aggregation of neighbors (direction-agnostic). B4 (hex CNN) distinguishes neighbors by compass direction. B4 is strictly more expressive at matched per-layer parameter count.

**Architectural difference from M1:** B4 has no equivariance constraint (kernel weights are independent across the 6 directions) and no dynamics layer. M1 will use the same direction-aware structure but with C6 equivariance via Cohen's kernel expansion, plus LTC dynamics on top.

### 2. `src/h3_region.py` — direction-sorted neighbor indices

Added `direction_sorted_neighbor_indices: np.ndarray (n_cells, 6)` to `H3Region`. Computed at init time:
- For each cell, get the angle of each of its 6 (or fewer) neighbors relative to cell center
- Bucket each neighbor into one of 6 directional slots (0°, 60°, ..., 300°)
- Position k stores the cell index of the neighbor in direction k×60°
- Boundary cells with fewer neighbors have -1 in missing slots

Verified on the main 817-cell region: all 721 interior cells have all 6 directional slots filled with sensible angles (~6°, 64°, 136°, 186°, 244°, 316° spacing for a sample cell — clean ~60° apart, matching hex topology).

### 3. API refactor: model.forward(x, region_tensors)

Previously each model's forward took `(x, neighbor_idx, valid_mask, n_valid)` — four separate tensors. With B4 needing direction-sorted variants, this would have required either:
- Passing 6 tensors (mess)
- Caching tensors via a side-channel `set_region_tensors()` (hacky)
- Refactoring to `model(x, region_tensors)` (clean)

Chose the refactor. Each model pulls what it needs from `region_tensors`. B3 reads `neighbor_idx`/`valid_mask`/`n_valid`; B4 reads `dir_neighbor_idx`/`dir_valid_mask`. M1 will read both plus more (e.g., chart-padding info if/when we add multi-chart in Phase 2b).

`RegionTensors` dataclass now has 5 fields, all populated automatically by `RegionTensors.from_region(region, device)`.

### 4. `scripts/train_b4_hex_cnn.py`

Mirror of `train_b3_gnn.py` with HexCNN model, default lr=1e-2 (per doc 05 §6.2: structured priors prefer higher LR), and `--sanity-check` mode.

---

## Verified

- Sanity check (smoke α, 5 epochs, ~5 sec): training works end-to-end ✓
- Medium γ run (20 epochs, ~2.5 min): clean signal ✓
- B3 retains its previous numbers after the API refactor ✓ (re-ran `--sanity-check` and got equivalent results)
- Direction-sorted indices verified for all 721 interior cells ✓

---

## B3 vs B4 at medium γ scale

This is the most informative cross-section we have so far:

| Surface | B3 (GNN) RMSE | B3 ratio | B4 (hex CNN) RMSE | B4 ratio |
|---|---|---|---|---|
| In-distribution (H=3) | 0.00698 | 0.47 | **0.00505** | **0.34** |
| OOD-temporal (H=8) | 0.01523 | 0.45 | **0.01097** | **0.33** |
| OOD-spatial | 0.00687 | 0.48 | **0.00495** | **0.34** |
| OOD-parameter (α) | **0.02809** | **14.5** | 0.03368 | 17.4 |
| OOD-parameter (δ) | **0.04553** | **2.06** | 0.05220 | 2.37 |

**Reading:**

- **B4 wins on in-distribution and spatial/temporal generalization** (~27-29% lower RMSE). Direction-aware kernels capture more detail than mean aggregation.
- **B4 loses more on cross-regime generalization** (~15-20% worse on α and δ tests). More capacity = more overfit to training distribution.
- **The tradeoff is sharp.** B4 isn't "uniformly better" or "uniformly worse" — it sits at a different point on a Pareto curve.

This is actually the **best possible setup for testing M1**. The B3 → B4 trajectory establishes a clear curve:

> *More expressive baselines → better in-distribution → worse cross-regime*

The H3-Oscillator's architectural claim is that **C6 equivariance + bounded dynamics bends this curve** — that M1 can match B4's in-distribution accuracy while approaching B3's cross-regime generalization (or even improving on B3). If true, the equivariance constraint is doing real work. If M1 sits on the same B3-B4 line, the constraint isn't paying off.

Either outcome is informative. **The experiment is well-posed.**

---

## Wall-time projection for full B4

- 7.25 sec/epoch on medium γ (CPU)
- Full dataset (10× pairs) at 30 epochs: ~36 min on CPU, ~12-15 min on MPS
- Multi-seed full (3 seeds × 2 regimes): ~1.5 hours on MPS

Slightly faster than B3 in this run (B3 was ~12 sec/epoch on CPU at medium γ). The 6 separate small linear layers in B4 parallelize well on Apple Silicon.

---

## What you do on your Mac

### Update local files (5 changed files, 1 new directory entry)

```bash
cd ~/h3_oscillator

# Pull updated files (these existed before, with API changes)
mv ~/Downloads/h3_region.py src/
mv ~/Downloads/training.py src/
mv ~/Downloads/gnn_baseline.py src/models/

# Pull new files
mv ~/Downloads/hex_cnn_baseline.py src/models/
mv ~/Downloads/train_b4_hex_cnn.py scripts/
mv ~/Downloads/PHASE_1_B4_STATUS.md .
```

### Verify B3 still works after the refactor

```bash
# Sanity check — should match the numbers you got last time
python -m scripts.train_b3_gnn --sanity-check
```

If that works, the refactor is clean.

### Run B4 sanity check, then full training

```bash
# Quick smoke test (~5 sec)
python -m scripts.train_b4_hex_cnn --sanity-check

# Full training, both regimes, single seed (~25 min total on MPS)
python -m scripts.train_b4_hex_cnn --regime gamma --dataset full --n-epochs 30
python -m scripts.train_b4_hex_cnn --regime alpha --dataset full --n-epochs 30
```

### Optional: multi-seed for B3 (if you still want it)

```bash
# Background overnight runs — gives proper error bars on B3
python -m scripts.train_b3_gnn --regime gamma --dataset full --n-epochs 30 --seeds 0 1 2
python -m scripts.train_b3_gnn --regime alpha --dataset full --n-epochs 30 --seeds 0 1 2
```

You can also do multi-seed B4 if desired. Total compute for B3 + B4, both regimes, 3 seeds each = ~6 hours on MPS.

### Visualize B4 predictions to sanity-check

```bash
python -c "
import torch, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_split
from src.models.hex_cnn_baseline import HexCNN
from src.training import RegionTensors, autoregressive_rollout
from src.visualization import plot_field

ds = load_split('data/full/gamma_test.npz')
model = HexCNN(n_features=2, hidden_dim=20, n_layers=3)
model.load_state_dict(torch.load('results/b4_gamma_full/seed_0/best_model.pt'))
model.eval()

device = torch.device('cpu')
rt = RegionTensors.from_region(ds.region, device)
inputs, targets = ds.split_input_target(input_frames=16, target_horizon=3)

x_init = torch.from_numpy(inputs[:1, -1]).float()
with torch.no_grad():
    pred = autoregressive_rollout(model, x_init, rt, n_steps=3).cpu().numpy()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for k in range(3):
    plot_field(targets[0, k, :, 1], ds.region, ax=axes[0, k], title=f'Truth t+{k+1}', show_colorbar=False)
    plot_field(pred[0, k, :, 1], ds.region, ax=axes[1, k], title=f'B4 pred t+{k+1}', show_colorbar=False)
plt.savefig('figures/b4_gamma_predictions.png', dpi=110, bbox_inches='tight')
print('saved figures/b4_gamma_predictions.png')
"
```

---

## Decisions made in this session

1. **API refactor: `model(x, region_tensors)`.** Single tensor object, models pull what they need. Adds extensibility for M1 (which will need additional info beyond what B3/B4 use).

2. **Direction sorting via global angle (atan2 in lat/lon).** Local Euclidean approximation is valid for our small region (~5° × 5°). For Phase 2b multi-chart with pentagons, we'd need proper local-frame computation, but that's deferred.

3. **60°-bucket assignment for directional slots.** Each neighbor goes to slot k that minimizes |angle - k×60°|. With collisions, the closer one wins. For the main region all interior cells have unique-slot neighbors (no collisions observed).

4. **lr=1e-2 default for B4** (per doc 05 §6.2, "Path F lesson: structured priors prefer higher LR"). B3 used lr=1e-3.

5. **No HexagDLy or external hex-CNN library.** Implemented from scratch using direction-sorted neighbor indices. Keeps dependencies minimal and code transparent (~50 lines for the conv layer).

---

## What's NOT in this session

- **B2 (transformer with H3 positional encoding)** — last baseline. Different shape: doesn't use neighbor info, instead encodes cell position. ~2 days work.
- **M1 (H3-Oscillator)** + **B5 (unconstrained LTC)** — Phase 2.
- **Full training results from real data** — for you to run on your Mac.
- **Multi-seed B3 results** — you mentioned wanting these; can run anytime.

---

## Files added/changed

```
src/h3_region.py                        (10.6 KB) — added direction-sorted indices
src/training.py                         (14.2 KB) — extended RegionTensors, refactored API
src/models/gnn_baseline.py              (6.1 KB)  — updated to new API
src/models/hex_cnn_baseline.py          (5.9 KB)  — NEW: B4 model
scripts/train_b4_hex_cnn.py             (8.0 KB)  — NEW: B4 training driver
PHASE_1_B4_STATUS.md                              — this memo
```

Total new code in this session: ~14 KB.

---

## How to commit

After running B4 training and verifying:

```bash
cd ~/h3_oscillator
git add src/h3_region.py src/training.py src/models/gnn_baseline.py
git add src/models/hex_cnn_baseline.py scripts/train_b4_hex_cnn.py
git add PHASE_1_B4_STATUS.md
git commit -m "Phase 1 / B4: static hex CNN baseline + RegionTensors API refactor

- HexCNN: direction-aware hex convolution, 7 weights per (in,out) channel
  pair, 8,562 params at default config
- H3Region: added direction_sorted_neighbor_indices for direction-aware
  kernels (and future M1 use)
- API refactor: model.forward now takes RegionTensors directly, cleaner
  and more extensible for M1
- B3 retains all previous behavior post-refactor (verified)
- Medium γ comparison vs B3: B4 ~28% better in-distribution and
  OOD-spatial/temporal, ~17% worse on OOD-parameter. Establishes the
  in-dist accuracy / cross-regime generalization tradeoff that M1 needs
  to address."
git push
```

---

## Calibration

Two baselines down (B3, B4), one to go (B2). After B4's full results come in from your Mac, we'll have a 2-row baseline table. The pattern visible at medium scale — that B4 is more accurate but less generalizing than B3 — should hold or sharpen at full scale.

The interesting thing is what M1 does. If M1 sits on the B3-B4 line (more expressive → less generalizing), the equivariance constraint isn't doing what we thought. If M1 *bends* the line — better OOD-parameter at matched in-distribution — that's the architectural claim validated.

Stay disciplined. Don't form M1 expectations from B3/B4 alone.
