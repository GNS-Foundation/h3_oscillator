# Phase 1 — B3 GNN Baseline Status

**Date:** Phase 1 / Week 2 — first baseline (B3) implementation complete.
**Context:** First baseline of three planned (B2 transformer, B3 GNN, B4 hex CNN). Builds on Phase 0 dataset infrastructure. Per doc 05 §10, B3 is the shortest baseline (~1 day) and the natural starting point because it directly uses the H3 hex adjacency we already built.

---

## What's new in this session

### 1. `src/models/gnn_baseline.py` — B3 model

Standard GraphSAGE-mean-style GNN on H3 hex adjacency, implemented from scratch (no PyTorch Geometric dependency — saves install complexity, uses our existing `H3Region.neighbor_indices` directly).

**Architecture:**
- Embedding: linear from (u, v) → hidden_dim
- N message-passing layers, each:
  - Mean aggregate over hex neighbors (boundary-aware, handles -1 sentinels)
  - Concat self with aggregated neighbors
  - GELU-activated linear update
  - Optional residual connection
- Decoder: linear from hidden_dim → (Δu, Δv)
- **Residual prediction:** `x_{t+1} = x_t + delta`, makes identity easy to learn

**Default size:** hidden_dim=32, n_layers=3 → **6,402 parameters** (in doc 05 spec range of 5K-10K for B3).

### 2. `src/training.py` — shared training infrastructure

This is the foundation that **all subsequent models** will reuse (B2, B4, M1, B5). Key components:

- `SingleStepPairs(Dataset)` — extracts (frame_t, frame_{t+1}) pairs from trajectory data. With 1000 train trajectories × 31 pairs each = 31,000 supervised single-step examples per epoch.
- `RegionTensors` — caches neighbor indices, valid mask, and valid count on the active device (CPU or MPS).
- `evaluate_rollout(model, dataset, region_tensors, horizon)` — autoregressive rollout from frame 16, measures RMSE against true future frames.
- `train_model(...)` — full training loop with AdamW, gradient clipping, early stopping, best-checkpoint tracking, CSV logging.
- `evaluate_all_surfaces(...)` — runs all four OOD surfaces from doc 05 §4.2.

The training loop assumes the model exposes `forward(x, neighbor_idx, valid_mask, n_valid)` as its API. This is the contract that B3, B4, M1, B5 will all satisfy. (B2 transformer doesn't need neighbor info — it gets H3-positional encoding instead — so B2's training will be a slight variation on this loop.)

### 3. `scripts/train_b3_gnn.py` — runnable training script

Reads from `data/<dataset>/`, trains B3, evaluates on all four surfaces, saves to `results/b3_<regime>_<dataset>/seed_<N>/`.

Modes:
- `--sanity-check` — smoke dataset, 5 epochs, ~10 sec total
- Custom `--regime`, `--dataset`, `--seeds` for full runs

Auto-detects MPS device on Apple Silicon (falls back to CPU if not available or in sanity mode).

---

## Sanity-check results

### Smoke dataset, α regime, 5 epochs (~10 sec)

Verifies the training loop works end-to-end. Loss drops from 0.0174 → 0.00001 in 5 epochs. With only 10 training trajectories, the model overfits and α has very stable spots so persistence is a strong baseline. Diagnostic only.

### Medium dataset, γ regime, 20 epochs (~4 min)

This run shows **genuine learning**, not just convergence:

| Surface | B3 RMSE | Persistence RMSE | Ratio |
|---|---|---|---|
| In-distribution (H=3) | 0.00698 | 0.01489 | **0.47** ← B3 beats persistence by 2× |
| OOD-temporal (H=8) | 0.01523 | 0.03363 | **0.45** ← stable advantage at longer horizons |
| OOD-spatial (held-out region) | 0.00687 | 0.01442 | **0.48** ← same as ID, excellent generalization |
| OOD-parameter (α regime) | 0.02809 | 0.00194 | 14.5 ← much worse than persistence |
| OOD-parameter (δ regime, novel) | 0.04553 | 0.02206 | 2.06 ← worse than persistence |

**Key signals:**

1. **B3 beats persistence by ~2× on in-distribution data.** The model is genuinely learning γ-regime dynamics, not just predicting "no change."

2. **B3 generalizes well across spatial OOD.** The held-out region (Missouri) gives nearly identical RMSE to in-distribution (France). This makes sense — the GNN uses purely relative neighbor structure, not absolute geographic features. So it transfers cleanly.

3. **B3 generalizes reasonably to longer horizons (OOD-temporal, H=8).** The 0.45 ratio at H=8 vs 0.47 at H=3 means the model isn't catastrophically failing on rollout — it's just gradually accumulating error.

4. **B3 fails dramatically on OOD-parameter.** Trained on γ (mazes, dynamic), tested on α (spots, mostly stable) — the model "thinks everything is dynamic" and generates large changes that don't match α's stable spots. Persistence wins easily.

   This is **exactly the inductive-bias gap that M1 (H3-Oscillator) is designed to address.** If M1 transfers better across regimes than B3 does — at matched parameter count — that's empirical evidence the equivariance constraint is buying generalization beyond what graph structure alone provides.

### Wall-time projection for full training

- 12 sec/epoch on medium γ (3,100 train pairs)
- Full dataset has 10× pairs (31,000)
- 60 epochs × 120 sec/epoch ≈ **2 hours per seed per regime** on CPU
- Apple Silicon MPS may be 2-3× faster
- 3 seeds × 2 regimes = ~12 hours total CPU, ~4-6 hours on MPS

Realistic to run overnight. Can also run with fewer epochs (model converged ~epoch 17 on medium) — `--n-epochs 30` should be sufficient with early stopping.

---

## What you do on your Mac

### Quick verification (recommended first)

```bash
cd ~/h3_oscillator
source .venv/bin/activate

# Pull the new files into your repo (assuming downloads at ~/Downloads)
mv ~/Downloads/training.py src/
mkdir -p src/models
mv ~/Downloads/__init__.py src/models/  # may need to overwrite an empty one
mv ~/Downloads/gnn_baseline.py src/models/
mv ~/Downloads/train_b3_gnn.py scripts/

# Sanity check: ~10 sec, verifies your environment matches sandbox
python -m scripts.train_b3_gnn --sanity-check
```

If sanity check passes (loss drops, model saves a checkpoint), proceed to full training.

### Full training (overnight)

```bash
# Train B3 on alpha regime, 3 seeds (~6 hours on MPS, ~12 hours CPU)
python -m scripts.train_b3_gnn --regime alpha --dataset full --seeds 0 1 2

# Train B3 on gamma regime, 3 seeds (~6 hours)
python -m scripts.train_b3_gnn --regime gamma --dataset full --seeds 0 1 2
```

Each command produces `results/b3_<regime>_full/seed_<N>/` containing:
- `best_model.pt` — best-validation checkpoint
- `train_log.csv` — per-epoch metrics
- `train_history.json` — full history
- `eval_summary.json` — multi-surface evaluation

Plus `all_seeds_summary.json` aggregating across seeds with mean ± std for each surface.

### Faster alternative (single seed for now)

If you want a quick read on whether B3 works at full scale before committing to the full multi-seed run:

```bash
python -m scripts.train_b3_gnn --regime gamma --dataset full --n-epochs 30
```

~1 hour on MPS, gives you the headline results for one seed. Multi-seed for proper error bars can come later.

---

## Decisions made in this session

1. **Built training infrastructure as `src/training.py`** rather than putting it in the script. This module will be reused by all subsequent models (B2, B4, M1, B5). Good investment — saves rewriting later.

2. **Single-step prediction with autoregressive rollout** rather than direct multi-step prediction. Standard for GNN dynamics prediction. Trains stably, maximizes data usage (31 pairs per trajectory).

3. **Residual prediction (`x_{t+1} = x_t + delta`)** rather than absolute. Makes identity easy to learn, helps stability. Standard trick for dynamical systems.

4. **No PyTorch Geometric dependency.** Implemented message passing directly using H3Region.neighbor_indices. Saves a substantial dependency, keeps the code transparent, and the implementation is ~30 lines.

5. **MPS device detection** with CPU fallback. Apple Silicon should use MPS for substantial speedup; sanity-check mode forces CPU for portability/debugging.

6. **`--sanity-check` mode** as a fast-path that uses the smoke dataset and 5 epochs. Standard tooling pattern: any change to the training code can be tested in 10 seconds before launching a 6-hour run.

---

## What's NOT in this session

- **B2 (transformer)** — next baseline. Needs custom positional encoding for H3 cell IDs. ~2-3 days work.
- **B4 (static hex CNN)** — third baseline. Needs hex convolution implementation (HexagDLy or our own). ~2 days work.
- **M1, B5** — the architecture and the architectural Acid Test. Phase 2, weeks 3-4.
- **Full training results from real data.** That's for you to run on your Mac.

---

## Files added

```
src/training.py                          (13.4 KB) — shared training infrastructure
src/models/__init__.py                   (empty)   — package marker
src/models/gnn_baseline.py               (6.5 KB)  — B3 GNN model
scripts/train_b3_gnn.py                  (9.0 KB)  — B3 training driver
PHASE_1_B3_STATUS.md                              — this memo
```

Total: ~29 KB of new code. The training infrastructure (~13 KB of `training.py`) will be reused for B2, B4, M1, and B5.

---

## How to commit

```bash
cd ~/h3_oscillator
# After running and verifying sanity check passes:
git add src/training.py src/models/__init__.py src/models/gnn_baseline.py
git add scripts/train_b3_gnn.py PHASE_1_B3_STATUS.md
git commit -m "Phase 1 / B3: GNN baseline + shared training infrastructure

- HexGNN: GraphSAGE-mean style, 6,402 params at default config
- Training loop in src/training.py, will be reused by B2, B4, M1, B5
- Sanity check passes on smoke dataset
- On medium γ-dataset, B3 beats persistence by 2× on in-dist and OOD-spatial,
  fails on OOD-parameter (different regime) — expected for non-equivariant baseline"
git push
```

---

## Calibration note

This is one baseline of five total models in the comparison set. The headline architectural claim from doc 05 §14 — that block-circulant equivariance is the source of any advantage — can't be tested until M1 and B5 exist. B3's role here is to establish "what a flexible graph-structured baseline achieves at matched parameter count." That's important context, but it's not the experiment yet.

The pattern to watch for as more baselines come online: each one will be strong at some surface and weak at others. The H3-Oscillator's signature, if it works, would be **smaller variance across surfaces** — better OOD-parameter generalization without giving up in-distribution accuracy.

Stay disciplined. Don't read tea leaves from B3's numbers in isolation.
