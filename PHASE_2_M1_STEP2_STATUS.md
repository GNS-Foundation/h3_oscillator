# Phase 2 / M1 — Step 2 Status: M1Static (gauge-equivariant static encoder/decoder)

**Status: ✅ IMPLEMENTATION COMPLETE — AWAITING TRAINING RESULTS**

## What was built

Two new files:

1. `src/models/equivariant_static.py` (171 lines) — `M1Static` model class
2. `scripts/train_m1_static.py` (214 lines) — train + 4-surface evaluation script

`M1Static` is the M1 architecture **minus** the dynamics layer. Pure
gauge-equivariant encoder + body + decoder. Predicts next state in a single
forward pass, like B3/B4, so it's a clean apples-to-apples ablation: same
data, same training loop, same evaluation surfaces — only architectural
difference is that all kernels are C6-equivariant by construction.

## Architecture

```
x (batch, n_cells, n_features=2)             — scalar field
  ↓ ScalarToRegular(2 → F=8)
h (batch, n_cells, 8, 6)                      — regular features
  ↓ RegularToRegular(8 → 8) → GELU → +residual
  ↓ RegularToRegular(8 → 8) → GELU → +residual
  ↓ RegularToRegular(8 → 8) → GELU → +residual
  ↓ RegularToScalar(8 → 2)
delta (batch, n_cells, 2)                     — scalar field
  ↓ + x   (residual prediction: x_{t+1} = x_t + delta)
```

End-to-end equivariance is preserved by composition:
- Each S2R / R2R / R2S is C6-equivariant by construction (Step 1)
- GELU is element-wise on regular features → preserves cyclic shift
- Residual connections (sums of equivariant maps) are equivariant
- The full model is therefore C6-equivariant

**Verified numerically end-to-end** (built-in test on synthetic 7-cell region,
5 seeds, F=4, 3-layer): max equivariance error ~1e-6 (float32 noise floor).

## Parameter counts

```
F=4,  n_layers=2:   1,470 params
F=4,  n_layers=3:   2,146 params
F=6,  n_layers=2:   3,212 params
F=6,  n_layers=3:   4,730 params
F=8,  n_layers=2:   5,626 params
F=8,  n_layers=3:   8,322 params  ← DEFAULT (B3=6,402  B4=8,562)
F=12, n_layers=2:  12,470 params
F=12, n_layers=3:  18,530 params
```

Default config (F=8, 3 layers, 8,322 params) is essentially param-parity with
B4. Slightly larger than B3 but on the same order. Fair comparison.

## Smoke test (sandbox)

```
Model: M1Static F=8 n_layers=3, params=8322
Loss progression (30 Adam steps @ lr=1e-3 on synthetic data):
  step  0: 27.0029
  step 10:  2.9587
  step 20:  1.6851
  step 29:  1.3748
  final grad norm: 9.41e+00
```

Loss decreases monotonically. Gradients healthy (no NaN, no collapse).
Training pipeline is wired correctly end-to-end.

## What you need to run on the Mac

Three commands. ~45 minutes total wall time on MPS, similar to B3/B4.

```bash
cd /Users/camiloayerbeposada/h3_oscillator

# 1) Quick sanity (5 epochs, smoke dataset, ~30s on MPS)
python -m scripts.train_m1_static --sanity-check

# 2) Full γ-regime training (~20 min)
python -m scripts.train_m1_static --regime gamma --dataset full --seeds 0

# 3) Full α-regime training (~20 min)
python -m scripts.train_m1_static --regime alpha --dataset full --seeds 0
```

Defaults: F=8, n_layers=3, lr=1e-3 (heeds the B4 lesson — 1e-2 caused
instability), n_epochs=60, batch_size=16, eval_horizon=3.

The script saves results in `results/m1_static_{regime}_full/seed_0/eval_summary.json`.

## What to look for in the results

The four-surface evaluation table will be printed at the end of each run.
Compare against the established baselines:

| Surface          | B3 γ     | B4 γ     | M1Static γ?  | B3 α     | B4 α     | M1Static α?  |
|------------------|----------|----------|--------------|----------|----------|--------------|
| In-distribution  | **0.027**| 0.038    | ?            | **0.21** | 0.354    | ?            |
| OOD-temporal H=8 | **0.029**| 0.043    | ?            | **0.18** | 0.329    | ?            |
| OOD-spatial      | **0.028**| 0.038    | ?            | **0.21** | 0.332    | ?            |
| OOD-param other  | 22.07    | 22.39    | ?            | 2.93     | **2.59** | ?            |
| OOD-param δ      | **2.97** | 2.79     | ?            | 4.20     | **3.70** | ?            |

**The headline question for Step 2: does the equivariance constraint by
itself buy us anything on the cross-regime wall (~22 ratio for γ)?**

Three possible outcomes worth pre-registering, in the spirit of calibration:

- **Helpful**: M1Static beats B3/B4 on cross-regime OOD (other-regime). The
  equivariance constraint helps the model generalize to physics it hasn't
  seen — even without dynamics. Strong positive signal for the full M1.
- **Neutral**: M1Static matches B3/B4 within noise on all surfaces. The
  static encoder/decoder doesn't gain from the equivariance constraint — but
  doesn't lose either. Step 3 (LTC dynamics) gets a clean test.
- **Hurtful**: M1Static is meaningfully worse on in-distribution / spatial /
  temporal (where B3/B4 do well). The constraint costs capacity without
  payoff at this scale. Would suggest the C6 prior isn't matched to
  Gray-Scott's actual symmetries on H3.

A fourth possibility: M1Static crushes everything — but that's unlikely on
priors and would warrant double-checking the eval setup (always be skeptical
of suspiciously good results, same as the SMM Sudoku phase).

## Calibration reminders

- We have **one seed**. B3/B4 results are also single-seed. Multi-seed
  variance check is deferred to a future session unless results are very
  close to baselines.
- The early-stop patience is 10 epochs. M1Static may converge faster or
  slower than B4 — n_epochs=60 (vs B4's 30) is a higher cap, but won't
  meaningfully change runtime since early stop kicks in.
- Don't over-interpret a single regime. We need both γ and α to triangulate.
- The eval surfaces are noisy. A 0.001 RMSE difference is not significant
  given single-seed runs; we want at least 5–10% relative improvement to
  be confident in any direction.

## What's next

- **You**: run the three commands above on your Mac, share results.
- **Next session**: I'll analyze the M1Static numbers vs B3/B4, calibrate
  expectations for the full M1, and then proceed to Step 3 (LTC dynamics
  layer with block-circulant R2R inside).

## Files modified / created

- `src/models/equivariant_static.py` — new file, 171 lines
- `scripts/train_m1_static.py` — new file, 214 lines
- `PHASE_2_M1_STEP2_STATUS.md` — this memo

No changes to existing files. M1 Step 2 is a clean addition.
