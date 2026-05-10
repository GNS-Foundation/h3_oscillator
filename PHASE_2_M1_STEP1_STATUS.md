# Phase 2 / M1 — Step 1 Status: Equivariant Convolution Primitives

**Date:** Phase 2 / Week 1 — Step 1 of 5 complete.
**Context:** First M1 implementation step. Builds the fundamental layers that
all subsequent M1 components use. Equivariance is verified numerically — if
this step is wrong, every later piece inherits the bug invisibly.

---

## What's in this session

### `src/models/equivariant_conv.py` — three C6-equivariant layer types

| Layer | Maps | Free params per (in_ch, out_ch) | Verified |
|---|---|---|---|
| `ScalarToRegular` (S2R) | scalar → regular | **7** (1 center + 6 directional) | ✓ |
| `RegularToRegular` (R2R) | regular → regular | **42** (6 center circulant + 36 one directional) | ✓ |
| `RegularToScalar` (R2S) | regular → scalar | **7** (1 center + 6 directional) | ✓ |

Each layer takes `forward(x, region_tensors)` matching the API from B3/B4.
S2R outputs gain a trailing dim of size 6 (the 6 angular directions of the
regular feature). R2R preserves it. R2S removes it.

The equivariance is enforced by the **kernel parameterization itself**, not
by training or regularization. The free parameters are mapped to the actual
kernel via a fixed expansion (Cohen-style):

- **S2R kernel:** `K[k, d] = w[(k - d) mod 6 + 1]` for the 6 directional weights
  (with index 0 reserved for the center weight). The output direction `d` sees
  neighbor slot `k` weighted by `w[(k-d) mod 6]` — a cyclic shift of the same
  base 6-vector across the 6 output directions.

- **R2R kernel:** at each spatial position p (1 center + 6 directional), a
  6×6 matrix mapping input feature directions to output feature directions:
  - Center matrix: must be circulant by C6 commutation. 6 free scalar params.
  - Directional matrices: `K_p_k = P^k @ K_p_0 @ P^{-k}` where P is the cyclic
    shift permutation. So all 6 directional matrices are determined by one
    base matrix `K_p_0` (36 free params).

- **R2S kernel:** dual of S2R. Output is scalar — center contribution is the
  C6-trivial sum across input directions, weighted by 1 free param. Each
  directional position couples neighbor slot `k` with input direction `d_in`
  via the shifted index `(k - d_in) mod 6 + 1`.

### Numerical equivariance verification

The test setup constructs a 7-cell hex region (1 center + 6 ring) where
60° CCW rotation is a clean cyclic permutation of the 6 outer cells. For
each layer, we verify:

```
K(g · x) [cell 0] == (g · K(x)) [cell 0]
```

across 5 random seeds. Result:

```
--- Seed 0 ---
  S2R: max diff = 5.96e-08  ✓ EQUIVARIANT
  R2R: max diff = 1.19e-07  ✓ EQUIVARIANT
  R2S: max diff = 1.79e-07  ✓ EQUIVARIANT
[... same for seeds 1, 2, 3, 42 ...]
```

The maximum difference across all 15 (5 seeds × 3 layers) tests is **2e-7**,
which is float32 numerical precision. Equivariance holds **exactly** by
construction; the residual is purely the floating-point representation of
the same arithmetic done in two orders.

### Smoke test on the real 817-cell H3 region

Full M1 pipeline (S2R encoder + 3× R2R + R2S decoder, F=8):

```
Input shape: torch.Size([2, 817, 2])
After S2R encoder: torch.Size([2, 817, 8, 6])
After R2R encoder: torch.Size([2, 817, 8, 6])
After R2R dynamics: torch.Size([2, 817, 8, 6])
After R2R decoder: torch.Size([2, 817, 8, 6])
After R2S decoder: torch.Size([2, 817, 2])

Total parameters: 8322
```

For comparison: B3 = 6,402 params, B4 = 8,562 params. M1's parameter count
will land in this same ballpark — exactly what we want for fair comparison.

Note: on the real H3 region, equivariance is only **approximate** because
H3 cells lie on a sphere and the local hex orientation drifts slightly across
the chart. For our small (~5° × 5°) region this drift is negligible, but it's
worth flagging as a Phase-2b concern if we ever extend to multi-chart.

---

## Calibration moment

Initial run: **all three layers failed equivariance** (max diff > 0.5). My
first instinct was that the kernel construction was wrong. After inspection,
the bug was in the **test setup**, not the implementation:

- I was constructing a different adjacency tensor pre-rotation vs. post-rotation
  (intending to model "the cells permute to new positions").
- This is wrong. Adjacency is determined by physical positions, which don't
  move under rotation. **Cells stay put; field values rotate.**
- With fixed adjacency and only the input field permuted, equivariance holds
  immediately.

Lesson: I almost convinced myself there was a deep bug in the kernel
parameterization (which would have meant rewriting hours of work). Walking
through one concrete example by hand identified that the test setup was
double-rotating. **The implementation was correct from the start.**

This is a useful sanity check on my pattern-matching habits. The "this looks
broken, must be a deep bug" reaction can be wrong; sometimes the bug is in
the surrounding scaffolding, not the math itself. Same lesson as the B4 LR
issue — don't over-interpret an initial failure before checking the test
conditions.

---

## What's next: Step 2 — Gauge-equivariant encoder/decoder

Next session, build a **static** model using these primitives:

```
S2R(2→F) → R2R(F→F) → R2R(F→F) → R2R(F→F) → R2S(F→2)
```

This is essentially **B4 with C6 equivariance instead of free directional
kernels**. No dynamics yet. We'll train it on the dataset and compare to
B4 directly:

- Same input/output shape, same data, same training loop
- Same parameter range
- Difference: M1's encoder/decoder enforce C6 equivariance, B4's don't

This gives us a cleanly isolated test of "what does C6 equivariance buy
us on its own, before adding the dynamics layer?" Two possible outcomes:

1. **Static-equivariant matches or beats B4** → equivariance is helping
   even without dynamics; foundation for full M1 is solid.
2. **Static-equivariant underperforms B4** → equivariance constraint is
   restrictive enough to hurt at this scale; need to think about whether
   dynamics can recover the lost expressiveness.

Either outcome is informative.

---

## Files added in this step

```
src/models/equivariant_conv.py   (~21 KB) — S2R, R2R, R2S layers + verification test
PHASE_2_M1_STEP1_STATUS.md                — this memo
```

The verification test is in the `__main__` block of the module — run it
anytime with `python -m src.models.equivariant_conv` to re-verify. It takes
about 2 seconds.

---

## Steps remaining for full M1

| Step | What | Estimated effort |
|---|---|---|
| 1 ✓ | Equivariant conv primitives + verification | done |
| 2 | Static encoder/decoder + train | next session |
| 3 | LTC dynamics layer with block-circulant GConv | session 3 |
| 4 | Assemble full M1 + train + 4-surface eval | session 4 |
| 5 | Build B5 (unconstrained LTC) + comparison | session 5 |

---

## What you do on your Mac

### Pull the new file

```bash
cd ~/h3_oscillator
mv ~/Downloads/equivariant_conv.py src/models/
mv ~/Downloads/PHASE_2_M1_STEP1_STATUS.md .
```

### Run the equivariance verification yourself

```bash
python -m src.models.equivariant_conv
```

You should see 15 `✓ EQUIVARIANT` results (5 seeds × 3 layers) with max
differences around 1e-7. If anything fails, that's something to flag
immediately — it would indicate a numerical issue specific to your platform.

### Optional smoke test on the real region

```bash
python -c "
import torch
from src.h3_region import H3Region
from src.training import RegionTensors
from src.models.equivariant_conv import ScalarToRegular, RegularToRegular, RegularToScalar

region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
rt = RegionTensors.from_region(region, torch.device('cpu'))

F = 8
encoder_s2r = ScalarToRegular(2, F)
r2r1 = RegularToRegular(F, F)
r2r2 = RegularToRegular(F, F)
r2r3 = RegularToRegular(F, F)
decoder_r2s = RegularToScalar(F, 2)

x = torch.randn(1, region.n_cells, 2)
h = encoder_s2r(x, rt)
h = r2r1(h, rt)
h = r2r2(h, rt)
h = r2r3(h, rt)
y = decoder_r2s(h, rt)
print(f'Pipeline OK: {x.shape} -> {y.shape}')

total = sum(p.numel() for m in [encoder_s2r, r2r1, r2r2, r2r3, decoder_r2s] for p in m.parameters())
print(f'Total params: {total}')
"
```

Should print 8322 params and confirm input/output shapes match.

### Commit Step 1

```bash
cd ~/h3_oscillator
git add src/models/equivariant_conv.py PHASE_2_M1_STEP1_STATUS.md
git commit -m "Phase 2 / M1 Step 1: C6-equivariant conv primitives (S2R, R2R, R2S)

Three layer types with C6 equivariance enforced by kernel parameterization:
- ScalarToRegular: 7 free params per (in_ch, out_ch) pair
- RegularToRegular: 42 free params per pair (6 circulant + 36 one directional,
  rest by cyclic conjugation)
- RegularToScalar: 7 free params per pair

Numerical verification: 5 seeds x 3 layers, max diff < 2e-7 (float32
precision). Full pipeline smoke-tested on 817-cell H3 region.

Note: doc 04 originally claimed R2R had 7 free params per pair — that was
wrong (it's 42). Total M1 architecture with F=8 is ~8.3K params, in the
same ballpark as B3 (6.4K) and B4 (8.5K)."
git push
```

---

## Calibration

Step 1 was the highest-risk part of M1. The kernel-expansion math is
subtle, and it's easy to get the index conventions wrong in ways that
look right but break equivariance. Now that we have **numerical proof**
that the primitives are equivariant by construction, every subsequent
M1 piece can build on this foundation with confidence.

The remaining steps are conceptually simpler — they're mostly assembly:
wire these primitives into an encoder/decoder (Step 2), wrap them in
LTC dynamics (Step 3), and put it all together (Step 4). The hard part
is done.

But: **stay disciplined.** We don't yet know if M1 will outperform B3/B4
on cross-regime generalization. The equivariance constraint is verified
to be implemented correctly, but whether it produces actual generalization
benefit is an empirical question we can only answer in Step 4.
