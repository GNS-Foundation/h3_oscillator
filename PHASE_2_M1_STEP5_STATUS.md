# Phase 2 / M1 — Step 5 Status: B5 (architectural Acid Test)

**Status: ✅ IMPLEMENTATION COMPLETE — AWAITING TRAINING RESULTS**

## What was built

Three new files:

1. `src/models/unconstrained_conv.py` (145 lines) — `UnconstrainedRegularToRegular`, the non-equivariant 6×6-per-position drop-in replacement for `RegularToRegular`.
2. `src/models/h3_oscillator_b5.py` (242 lines) — `H3OscillatorDynamicsB5` and `H3OscillatorB5`, the full B5 model.
3. `scripts/train_b5.py` (241 lines) — train + 4-surface evaluation script.

## The ablation in one sentence

**B5 = M1 with the C6 block-circulant constraint removed from the dynamics layer's recurrent operator W_h.** Nothing else changes.

| Component | M1 | B5 |
|---|---|---|
| Encoder S2R + R2R | C6-equivariant | C6-equivariant (same) |
| Decoder R2R + R2S | C6-equivariant | C6-equivariant (same) |
| Dynamics `W_x` | C6-equivariant S2R | C6-equivariant S2R (same) |
| Dynamics `W_h` | **C6-equivariant R2R (42 params/pair)** | **Unconstrained R2R_U (252 params/pair)** |
| Per-channel A, log_τ | Same | Same |
| CfC integrator + K=4 | Same | Same |
| Training loop, optimizer | Same | Same |

## Parameter accounting

```
                          M1        B5      Δ
encoder_s2r              120       120     0
encoder_r2r            2,696     2,696     0
dynamics.W_h           2,696    16,136   +13,440   ← THE DIFFERENCE
dynamics.W_x             120       120     0
dynamics A + log_τ        16        16     0
decoder_r2r            2,696     2,696     0
decoder_r2s              114       114     0
────────────────────────────────────────────
TOTAL                  8,458    21,898   +13,440  (2.59× more)
```

**This asymmetry is deliberate.** Removing a constraint inherently adds free parameters — there's no way around it. The interpretation:

- **If B5 ≤ M1 despite 2.6× more capacity** → the C6 constraint is doing real work that capacity alone cannot replace. Strong positive result for the synergy claim.
- **If B5 ≈ M1** → the constraint is roughly neutral. The LTC dynamics is what helps; the C6 prior in the recurrent operator is redundant given equivariant encoder/decoder.
- **If B5 > M1** → harder to interpret. Could be capacity, could be that the constraint actually hurts.

## Numerical verification (sandbox)

**Step 1: confirm the standalone unconstrained operator is NOT equivariant.**

```
UnconstrainedRegularToRegular alone (synthetic 7-cell test):
seed 0:  error = 3.47   ✓ NOT EQUIVARIANT
seed 1:  error = 3.74   ✓ NOT EQUIVARIANT
seed 2:  error = 3.33   ✓ NOT EQUIVARIANT
seed 3:  error = 4.38   ✓ NOT EQUIVARIANT
seed 42: error = 4.15   ✓ NOT EQUIVARIANT
```

O(1) error confirms the constraint is genuinely removed.

**Step 2: confirm the full B5 model breaks end-to-end equivariance.**

```
H3OscillatorB5 (F=4, K=4, 5 seeds):
seed 0:  error = 3.49e-02  ✓ NOT EQUIVARIANT (ablation works)
seed 1:  error = 6.39e-03  ✓ NOT EQUIVARIANT
seed 2:  error = 8.10e-02  ✓ NOT EQUIVARIANT
seed 3:  error = 7.01e-02  ✓ NOT EQUIVARIANT
seed 42: error = 1.66e-02  ✓ NOT EQUIVARIANT
```

The error is smaller (~10⁻²) than the standalone operator (~1) because the equivariant encoder/decoder/W_x absorb some of the asymmetry. But it's still ~10⁸× larger than M1's machine-precision equivariance, confirming the ablation is genuine.

**Step 3: gradient flow smoke test.**

```
B5 (F=8, K=4, params=21,898), 30 Adam steps @ lr=1e-3, grad_clip=1.0:
step  0: loss=1.86  grad_norm=2.37
step 10: loss=1.14  grad_norm=0.93
step 20: loss=1.07  grad_norm=0.43
step 29: loss=1.05  grad_norm=0.22

log_tau drift: -0.693 → [-0.705, -0.684]   (essentially unchanged)
1/τ stays in [1.98, 2.03]                  (stable LTC regime)
A drift: 0.0 → [-0.016, 0.014]             (small)
```

Healthy: loss decreases monotonically, gradients well-conditioned, dynamics scalars stay near init — same pattern as M1.

## What you need to run on the Mac

Same shape as M1. ~5-6 hours total for 3 seeds × 2 regimes.

```bash
cd /Users/camiloayerbeposada/h3_oscillator

# 1) Sanity check (~30s)
python -m scripts.train_b5 --sanity-check

# 2) Full runs, both regimes, 3 seeds each (~5-6 hours)
mkdir -p logs
caffeinate -i bash -c '
  python -m scripts.train_b5 --regime gamma --dataset full --seeds 0 1 2 2>&1 \
    | tee logs/b5_gamma_seeds_0-2.log
  python -m scripts.train_b5 --regime alpha --dataset full --seeds 0 1 2 2>&1 \
    | tee logs/b5_alpha_seeds_0-2.log
'
```

Saves results to `results/b5_{regime}_full/seed_N/eval_summary.json`. The JSON includes a new `model_name: "B5"` field for clarity.

## The decision tree once results land

Once B5 finishes, we'll have a complete 4-architecture × 2-regime × 5-surface table. The interpretation depends on the M1-vs-B5 comparison:

**Scenario A: B5 substantially worse than M1 (e.g., cross-regime ratio +30% or more):**
- Result: equivariance in the dynamics is essential. The 2.6× capacity advantage doesn't compensate for the lost prior.
- Paper-quality claim: "We demonstrate that combining C6 gauge equivariance with closed-form Liquid Time-Constant dynamics yields measurable cross-regime generalization gains, where the equivariance constraint is shown to be necessary via ablation."

**Scenario B: B5 ≈ M1 (within seed noise):**
- Result: the C6 constraint in W_h is redundant given equivariant encoder/decoder. Dynamics is what helps; equivariance has already been "spent" upstream.
- Paper-quality claim: "We find that LTC dynamics improves cross-regime generalization but the recurrent-operator equivariance constraint is redundant; equivariance in the encoder/decoder suffices."

**Scenario C: B5 better than M1 (with 2.6× more params):**
- Result: hard to interpret. The constraint might be hurting at this scale, OR capacity matters more than priors.
- Then we'd run a param-matched ablation (B5 with F=5, giving ~8.6K params, see below) to disentangle.

## Calibration reminders

- Same single-seed/multi-seed discipline as Step 2 and Step 4.
- **B5's variance pattern matters as much as its mean.** M1 had remarkably low variance across seeds. If B5 variance is much higher, the LTC + equivariance combination stabilizes training in a way capacity alone doesn't.
- **The dynamics scalars** (`1/τ`, `A`) at end of training are again logged. If B5 drifts them far more than M1 does, that's a signal the unconstrained operator is being used to "hack around" the lack of structure, rather than learning interpretable physics.
- **Don't anchor on a single regime.** γ and α may give different B5 vs M1 stories, just like they did for M1 vs M1Static.

## Optional follow-up: param-matched B5

If Scenario C occurs (B5 wins), the cleanest follow-up is to run B5 with `F=5` instead of `F=8`. This gives:

```
B5 (F=5): encoder + 5×5×252 dynamics W_h + decoder ≈ 8,647 params
```

— almost identical to M1's 8,458, isolating the constraint effect from the capacity effect. Not needed yet, but a useful tool to have if results call for it. Just run:

```bash
python -m scripts.train_b5 --regime alpha --dataset full --seeds 0 1 2 --F-dim 5
```

## Files modified / created

- `src/models/unconstrained_conv.py` — new file, 145 lines
- `src/models/h3_oscillator_b5.py` — new file, 242 lines
- `scripts/train_b5.py` — new file, 241 lines
- `PHASE_2_M1_STEP5_STATUS.md` — this memo

No changes to existing files. Step 5 is a clean addition.

## What's next

- **You**: run the B5 training commands above. ~5-6 hours overnight.
- **Next session**: complete 4-architecture × 2-regime × 3-seed table + final analysis. Then we'll have everything we need to update doc 04 / 05 and write the findings memo (doc 06).
- **Likely shape of the final story**: depends entirely on B5 outcome. The narrative has three possible endings (Scenarios A/B/C above), each well-defined and reportable.
