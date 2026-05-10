# Phase 0 — Dataset Construction Status

**Date:** Phase 0 / Week 1 — dataset construction complete.
**Context:** Continuation of Phase 0 implementation after `PHASE_0_STATUS.md`. This memo documents the dataset-construction additions: the `build_dataset.py` script, the `data_loader.py` module, and verification of the OOD-spatial split.

---

## What's new in this session

### 1. `scripts/build_dataset.py` — main dataset construction script

Configurable script that generates the full 9-split dataset per doc 05 §4. Three modes:

| Mode | Train | Val | Test | Held-out | δ-test | Total trajectories | Wall time | Disk |
|---|---|---|---|---|---|---|---|---|
| `smoke` | 10 | 5 | 5 | 5 | 5 | 55 | ~11 sec | ~10 MB |
| `medium` | 100 | 20 | 20 | 20 | 20 | 340 | ~66 sec | ~59 MB |
| `full` | 1000 | 100 | 100 | 100 | 100 | 2700 | ~8 min | ~470 MB |

Wall-time benchmarks measured on Linux x86_64 sandbox (~5.5 traj/sec). Apple Silicon should be similar or faster.

Splits produced (per regime: α and γ, plus δ for OOD-parameter test):
- `<regime>_train.npz`: training trajectories
- `<regime>_val.npz`: validation
- `<regime>_test.npz`: in-distribution test
- `<regime>_test_ood_spatial.npz`: held-out region test (40°N, -90°E, central US)
- `delta_test.npz`: novel-regime OOD-parameter test (main region)

Plus `metadata.json` with full provenance: format version, split sizes, region info, generator config, generation timestamps.

**Each `.npz` file contains:**
- `trajectories`: float32, shape `(N, 32, 817, 2)` — the (u, v) field at 32 sampled frames
- `seeds`: int64, shape `(N,)` — RNG seed per trajectory (for reproducibility)
- `regime`, `region_info_json`, `params_json`, `generator_metadata_json` — full provenance

**Resumability:** if a `.npz` already exists, the script skips it. Allows partial runs to be completed without redoing finished splits. Use `--no-skip-existing` to force regeneration.

### 2. `src/data_loader.py` — clean dataset loading API

`load_split(path)` returns a `TrajectoryDataset` dataclass with:
- `trajectories`, `seeds`: the raw arrays
- `regime`: regime name
- `region`: reconstructed `H3Region` object (verified to match what was saved)
- `params`: Gray-Scott parameters
- `generator_metadata`: spin-up steps, frame spacing, etc.
- `split_input_target(input_frames, target_horizon)`: split for training / OOD-temporal eval

`load_all_splits(data_dir)` loads everything in a directory at once. Used by training code in Phase 1+.

### 3. `scripts/visualize_ood_spatial.py` — OOD-spatial verification

Side-by-side visualization of α and γ patterns from main region (45°N, 0°E, central France) vs held-out region (40°N, -90°E, central US). Confirms:
- Same regime → same pattern type (spots/mazes)
- Different geographic location → different specific realizations
- Different H3 cell IDs (verified by sampling `region.cells[:3]` from each)
- Same cell count (both regions: 817 cells exactly)

See `figures/ood_spatial_comparison.png`.

### 4. New region: held-out at (40°N, -90°W)

Properties:
- 817 cells (matches main region exactly — no architectural changes needed)
- 0 pentagons (verified)
- Fully disjoint from main region (different cell IDs, different geographic area)

---

## Verified

- `smoke` mode: 11 seconds total, 9 splits, all patterns formed, all data integrity checks pass
- `medium` mode: 66 seconds total, 9 splits, all patterns formed, all data integrity checks pass
- Data loader round-trips correctly: saved data == loaded data
- Region reconstruction is deterministic: H3Region rebuilt from metadata has same cell ordering and adjacency as original
- OOD-spatial visualization: confirms different cell IDs, same dynamics

---

## What you need to do on your Mac

### Run full-mode dataset generation

```bash
cd ~/h3_oscillator
source .venv/bin/activate
python -m scripts.build_dataset --mode full --output-dir data/full
```

Expected output:
- 2700 trajectories generated
- ~5-10 minutes wall time on Apple Silicon
- ~470 MB written to `data/full/` (gitignored)
- `metadata.json` with full provenance

The script prints a progress line every 50 trajectories and ETA for each split. If it crashes mid-way, just re-run — it skips completed splits.

### Verify the full dataset

```bash
python -m scripts.build_dataset --verify-only --output-dir data/full
```

This loads each split and checks:
- File integrity (no corruption)
- Shape matches expected dimensions
- All values finite, in physical range [-0.1, 1.1]
- v-field std > 0.03 (pattern actually formed, not just noise)

### Optional: visualize OOD-spatial split

```bash
python -m scripts.visualize_ood_spatial
```

Generates `figures/ood_spatial_comparison.png`. Useful for confirming visually that the held-out region looks distinct from main region.

---

## Decisions made in this session

1. **Held-out region location:** chose (40°N, -90°W, central US) to be far from main region (central France) and avoid pentagons. Both regions happen to have exactly 817 cells, which simplifies architecture (no padding logic needed for variable cell counts).

2. **δ regime added** as the OOD-parameter "novel regime" test. Doc 05 §4.2 mentioned this in passing; I included it as a generated split. The δ regime (solitons, F=0.018, k=0.051) produces self-replicating patterns, qualitatively different from α (spots) and γ (mazes). This gives us the strongest possible OOD-parameter test: a regime no model sees during training, where the dynamics produce a fundamentally different pattern type.

3. **Seed-base spacing of 100,000** between splits ensures no seed collisions even at full mode (largest split is 1000 trajectories).

4. **`np.savez_compressed`** rather than `.npz` (uncompressed). Saves ~3× disk space at minor decompression cost. Files load in <1 sec each.

5. **JSON-encoded metadata in npz files** rather than separate metadata files. Each split is fully self-describing — you can `load_split(path)` without needing a separate metadata index. Slight redundancy (every split file repeats region info) but the simplicity is worth it.

---

## What's NOT in this session

- **Phase 1 baselines** (B2 transformer, B3 GNN, B4 hex CNN) — that's next session, ~1 week of work each per doc 05 §10.
- **Architecture M1** (the H3-Oscillator) — Phase 2, weeks 3-4.
- **Architectural Acid Test B5** (unconstrained LTC) — Phase 2 alongside M1.
- **Real training runs** — Phase 3, week 5.

The dataset is now infrastructure-complete. The next concrete coding work is implementing the baselines, which can begin once you've decided to proceed.

---

## Files added to commit

```
src/data_loader.py                        (5.9 KB) — clean .npz loading API
scripts/build_dataset.py                 (14.9 KB) — main dataset construction script
scripts/visualize_ood_spatial.py          (3.2 KB) — OOD-spatial verification figure
figures/ood_spatial_comparison.png       (735 KB) — verification figure
PHASE_0_DATASET_STATUS.md                          — this memo
```

Total: ~759 KB of new files, all under git (the actual datasets in `data/full/` are gitignored).

---

## How to commit

After running the dataset generator on your Mac and verifying:

```bash
cd ~/h3_oscillator
git add src/data_loader.py scripts/build_dataset.py scripts/visualize_ood_spatial.py
git add figures/ood_spatial_comparison.png PHASE_0_DATASET_STATUS.md
git commit -m "Phase 0 dataset construction: build_dataset.py + data_loader

- build_dataset.py: configurable smoke/medium/full modes
- 9 splits per dataset: train/val/test for alpha and gamma regimes,
  plus held-out region for OOD-spatial and delta regime for OOD-parameter
- Per-trajectory deterministic seeding for reproducibility
- data_loader.py: TrajectoryDataset dataclass with full metadata
- visualize_ood_spatial.py: side-by-side main vs held-out figure
- Verified: smoke (55 traj, 11s), medium (340 traj, 66s) both pass"

git push
```
