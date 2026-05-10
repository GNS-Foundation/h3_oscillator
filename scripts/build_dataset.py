"""
Dataset construction for the H3-Oscillator experiment.

Generates the synthetic Gray-Scott trajectory datasets per doc 05 §4:
  - Train/val/test splits for regimes alpha (spots) and gamma (mazes)
  - Held-out region for OOD-spatial test
  - Delta regime test set for OOD-parameter novel-regime test

Output format: each split is saved as a single .npz file containing:
  - trajectories: float32, shape (N, T_frames, n_cells, 2)
  - seeds:        int64,   shape (N,) — RNG seed used for each trajectory
  - region_info:  dict with center_lat, center_lon, resolution, k_ring, n_cells
  - regime:       str
  - params:       dict with F, k, D_u, D_v, dt
  - generator_metadata: dict with frame spacing, spin-up steps, format version

Modes:
  - smoke:  10 train, 5 val, 5 test per regime — for pipeline verification
  - medium: 100 train, 20 val, 20 test per regime — for quick training experiments
  - full:   1000 train, 100 val, 100 test per regime — full v0.1 spec from doc 05

Estimated wall time on Apple Silicon CPU (per trajectory: ~1 second):
  - smoke:  ~1 min  total
  - medium: ~10 min total
  - full:   ~50 min total

Usage:
  python -m scripts.build_dataset --mode smoke
  python -m scripts.build_dataset --mode full --output-dir data/
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.h3_region import H3Region
from src.gray_scott_h3 import (
    GrayScottParams, integrate_trajectory, GS_REGIMES,
)


# Per doc 05 §2.6: spin-up + sample every record_every steps × n_frames
SPIN_UP_STEPS = 1000
RECORD_EVERY = 8
N_FRAMES = 32  # input (16) + future (16) frames per trajectory

# Per-regime initialization configuration (set during sanity check)
INIT_CONFIGS = {
    "alpha": {"n_seeds": 12, "perturbation_fraction": 0.10},
    "beta":  {"n_seeds": 6,  "perturbation_fraction": 0.08},
    "gamma": {"n_seeds": 1,  "perturbation_fraction": 0.05},
    "delta": {"n_seeds": 4,  "perturbation_fraction": 0.06},
}

# Mode-dependent split sizes. Maintain ratios consistent with doc 05.
SPLIT_SIZES = {
    "smoke":  {"train": 10,   "val": 5,   "test": 5,    "held_out": 5,   "delta": 5},
    "medium": {"train": 100,  "val": 20,  "test": 20,   "held_out": 20,  "delta": 20},
    "full":   {"train": 1000, "val": 100, "test": 100,  "held_out": 100, "delta": 100},
}

# Regions: main + held-out for OOD-spatial test
MAIN_REGION = {"center_lat": 45.0, "center_lon": 0.0, "resolution": 5, "k_ring": 16}
HELD_OUT_REGION = {"center_lat": 40.0, "center_lon": -90.0, "resolution": 5, "k_ring": 16}

# Format version: bump if schema changes incompatibly
DATA_FORMAT_VERSION = "0.1"


def generate_trajectory_set(
    region: H3Region,
    regime: str,
    n_trajectories: int,
    seed_offset: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n_trajectories` Gray-Scott trajectories on the given region.

    Each trajectory uses a unique seed (seed_offset + i for i in 0..N).
    Returns the trajectory array and the seed array.

    The trajectory has shape (N_FRAMES, n_cells, 2). It contains frames
    sampled every RECORD_EVERY generator steps after SPIN_UP_STEPS.

    Returns
    -------
    trajectories : np.ndarray, float32, shape (n_trajectories, N_FRAMES, n_cells, 2)
    seeds : np.ndarray, int64, shape (n_trajectories,)
    """
    if regime not in GS_REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    params = GrayScottParams.from_regime(regime)
    init_cfg = INIT_CONFIGS[regime]

    n_steps_total = SPIN_UP_STEPS + (N_FRAMES - 1) * RECORD_EVERY
    burn_frames = SPIN_UP_STEPS // RECORD_EVERY

    trajectories = np.zeros((n_trajectories, N_FRAMES, region.n_cells, 2), dtype=np.float32)
    seeds = np.zeros(n_trajectories, dtype=np.int64)

    t_start = time.time()
    for i in range(n_trajectories):
        traj_seed = seed_offset + i
        # integrate_trajectory returns (full_n_frames, n_cells, 2)
        # full_n_frames = n_steps_total // RECORD_EVERY + 1
        full_traj = integrate_trajectory(
            region=region,
            params=params,
            n_steps=n_steps_total,
            seed=int(traj_seed),
            record_every=RECORD_EVERY,
            n_seeds=init_cfg["n_seeds"],
            perturbation_fraction=init_cfg["perturbation_fraction"],
        )
        # Take frames after spin-up, exactly N_FRAMES of them.
        # full_traj[burn_frames:burn_frames + N_FRAMES] is the recorded window.
        trajectories[i] = full_traj[burn_frames:burn_frames + N_FRAMES].astype(np.float32)
        seeds[i] = traj_seed

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_trajectories - (i + 1)) / rate
            print(f"  [{i+1}/{n_trajectories}] {rate:.1f} traj/sec, ETA {eta:.1f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"  Generated {n_trajectories} trajectories in {elapsed:.1f}s "
              f"({n_trajectories/elapsed:.1f} traj/sec)")

    return trajectories, seeds


def save_split(
    output_path: Path,
    trajectories: np.ndarray,
    seeds: np.ndarray,
    region: H3Region,
    regime: str,
):
    """Save a dataset split to .npz file with all metadata."""
    params = GrayScottParams.from_regime(regime)

    # Build metadata. Use np.array to keep it inside the .npz (Python objects
    # are not directly serializable in npz, so we'll store as JSON string).
    region_info = {
        "center_lat": float(region.center_lat),
        "center_lon": float(region.center_lon),
        "resolution": int(region.resolution),
        "k_ring": int(region.k_ring),
        "n_cells": int(region.n_cells),
    }
    params_dict = asdict(params)
    generator_metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "spin_up_steps": SPIN_UP_STEPS,
        "record_every": RECORD_EVERY,
        "n_frames": N_FRAMES,
        "init_config": INIT_CONFIGS[regime],
    }

    np.savez_compressed(
        output_path,
        trajectories=trajectories,
        seeds=seeds,
        region_info_json=np.array(json.dumps(region_info)),
        params_json=np.array(json.dumps(params_dict)),
        regime=np.array(regime),
        generator_metadata_json=np.array(json.dumps(generator_metadata)),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path.name}: {trajectories.shape}, {size_mb:.1f} MB")


def build_dataset(
    mode: str,
    output_dir: Path,
    seed_base: int = 1000,
    skip_existing: bool = True,
):
    """Build the full dataset for the H3-Oscillator experiment.

    Parameters
    ----------
    mode : str
        One of 'smoke', 'medium', 'full'. Controls split sizes.
    output_dir : Path
        Where to save .npz files.
    seed_base : int
        Base seed for trajectory RNG. Different seeds are used for different
        splits and regimes to ensure no leakage.
    skip_existing : bool
        If True, skip generating splits whose output file already exists.
        Allows resuming partial runs.
    """
    if mode not in SPLIT_SIZES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(SPLIT_SIZES)}")

    sizes = SPLIT_SIZES[mode]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up regions
    print("=" * 70)
    print(f"Building dataset (mode={mode}) -> {output_dir}")
    print("=" * 70)
    print(f"Main region: {MAIN_REGION}")
    main_region = H3Region(**MAIN_REGION)
    print(f"  -> {main_region}")

    print(f"Held-out region: {HELD_OUT_REGION}")
    held_out_region = H3Region(**HELD_OUT_REGION)
    print(f"  -> {held_out_region}")

    if main_region.n_cells != held_out_region.n_cells:
        print(f"  ⚠️  Note: regions have different cell counts "
              f"({main_region.n_cells} vs {held_out_region.n_cells}). "
              f"OOD-spatial models will need to handle variable region size.")
    print()

    # Generation plan: (split_name, region, regime, n, seed_offset, output_filename)
    # Seed offsets are spaced by 100000 to avoid collisions even at 'full' mode.
    plan = [
        # alpha regime, main region
        ("alpha train",       main_region,     "alpha", sizes["train"],    seed_base + 0,      "alpha_train.npz"),
        ("alpha val",         main_region,     "alpha", sizes["val"],      seed_base + 100000, "alpha_val.npz"),
        ("alpha test (ID)",   main_region,     "alpha", sizes["test"],     seed_base + 200000, "alpha_test.npz"),
        ("alpha test (OOD-spatial)", held_out_region, "alpha", sizes["held_out"], seed_base + 300000, "alpha_test_ood_spatial.npz"),

        # gamma regime, main region
        ("gamma train",       main_region,     "gamma", sizes["train"],    seed_base + 400000, "gamma_train.npz"),
        ("gamma val",         main_region,     "gamma", sizes["val"],      seed_base + 500000, "gamma_val.npz"),
        ("gamma test (ID)",   main_region,     "gamma", sizes["test"],     seed_base + 600000, "gamma_test.npz"),
        ("gamma test (OOD-spatial)", held_out_region, "gamma", sizes["held_out"], seed_base + 700000, "gamma_test_ood_spatial.npz"),

        # delta regime: only test set (used as novel-regime OOD-parameter test)
        ("delta test (OOD-parameter, novel)", main_region, "delta", sizes["delta"], seed_base + 800000, "delta_test.npz"),
    ]

    total_t_start = time.time()
    summary = []

    for split_name, region, regime, n, seed_offset, fname in plan:
        out_path = output_dir / fname
        print(f"--- {split_name} (regime={regime}, n={n}, region={region.center}) ---")

        if skip_existing and out_path.exists():
            print(f"  Skipping (file exists): {out_path}")
            # Still record in summary
            with np.load(out_path, allow_pickle=False) as data:
                shape = data["trajectories"].shape
            summary.append({
                "split": split_name, "regime": regime, "n": n,
                "shape": list(shape), "file": fname, "skipped": True
            })
            continue

        trajectories, seeds = generate_trajectory_set(
            region=region, regime=regime,
            n_trajectories=n, seed_offset=seed_offset,
            verbose=True,
        )
        save_split(out_path, trajectories, seeds, region, regime)
        summary.append({
            "split": split_name, "regime": regime, "n": n,
            "shape": list(trajectories.shape), "file": fname, "skipped": False
        })
        print()

    total_elapsed = time.time() - total_t_start

    # Save summary metadata
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "mode": mode,
        "split_sizes": sizes,
        "main_region": MAIN_REGION,
        "held_out_region": HELD_OUT_REGION,
        "main_region_n_cells": int(main_region.n_cells),
        "held_out_region_n_cells": int(held_out_region.n_cells),
        "splits": summary,
        "spin_up_steps": SPIN_UP_STEPS,
        "record_every": RECORD_EVERY,
        "n_frames": N_FRAMES,
        "init_configs": INIT_CONFIGS,
        "total_generation_time_seconds": float(total_elapsed),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print(f"Dataset build complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Wrote {len(plan)} split files + metadata.json to {output_dir}")
    print(f"Total disk usage: "
          f"{sum(p.stat().st_size for p in output_dir.iterdir())/1024/1024:.1f} MB")


def verify_dataset(output_dir: Path):
    """Verify the generated dataset by loading each split and checking integrity."""
    print("=" * 70)
    print("Verifying dataset integrity...")
    print("=" * 70)

    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  ✗ metadata.json missing")
        return False

    with open(metadata_path) as f:
        metadata = json.load(f)

    all_ok = True
    for split in metadata["splits"]:
        path = output_dir / split["file"]
        if not path.exists():
            print(f"  ✗ {split['file']} missing")
            all_ok = False
            continue

        data = np.load(path, allow_pickle=False)
        trajs = data["trajectories"]
        seeds = data["seeds"]

        # Sanity checks
        ok = True
        ok &= (trajs.shape == tuple(split["shape"]))
        ok &= (trajs.dtype == np.float32)
        ok &= (seeds.shape == (trajs.shape[0],))
        ok &= np.isfinite(trajs).all()
        ok &= (trajs.min() >= -0.1) and (trajs.max() <= 1.1)  # sanity range

        # Check pattern formed (final v field has nontrivial std)
        v_final = trajs[:, -1, :, 1]  # (N, n_cells)
        v_std = v_final.std(axis=1).mean()
        pattern_formed = v_std > 0.03  # well above noise

        marker = "✓" if (ok and pattern_formed) else "✗"
        print(f"  {marker} {split['file']}: shape={trajs.shape}, "
              f"v_final_std={v_std:.3f} {'(pattern formed)' if pattern_formed else '(NO PATTERN)'}")
        if not ok:
            print(f"      shape match: {trajs.shape == tuple(split['shape'])}, "
                  f"finite: {np.isfinite(trajs).all()}, "
                  f"range: [{trajs.min():.3f}, {trajs.max():.3f}]")
        all_ok &= ok and pattern_formed

    print()
    if all_ok:
        print("  ✓ All splits verified successfully")
    else:
        print("  ✗ Some splits failed verification")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["smoke", "medium", "full"], default="smoke",
                        help="Dataset size mode (default: smoke)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Output directory (default: data/)")
    parser.add_argument("--seed-base", type=int, default=1000,
                        help="Base seed for trajectory RNG (default: 1000)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-generate splits even if output file exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip generation, only verify existing dataset")
    args = parser.parse_args()

    if args.verify_only:
        ok = verify_dataset(args.output_dir)
        return 0 if ok else 1

    build_dataset(
        mode=args.mode,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
        skip_existing=not args.no_skip_existing,
    )

    print()
    verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
