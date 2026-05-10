"""
Phase 0 sanity check: generate Gray-Scott trajectories on H3 and visualize them.

This script verifies that the data generator produces actual Gray-Scott patterns
(spots, mazes) and not just noise or trivial dynamics. If the visualizations look
right, we have confidence to move to dataset construction (Phase 0 milestone).

Outputs:
    figures/gs_alpha_spots.png — final state under alpha (spots) regime
    figures/gs_gamma_mazes.png — final state under gamma (mazes) regime
    figures/gs_alpha_panels.png — multiple frames of alpha trajectory
    figures/gs_gamma_panels.png — multiple frames of gamma trajectory

Usage:
    python -m scripts.phase0_sanity_check
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.h3_region import H3Region
from src.gray_scott_h3 import (
    GrayScottParams, integrate_trajectory, GS_REGIMES,
)
from src.visualization import plot_field, plot_trajectory_panels


def main():
    # Output directory for figures
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Set up region (matches doc 05 spec: 817 cells, no pentagons)
    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    print(f"Region: {region}")
    print(f"Pentagons: {region.has_pentagons}")
    print()

    # Long trajectory to allow patterns to fully develop.
    # Doc 05 spec: T_burn = 1000 timesteps spin-up + 256 timesteps to record.
    # We'll do 5000 total to be sure patterns are mature.
    n_steps = 5000
    record_every = 250  # gives 21 frames

    # Per-regime initialization config:
    #   alpha (spots) - many scattered seeds because spots don't propagate
    #   gamma (mazes) - one seed is fine because maze fronts propagate
    init_configs = {
        "alpha": {"n_seeds": 12, "perturbation_fraction": 0.10},
        "gamma": {"n_seeds": 1,  "perturbation_fraction": 0.05},
    }

    for regime in ["alpha", "gamma"]:
        print(f"=== Regime: {regime} ({GS_REGIMES[regime]['description']}) ===")
        params = GrayScottParams.from_regime(regime)
        cfg = init_configs[regime]
        print(f"Parameters: F={params.F}, k={params.k}; init: n_seeds={cfg['n_seeds']}, "
              f"perturb_frac={cfg['perturbation_fraction']}")

        traj = integrate_trajectory(
            region=region, params=params, n_steps=n_steps,
            seed=42, record_every=record_every, verbose=False,
            **cfg,
        )
        print(f"Trajectory shape: {traj.shape}")

        # Statistics on final frame
        u_final = traj[-1, :, 0]
        v_final = traj[-1, :, 1]
        print(f"Final u: mean={u_final.mean():.3f}, std={u_final.std():.3f}, "
              f"range=[{u_final.min():.3f}, {u_final.max():.3f}]")
        print(f"Final v: mean={v_final.mean():.3f}, std={v_final.std():.3f}, "
              f"range=[{v_final.min():.3f}, {v_final.max():.3f}]")

        # Did pattern actually form? (Compare std to noise level ~0.01)
        if v_final.std() > 0.05:
            print(f"  ✓ Pattern formed (v std = {v_final.std():.3f} >> noise 0.01)")
        else:
            print(f"  ✗ WARNING: no clear pattern (v std = {v_final.std():.3f})")

        # Save final state visualization
        fig, ax = plot_field(
            v_final, region, cmap="viridis", title=f"Gray-Scott {regime} ({GS_REGIMES[regime]['description']}), v field at step {n_steps}"
        )
        out_path = fig_dir / f"gs_{regime}_final.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        # Save trajectory panels (showing pattern emergence over time)
        fig = plot_trajectory_panels(
            traj, region, field_idx=1, n_panels=6, cmap="viridis",
            title_prefix=f"{regime} step",
        )
        fig.suptitle(f"Gray-Scott {regime}: pattern emergence (v field)", fontsize=14)
        out_path = fig_dir / f"gs_{regime}_panels.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
        print()

    print("=" * 60)
    print("Phase 0 sanity check complete.")
    print("Verify the figures show real Gray-Scott patterns:")
    print("  - alpha: stable spots scattered across region")
    print("  - gamma: maze-like winding patterns")
    print("If patterns look correct -> proceed to Phase 0 dataset construction")


if __name__ == "__main__":
    main()
