"""
Visualize OOD-spatial split: compare patterns from main region vs held-out region.

Same Gray-Scott dynamics, same regime — but different geographic location and
different underlying H3 cell IDs. This is the test set used to evaluate spatial
generalization (doc 05 §4.2 OOD-spatial).

Outputs:
    figures/ood_spatial_comparison.png — side-by-side main vs held-out for both regimes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_split
from src.visualization import plot_field


def main():
    data_dir = Path("data/smoke")
    if not data_dir.exists():
        print(f"Run `python -m scripts.build_dataset --mode smoke` first.")
        return 1

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Load main and OOD-spatial splits for both regimes
    splits = {
        "alpha_main":     load_split(data_dir / "alpha_test.npz"),
        "alpha_held_out": load_split(data_dir / "alpha_test_ood_spatial.npz"),
        "gamma_main":     load_split(data_dir / "gamma_test.npz"),
        "gamma_held_out": load_split(data_dir / "gamma_test_ood_spatial.npz"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    titles = {
        ("alpha", "main"):     "α (spots) — main region (45.0°N, 0.0°E)",
        ("alpha", "held_out"): "α (spots) — held-out region (40.0°N, -90.0°E)",
        ("gamma", "main"):     "γ (mazes) — main region (45.0°N, 0.0°E)",
        ("gamma", "held_out"): "γ (mazes) — held-out region (40.0°N, -90.0°E)",
    }

    # Use frame 16 (end of input window) of trajectory 0 for each split
    frame_idx = 16
    traj_idx = 0

    # Set per-regime color scales (same scale for main vs held-out, comparable)
    for row, regime in enumerate(["alpha", "gamma"]):
        main_v = splits[f"{regime}_main"].trajectories[traj_idx, frame_idx, :, 1]
        held_v = splits[f"{regime}_held_out"].trajectories[traj_idx, frame_idx, :, 1]
        vmin = min(main_v.min(), held_v.min())
        vmax = max(main_v.max(), held_v.max())

        for col, sub in enumerate(["main", "held_out"]):
            ds = splits[f"{regime}_{sub}"]
            v = ds.trajectories[traj_idx, frame_idx, :, 1]
            plot_field(
                v, ds.region, ax=axes[row, col],
                cmap="viridis", vmin=vmin, vmax=vmax,
                title=titles[(regime, sub)],
                show_colorbar=True,
            )

    fig.suptitle(
        f"OOD-spatial split: same dynamics, different geographic regions (frame {frame_idx} of trajectory {traj_idx})",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = fig_dir / "ood_spatial_comparison.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also print region cell-ID samples to show they're genuinely different
    print()
    print("Region cell-ID samples (confirming OOD-spatial uses different cells):")
    for split_name in ["alpha_main", "alpha_held_out"]:
        ds = splits[split_name]
        sample_cells = ds.region.cells[:3]
        print(f"  {split_name}: first 3 cells = {list(sample_cells)}")


if __name__ == "__main__":
    main()
