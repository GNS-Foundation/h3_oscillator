"""
Dataset loader for the Gray-Scott H3 trajectory datasets.

This module provides a clean API for loading the .npz splits produced by
`scripts/build_dataset.py`. It returns dataclasses with all metadata
(region, params, generator config) attached, so downstream training code
can verify it's loading the data it expects.

Usage:
    from src.data_loader import load_split
    data = load_split("data/full/alpha_train.npz")
    print(data.trajectories.shape)  # (1000, 32, 817, 2)
    print(data.regime)              # "alpha"
    print(data.region.center)       # H3 cell ID of main region center
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .h3_region import H3Region


@dataclass
class TrajectoryDataset:
    """A loaded trajectory dataset split."""
    trajectories: np.ndarray          # float32, (N, T_frames, n_cells, 2)
    seeds: np.ndarray                  # int64, (N,)
    regime: str                        # "alpha", "gamma", "delta", etc.
    region: H3Region                   # reconstructed H3 region object
    params: dict                       # F, k, D_u, D_v, dt
    generator_metadata: dict           # spin_up_steps, record_every, n_frames, etc.

    @property
    def n_trajectories(self) -> int:
        return self.trajectories.shape[0]

    @property
    def n_frames(self) -> int:
        return self.trajectories.shape[1]

    @property
    def n_cells(self) -> int:
        return self.trajectories.shape[2]

    def split_input_target(
        self,
        input_frames: int = 16,
        target_horizon: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split trajectories into input frames and target frames.

        Per doc 05 §4: input is frames [0..input_frames-1], target is the next
        `target_horizon` frames.

        For OOD-temporal evaluation: pass target_horizon=8 (longer than trained on).

        Returns
        -------
        inputs : np.ndarray, shape (N, input_frames, n_cells, 2)
        targets : np.ndarray, shape (N, target_horizon, n_cells, 2)
        """
        if input_frames + target_horizon > self.n_frames:
            raise ValueError(
                f"input_frames ({input_frames}) + target_horizon ({target_horizon}) "
                f"= {input_frames + target_horizon} > n_frames ({self.n_frames})"
            )
        inputs = self.trajectories[:, :input_frames]
        targets = self.trajectories[:, input_frames:input_frames + target_horizon]
        return inputs, targets

    def __repr__(self) -> str:
        return (
            f"TrajectoryDataset(regime={self.regime!r}, "
            f"n_trajectories={self.n_trajectories}, n_frames={self.n_frames}, "
            f"n_cells={self.n_cells}, region_center=({self.region.center_lat:.1f}, "
            f"{self.region.center_lon:.1f}))"
        )


def load_split(path: str | Path) -> TrajectoryDataset:
    """Load a single trajectory dataset split from a .npz file.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file produced by build_dataset.py.

    Returns
    -------
    dataset : TrajectoryDataset
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    data = np.load(path, allow_pickle=False)

    trajectories = data["trajectories"]
    seeds = data["seeds"]
    regime = str(data["regime"].item())
    region_info = json.loads(str(data["region_info_json"].item()))
    params = json.loads(str(data["params_json"].item()))
    generator_metadata = json.loads(str(data["generator_metadata_json"].item()))

    region = H3Region(
        center_lat=region_info["center_lat"],
        center_lon=region_info["center_lon"],
        resolution=region_info["resolution"],
        k_ring=region_info["k_ring"],
    )
    # Sanity check: cell count matches what was saved
    if region.n_cells != region_info["n_cells"]:
        raise RuntimeError(
            f"Region reconstruction failed: saved n_cells={region_info['n_cells']}, "
            f"reconstructed n_cells={region.n_cells}. h3 library version mismatch?"
        )

    return TrajectoryDataset(
        trajectories=trajectories,
        seeds=seeds,
        regime=regime,
        region=region,
        params=params,
        generator_metadata=generator_metadata,
    )


def load_all_splits(data_dir: str | Path) -> dict[str, TrajectoryDataset]:
    """Load all dataset splits from a directory.

    Returns a dict mapping split-name (e.g., 'alpha_train') to TrajectoryDataset.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    splits = {}
    for npz_path in sorted(data_dir.glob("*.npz")):
        split_name = npz_path.stem  # filename without .npz
        splits[split_name] = load_split(npz_path)
    return splits


if __name__ == "__main__":
    # Smoke test: load all splits from data/smoke
    import sys
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("data/smoke")

    if not data_dir.exists():
        print(f"Run `python -m scripts.build_dataset --mode smoke` first to create {data_dir}")
        sys.exit(1)

    print(f"Loading all splits from {data_dir}")
    print("=" * 60)
    splits = load_all_splits(data_dir)
    for name, ds in sorted(splits.items()):
        print(f"  {name}: {ds}")
        # Test split_input_target
        inputs, targets = ds.split_input_target(input_frames=16, target_horizon=3)
        print(f"    -> input: {inputs.shape}, target (H=3): {targets.shape}")
        if ds.n_frames >= 24:
            inputs8, targets8 = ds.split_input_target(input_frames=16, target_horizon=8)
            print(f"    -> input: {inputs8.shape}, target (H=8 OOD-temporal): {targets8.shape}")
