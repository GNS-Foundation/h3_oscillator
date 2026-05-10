"""
Phase 0 verification of baseline B1: hand-crafted finite-difference solver.

B1 is the experiment's upper-bound: a model with perfect knowledge of the
dynamics. This test confirms B1 reconstructs trajectory continuation with
zero error when given the true intermediate state.

This sets up the test format used throughout the experiment:
  - Generate ground-truth trajectory of length T+H
  - Take frames [0, T] as "input"
  - Take frames [T+1, T+H] as "ground truth target"
  - Predict frames [T+1, T+H] from the state at frame T
  - Compute RMSE between prediction and ground truth

For B1 specifically, the prediction error should be ~0 (within float precision)
because B1 IS the dynamics that generated the trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.h3_region import H3Region
from src.gray_scott_h3 import (
    GrayScottParams, integrate_trajectory, predict_trajectory_handcrafted,
)


def main():
    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    print(f"Region: {region}\n")

    # Generate full trajectory: T_input + H_predict frames
    T_input = 16  # input frames
    H_predict = 8  # prediction horizon (matches OOD-temporal in doc 05)
    n_total_frames = T_input + H_predict

    # We sample every 8 generator-steps as per doc 05 §2.6
    record_every = 8
    n_steps = (n_total_frames - 1) * record_every

    # Spin-up first to get past the initial transient
    burn_steps = 1000

    for regime in ["alpha", "gamma"]:
        print(f"=== Regime: {regime} ===")
        params = GrayScottParams.from_regime(regime)
        n_seeds = 12 if regime == "alpha" else 1

        # Generate ground truth: spin-up + recorded frames
        full_n_steps = burn_steps + n_steps
        full_traj = integrate_trajectory(
            region=region, params=params,
            n_steps=full_n_steps,
            seed=42, record_every=record_every,
            n_seeds=n_seeds,
            verbose=False,
        )
        # Take frames after spin-up
        burn_frames = burn_steps // record_every
        traj = full_traj[burn_frames:burn_frames + n_total_frames]
        print(f"Ground-truth trajectory: shape {traj.shape}, "
              f"frames after spin-up = {len(traj)}")

        # Now predict frames T_input...T_input+H_predict-1 from frame T_input-1
        # (B1 takes the state at frame T_input-1 and integrates forward)
        u_init = traj[T_input - 1, :, 0]
        v_init = traj[T_input - 1, :, 1]

        # Number of generator steps to span the H_predict frames
        predict_steps = H_predict * record_every
        prediction_frames_full = predict_trajectory_handcrafted(
            region=region, params=params,
            u_init=u_init, v_init=v_init,
            n_steps=predict_steps,
        )
        # Sample at the same record_every interval
        # prediction_frames_full has shape (predict_steps, n_cells, 2)
        # We want the H_predict frames at multiples of record_every
        frame_indices = np.arange(record_every, predict_steps + 1, record_every) - 1
        prediction = prediction_frames_full[frame_indices]

        ground_truth = traj[T_input:T_input + H_predict]
        print(f"Prediction shape: {prediction.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")

        # Compute RMSE
        diff = prediction - ground_truth
        rmse = np.sqrt(np.mean(diff ** 2))
        rmse_per_frame = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))

        print(f"B1 prediction RMSE (overall): {rmse:.2e}")
        print(f"B1 prediction RMSE per frame: "
              f"{[f'{x:.2e}' for x in rmse_per_frame]}")

        if rmse < 1e-10:
            print(f"  ✓ B1 reconstructs trajectory exactly (RMSE within float precision)")
        else:
            print(f"  ✗ B1 has unexpected error — investigate")
        print()

    print("=" * 60)
    print("Phase 0 B1 verification complete.")
    print("B1 = hand-crafted finite-difference solver = experiment upper bound.")
    print("Learned models (B2, B3, B4, M1, B5) must approach B1's accuracy.")


if __name__ == "__main__":
    main()
