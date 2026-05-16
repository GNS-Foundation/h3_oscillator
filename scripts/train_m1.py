"""
Train M1 (H3Oscillator): full architecture with LTC dynamics layer.

Same training shape as M1Static — single-step prediction loss, autoregressive
rollout for evaluation. Architectural difference: between the equivariant
encoder and decoder, a learned LTC dynamics layer with K closed-form CfC
iterations (shared weights). The recurrent state-to-state operator is the
block-circulant R2R that B5 will ablate.

Usage:
    python -m scripts.train_m1 --sanity-check
    python -m scripts.train_m1 --regime gamma --dataset full --seeds 0
    python -m scripts.train_m1 --regime alpha --dataset full --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.data_loader import load_split
from src.models.h3_oscillator import H3Oscillator
from src.training import (
    RegionTensors, TrainConfig,
    train_model, evaluate_rollout,
)


def run_one_seed(
    regime: str,
    dataset_dir: Path,
    output_dir: Path,
    config: TrainConfig,
    seed: int,
    F_dim: int,
    K: int,
    device: torch.device,
):
    """Train one H3Oscillator with the given seed."""
    print(f"\n{'='*70}")
    print(f"M1 (H3Oscillator) training: regime={regime}, seed={seed}, "
          f"F_dim={F_dim}, K={K}")
    print(f"  dataset:  {dataset_dir}")
    print(f"  output:   {output_dir}")
    print(f"  device:   {device}")
    print(f"{'='*70}")

    # Load data
    train_ds = load_split(dataset_dir / f"{regime}_train.npz")
    val_ds = load_split(dataset_dir / f"{regime}_val.npz")
    test_id = load_split(dataset_dir / f"{regime}_test.npz")
    test_spatial = load_split(dataset_dir / f"{regime}_test_ood_spatial.npz")
    other_regime = "gamma" if regime == "alpha" else "alpha"
    test_other = load_split(dataset_dir / f"{other_regime}_test.npz")
    test_delta = load_split(dataset_dir / "delta_test.npz")

    main_region = train_ds.region
    held_out_region = test_spatial.region
    print(f"  main region:     {main_region}")
    print(f"  held-out region: {held_out_region}")
    print(f"  train pairs:     {(train_ds.n_frames - 1) * train_ds.n_trajectories}")

    main_tensors = RegionTensors.from_region(main_region, device)
    held_out_tensors = RegionTensors.from_region(held_out_region, device)

    # Model
    model = H3Oscillator(
        n_features=2, F_dim=F_dim, K=K, residual_prediction=True,
    ).to(device)
    print(f"  model parameters: {model.n_parameters()}  "
          f"(B3=6402, B4=8562, M1Static=8322 for comparison)")

    # Train
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_result = train_model(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds,
        region_tensors=main_tensors,
        config=config, device=device,
        output_dir=seed_dir, seed=seed, verbose=True,
    )
    print(f"  best val RMSE: {train_result['best_val_rmse']:.5f} "
          f"after {train_result['n_epochs_run']} epochs "
          f"(wall time: {train_result['wall_time_seconds']:.0f}s)")

    # Multi-surface evaluation
    print(f"\n  evaluating on all four surfaces (doc 05 §4.2):")
    eval_results = {}
    eval_results["in_distribution"] = evaluate_rollout(
        model, test_id, main_tensors, horizon=3, device=device,
    )
    eval_results["ood_temporal"] = evaluate_rollout(
        model, test_id, main_tensors, horizon=8, device=device,
    )
    eval_results["ood_spatial"] = evaluate_rollout(
        model, test_spatial, held_out_tensors, horizon=3, device=device,
    )
    eval_results["ood_parameter_other"] = evaluate_rollout(
        model, test_other, main_tensors, horizon=3, device=device,
    )
    eval_results["ood_parameter_delta"] = evaluate_rollout(
        model, test_delta, main_tensors, horizon=3, device=device,
    )

    # Print summary
    print(f"\n  surface          | rmse     | persist  | ratio  | n_examples")
    print(f"  ---------------- | -------- | -------- | ------ | ----------")
    for surface_name, result in eval_results.items():
        ratio = result["rmse"] / result["persistence_rmse"] if result["persistence_rmse"] > 0 else float("inf")
        print(f"  {surface_name:16} | {result['rmse']:.5f} | "
              f"{result['persistence_rmse']:.5f} | {ratio:.3f} | {result['n_examples']}")

    # Also report final dynamics scalars (stability + interpretability check)
    print(f"\n  dynamics scalars (final):")
    inv_tau = torch.exp(-model.dynamics.log_tau).detach().cpu()
    A_final = model.dynamics.A.detach().cpu()
    print(f"    1/τ:  min={inv_tau.min():.3f}  max={inv_tau.max():.3f}  "
          f"mean={inv_tau.mean():.3f}  (init: 2.0)")
    print(f"    A:    min={A_final.min():.3f}  max={A_final.max():.3f}  "
          f"mean={A_final.mean():.3f}  (init: 0.0)")

    eval_summary = {
        "regime": regime,
        "seed": seed,
        "model_params": model.n_parameters(),
        "best_val_rmse": train_result["best_val_rmse"],
        "n_epochs_run": train_result["n_epochs_run"],
        "wall_time_seconds": train_result["wall_time_seconds"],
        "eval_results": eval_results,
        "config": asdict(config),
        "model_config": {"F_dim": F_dim, "K": K},
        "dynamics_scalars": {
            "inv_tau_min": float(inv_tau.min()),
            "inv_tau_max": float(inv_tau.max()),
            "inv_tau_mean": float(inv_tau.mean()),
            "A_min": float(A_final.min()),
            "A_max": float(A_final.max()),
            "A_mean": float(A_final.mean()),
        },
    }
    with open(seed_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\n  Saved: {seed_dir / 'eval_summary.json'}")

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["alpha", "gamma"], default="alpha")
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--F-dim", type=int, default=8,
                        help="Regular-feature multiplicity (default 8 for ~8.5K params)")
    parser.add_argument("--K", type=int, default=4,
                        help="Number of LTC integration steps per prediction (default 4)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default 1e-3 (B4 lesson: 1e-2 caused instability)")
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Quick test: smoke dataset, 5 epochs, 1 seed")
    args = parser.parse_args()

    if args.sanity_check:
        args.dataset = "smoke"
        args.seeds = [0]
        args.n_epochs = 5
        args.batch_size = 8

    dataset_dir = Path("data") / args.dataset
    if not dataset_dir.exists():
        print(f"Dataset directory {dataset_dir} not found.")
        print(f"Run: python -m scripts.build_dataset --mode {args.dataset}")
        return 1

    output_dir = Path(args.output) if args.output else Path("results") / f"m1_{args.regime}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available() and not args.sanity_check:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = TrainConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        gradient_clip=1.0,
        eval_horizon=3,
        early_stop_patience=10,
    )

    all_results = []
    for seed in args.seeds:
        result = run_one_seed(
            regime=args.regime,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            config=config,
            seed=seed,
            F_dim=args.F_dim,
            K=args.K,
            device=device,
        )
        all_results.append(result)

    if len(args.seeds) > 1:
        print(f"\n{'='*70}")
        print(f"Multi-seed aggregate (n_seeds={len(args.seeds)})")
        print(f"{'='*70}")
        import numpy as np
        surfaces = ["in_distribution", "ood_temporal", "ood_spatial",
                    "ood_parameter_other", "ood_parameter_delta"]
        print(f"  surface          | rmse mean ± std")
        for s in surfaces:
            rmses = [r["eval_results"][s]["rmse"] for r in all_results]
            print(f"  {s:16} | {np.mean(rmses):.5f} ± {np.std(rmses):.5f}")

    aggregate_path = output_dir / "all_seeds_summary.json"
    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregate: {aggregate_path}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
