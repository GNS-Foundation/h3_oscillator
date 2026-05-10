"""
Shared training and evaluation infrastructure for the H3-Oscillator experiment.

Used by all model training scripts (B2, B3, B4, M1, B5). Provides:
  - Trajectory-to-batch dataloaders (single-step prediction pairs)
  - Training loop with epoch-level validation and CSV logging
  - Multi-surface evaluation: in-distribution, OOD-temporal, OOD-spatial,
    OOD-parameter (per doc 05 §4.2)
  - Trivial baselines (persistence) for sanity-check floor
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data_loader import TrajectoryDataset, load_split
from .h3_region import H3Region


# -------------------------------------------------------------------------
# PyTorch dataset wrapper for single-step prediction pairs
# -------------------------------------------------------------------------

class SingleStepPairs(Dataset):
    """Yields (state at frame t, state at frame t+1) pairs from a trajectory dataset.

    Each trajectory contributes (n_frames - 1) pairs. Across N trajectories of
    32 frames each, this gives 31*N training examples. With N=1000 that's 31,000
    pairs per epoch — plenty of training signal.

    Returns:
        x : tensor (n_cells, n_features), state at frame t
        y : tensor (n_cells, n_features), state at frame t+1
    """

    def __init__(self, trajectories: np.ndarray):
        # trajectories: (N, T, n_cells, n_features)
        if trajectories.ndim != 4:
            raise ValueError(f"Expected 4D array (N, T, n_cells, n_features); got shape {trajectories.shape}")
        self.trajectories = torch.from_numpy(trajectories).float()
        self.N, self.T, self.n_cells, self.n_features = self.trajectories.shape
        self.pairs_per_traj = self.T - 1

    def __len__(self) -> int:
        return self.N * self.pairs_per_traj

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_idx = idx // self.pairs_per_traj
        frame_idx = idx % self.pairs_per_traj
        x = self.trajectories[traj_idx, frame_idx]      # (n_cells, n_features)
        y = self.trajectories[traj_idx, frame_idx + 1]  # (n_cells, n_features)
        return x, y


# -------------------------------------------------------------------------
# Region-based tensors (cached on the right device)
# -------------------------------------------------------------------------

@dataclass
class RegionTensors:
    """Region adjacency tensors needed by GNN/CNN-style models, on the active device.

    Holds both the standard (unordered) neighbor tensors used by B3 (GNN with
    mean aggregation) and direction-sorted variants used by B4 (hex CNN with
    directional kernels) and M1 (gauge-equivariant convolution).
    """
    # Standard (unordered) neighbor info — used by B3 (GNN)
    neighbor_idx: torch.Tensor   # (n_cells, 6) int64; -1 sentinels for missing
    valid_mask: torch.Tensor     # (n_cells, 6) bool
    n_valid: torch.Tensor        # (n_cells,) int — number of valid neighbors

    # Direction-sorted neighbor info — used by B4 (hex CNN), M1 (gauge-equiv conv)
    # Position k corresponds to direction k*60° from East
    dir_neighbor_idx: torch.Tensor   # (n_cells, 6) int64; -1 for missing direction
    dir_valid_mask: torch.Tensor     # (n_cells, 6) bool

    @classmethod
    def from_region(cls, region: H3Region, device: torch.device) -> "RegionTensors":
        nbr = torch.from_numpy(region.neighbor_indices).long().to(device)
        valid = (nbr >= 0)
        n_valid = valid.sum(dim=1)
        dir_nbr = torch.from_numpy(region.direction_sorted_neighbor_indices).long().to(device)
        dir_valid = (dir_nbr >= 0)
        return cls(
            neighbor_idx=nbr, valid_mask=valid, n_valid=n_valid,
            dir_neighbor_idx=dir_nbr, dir_valid_mask=dir_valid,
        )


# -------------------------------------------------------------------------
# Loss and basic metrics
# -------------------------------------------------------------------------

def field_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RMSE over all cells, frames, and field components.

    Both tensors should have matching shape: (..., n_cells, n_features).
    """
    return float(torch.sqrt(F.mse_loss(pred, target)))


def persistence_rmse(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Trivial-baseline RMSE: predict the last input frame as the target sequence.

    inputs : (batch, n_input_frames, n_cells, n_features)
    targets: (batch, n_target_frames, n_cells, n_features)
    """
    last_frame = inputs[:, -1:].expand_as(targets)
    return field_rmse(last_frame, targets)


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    eval_horizon: int = 3       # forecast horizon for in-dist evaluation (doc 05 default)
    early_stop_patience: int = 10
    log_every: int = 1           # log every N epochs


def autoregressive_rollout(
    model: nn.Module,
    x_init: torch.Tensor,        # (batch, n_cells, n_features)
    region_tensors: RegionTensors,
    n_steps: int,
) -> torch.Tensor:
    """Generic autoregressive rollout. The model's forward signature is
    `model(x, region_tensors)` — each model picks the tensors it needs.

    Returns (batch, n_steps, n_cells, n_features).
    """
    states = []
    x = x_init
    for _ in range(n_steps):
        x = model(x, region_tensors)
        states.append(x)
    return torch.stack(states, dim=1)


def evaluate_rollout(
    model: nn.Module,
    dataset: TrajectoryDataset,
    region_tensors: RegionTensors,
    horizon: int,
    device: torch.device,
    batch_size: int = 32,
    input_frames: int = 16,
) -> dict:
    """Evaluate a model on a dataset by rolling out from frame `input_frames-1`
    for `horizon` steps and computing RMSE against ground-truth frames.

    Returns:
        {
            'rmse': float — overall RMSE
            'rmse_per_frame': list of float — per-frame RMSE
            'persistence_rmse': float — trivial-baseline RMSE for comparison
            'n_examples': int
        }
    """
    model.eval()
    all_preds = []
    all_targets = []

    inputs_full, targets_full = dataset.split_input_target(
        input_frames=input_frames, target_horizon=horizon,
    )
    inputs_full = torch.from_numpy(inputs_full).float()
    targets_full = torch.from_numpy(targets_full).float()

    with torch.no_grad():
        for start in range(0, len(inputs_full), batch_size):
            end = min(start + batch_size, len(inputs_full))
            inputs = inputs_full[start:end].to(device)
            targets = targets_full[start:end].to(device)

            x_init = inputs[:, -1]  # last input frame
            preds = autoregressive_rollout(model, x_init, region_tensors, horizon)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Per-frame RMSE
    rmse_per_frame = []
    for k in range(horizon):
        rmse_per_frame.append(field_rmse(all_preds[:, k], all_targets[:, k]))

    persistence = persistence_rmse(inputs_full, targets_full)

    return {
        "rmse": field_rmse(all_preds, all_targets),
        "rmse_per_frame": rmse_per_frame,
        "persistence_rmse": persistence,
        "n_examples": int(len(all_preds)),
    }


def train_model(
    model: nn.Module,
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    region_tensors: RegionTensors,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the standard training loop. Returns a dict of training history.

    The model trains on single-step prediction pairs from `train_dataset`,
    and is evaluated each epoch via `eval_horizon`-step autoregressive rollout
    on `val_dataset`.

    Saves per-epoch metrics CSV and best-checkpoint `.pt` to `output_dir` if provided.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pairs_dataset = SingleStepPairs(train_dataset.trajectories)
    loader = DataLoader(
        pairs_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0,  # CPU/Apple Silicon: 0 workers is fine
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "epoch": [], "train_loss": [], "val_rmse": [],
        "val_persistence_rmse": [], "wall_time_seconds": [],
    }

    best_val = float("inf")
    epochs_since_best = 0
    best_state = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "train_log.csv"
        # Write header once
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(history.keys())

    t_start = time.time()
    for epoch in range(1, config.n_epochs + 1):
        # --- Training ---
        model.train()
        epoch_losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x, region_tensors)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # --- Validation ---
        val_metrics = evaluate_rollout(
            model, val_dataset, region_tensors,
            horizon=config.eval_horizon, device=device,
        )
        elapsed = time.time() - t_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_persistence_rmse"].append(val_metrics["persistence_rmse"])
        history["wall_time_seconds"].append(elapsed)

        if output_dir is not None:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, train_loss, val_metrics["rmse"],
                    val_metrics["persistence_rmse"], elapsed,
                ])

        improved = val_metrics["rmse"] < best_val - 1e-6
        if improved:
            best_val = val_metrics["rmse"]
            epochs_since_best = 0
            # Save best checkpoint state in memory; persist at end.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1

        if verbose and epoch % config.log_every == 0:
            persistence = val_metrics["persistence_rmse"]
            ratio = val_metrics["rmse"] / persistence if persistence > 0 else float("inf")
            star = " *" if improved else "  "
            print(f"  epoch {epoch:3d}{star} | train_loss={train_loss:.5f} | "
                  f"val_rmse={val_metrics['rmse']:.5f} | "
                  f"persistence={persistence:.5f} | "
                  f"ratio={ratio:.3f} | t={elapsed:.0f}s")

        if epochs_since_best >= config.early_stop_patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no improvement for "
                      f"{config.early_stop_patience} epochs)")
            break

    # Restore best state and save
    if best_state is not None:
        model.load_state_dict(best_state)

    if output_dir is not None and best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
        with open(output_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    return {
        "history": history,
        "best_val_rmse": best_val,
        "n_epochs_run": history["epoch"][-1],
        "wall_time_seconds": history["wall_time_seconds"][-1],
    }


# -------------------------------------------------------------------------
# Multi-surface evaluation (doc 05 §4.2)
# -------------------------------------------------------------------------

def evaluate_all_surfaces(
    model: nn.Module,
    main_region_tensors: RegionTensors,
    held_out_region_tensors: RegionTensors,
    test_in_dist: TrajectoryDataset,
    test_ood_temporal_horizon: int,
    test_ood_spatial: TrajectoryDataset,
    test_ood_parameter: TrajectoryDataset,
    device: torch.device,
) -> dict:
    """Evaluate on the four surfaces from doc 05 §4.2.

    Returns:
        {
            'in_distribution':  evaluate_rollout output (horizon=3)
            'ood_temporal':     evaluate_rollout output (longer horizon)
            'ood_spatial':      evaluate_rollout output on held-out region
            'ood_parameter':    evaluate_rollout output on novel regime
        }
    """
    return {
        "in_distribution": evaluate_rollout(
            model, test_in_dist, main_region_tensors,
            horizon=3, device=device,
        ),
        "ood_temporal": evaluate_rollout(
            model, test_in_dist, main_region_tensors,
            horizon=test_ood_temporal_horizon, device=device,
        ),
        "ood_spatial": evaluate_rollout(
            model, test_ood_spatial, held_out_region_tensors,
            horizon=3, device=device,
        ),
        "ood_parameter": evaluate_rollout(
            model, test_ood_parameter, main_region_tensors,
            horizon=3, device=device,
        ),
    }
