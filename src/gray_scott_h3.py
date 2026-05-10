"""
Gray-Scott reaction-diffusion dynamics on H3 cells.

This module implements the synthetic data generator for the H3-Oscillator
experiment. It also provides baseline B1: the hand-crafted finite-difference
solver, which establishes the upper bound on prediction accuracy when the
true dynamics are known.

Gray-Scott model:
    du/dt = D_u * Laplacian(u) - u*v^2 + F*(1-u)
    dv/dt = D_v * Laplacian(v) + u*v^2 - (F+k)*v

Parameter regimes (from Pearson 1993):
    alpha (spots):   F=0.0367, k=0.0649  -> stable spots
    beta  (stripes): F=0.025,  k=0.050   -> striped patterns
    gamma (mazes):   F=0.029,  k=0.057   -> maze-like patterns
    delta (solitons):F=0.018,  k=0.051   -> self-replicating solitons

Diffusion coefficients fixed: D_u = 0.16, D_v = 0.08 (D_v < D_u for instability).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .h3_region import H3Region, laplacian


# Standard parameter regimes (Pearson 1993)
GS_REGIMES = {
    "alpha": {"F": 0.0367, "k": 0.0649, "description": "spots"},
    "beta":  {"F": 0.025,  "k": 0.050,  "description": "stripes"},
    "gamma": {"F": 0.029,  "k": 0.057,  "description": "mazes"},
    "delta": {"F": 0.018,  "k": 0.051,  "description": "solitons"},
}


@dataclass
class GrayScottParams:
    """Parameters for the Gray-Scott model."""
    F: float          # feed rate
    k: float          # kill rate
    D_u: float = 0.16 # diffusion coefficient for u
    D_v: float = 0.08 # diffusion coefficient for v
    dt: float = 1.0   # integration time step (in dimensionless units)

    @classmethod
    def from_regime(cls, regime: str, **overrides) -> "GrayScottParams":
        if regime not in GS_REGIMES:
            raise ValueError(f"Unknown regime '{regime}'. Choose from {list(GS_REGIMES)}")
        params = {"F": GS_REGIMES[regime]["F"], "k": GS_REGIMES[regime]["k"]}
        params.update(overrides)
        return cls(**params)


def gray_scott_step(
    u: np.ndarray,
    v: np.ndarray,
    region: H3Region,
    params: GrayScottParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Single forward Euler step of Gray-Scott dynamics.

    This is the core dynamics function. It serves both as the data generator
    (when iterated) and as baseline B1 (the hand-crafted finite-difference
    solver) when applied to predict future states from a given initial state.

    Parameters
    ----------
    u, v : np.ndarray of shape (n_cells,)
        Current state of the two coupled fields.
    region : H3Region
    params : GrayScottParams

    Returns
    -------
    u_next, v_next : np.ndarray of shape (n_cells,)
    """
    lap_u = laplacian(u, region)
    lap_v = laplacian(v, region)
    uvv = u * v * v

    du = params.D_u * lap_u - uvv + params.F * (1.0 - u)
    dv = params.D_v * lap_v + uvv - (params.F + params.k) * v

    u_next = u + params.dt * du
    v_next = v + params.dt * dv
    return u_next, v_next


def initialize_gs(
    region: H3Region,
    seed: int = 0,
    perturbation_fraction: float = 0.05,
    noise_sigma: float = 0.01,
    n_seeds: int = 1,
    seed_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize Gray-Scott fields with random perturbations.

    The steady state is u=1, v=0. We perturb several small contiguous regions
    (`n_seeds` of them, each of approximate size `seed_radius`) to u=0.5,
    v=0.25 to trigger pattern formation. Plus small Gaussian noise everywhere.

    For alpha (spots) regime, use n_seeds=8-16 because spots don't propagate
    and we need pattern coverage across the region.

    For gamma (mazes) regime, n_seeds=1 is fine because maze patterns
    propagate rapidly via reaction-diffusion fronts.

    Parameters
    ----------
    region : H3Region
    seed : int
        Random seed for reproducibility.
    perturbation_fraction : float in (0, 1)
        Total fraction of cells to perturb from steady state. Distributed
        across `n_seeds` seed locations.
    noise_sigma : float
        Standard deviation of Gaussian noise added to all cells.
    n_seeds : int, default 1
        Number of seed perturbation locations scattered across the region.
    seed_radius : int, default 3
        Approximate radius (in cells) of each seed perturbation.

    Returns
    -------
    u, v : np.ndarray of shape (n_cells,)
    """
    rng = np.random.default_rng(seed)
    n = region.n_cells
    centers = region.cell_centers  # (n, 2) lat/lon

    # Steady state
    u = np.ones(n)
    v = np.zeros(n)

    # Place n_seeds perturbation regions across the field.
    # Each seed perturbs ~seed_radius^2 cells (a rough disk).
    cells_per_seed = max(1, int(n * perturbation_fraction / n_seeds))
    perturb_indices_all = set()

    # Pick seed indices: try to spread them across the region by selecting
    # candidates uniformly at random, but accept only if far enough from
    # already-chosen seeds.
    seed_indices = []
    n_attempts = 0
    while len(seed_indices) < n_seeds and n_attempts < 100 * n_seeds:
        candidate = int(rng.integers(0, n))
        if not seed_indices:
            seed_indices.append(candidate)
        else:
            # Min distance to any chosen seed (in lat/lon units)
            chosen_centers = centers[seed_indices]
            dists = np.linalg.norm(chosen_centers - centers[candidate], axis=1)
            if dists.min() > 1.0:  # at least 1 degree apart
                seed_indices.append(candidate)
        n_attempts += 1
    # Fallback: if we couldn't find enough well-spread seeds, accept duplicates
    while len(seed_indices) < n_seeds:
        seed_indices.append(int(rng.integers(0, n)))

    for seed_idx in seed_indices:
        dists = np.linalg.norm(centers - centers[seed_idx], axis=1)
        nearest = np.argsort(dists)[:cells_per_seed]
        perturb_indices_all.update(int(i) for i in nearest)

    perturb_indices = np.array(sorted(perturb_indices_all), dtype=np.int64)
    u[perturb_indices] = 0.5
    v[perturb_indices] = 0.25

    # Small noise everywhere
    u += rng.normal(0, noise_sigma, n)
    v += rng.normal(0, noise_sigma, n)

    # Clip to valid range
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    return u, v


def integrate_trajectory(
    region: H3Region,
    params: GrayScottParams,
    n_steps: int,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    seed: int = 0,
    record_every: int = 1,
    verbose: bool = False,
    n_seeds: int = 1,
    perturbation_fraction: float = 0.05,
) -> np.ndarray:
    """Integrate Gray-Scott dynamics forward and record sampled frames.

    Parameters
    ----------
    region : H3Region
    params : GrayScottParams
    n_steps : int
        Total number of forward Euler steps to simulate.
    u0, v0 : np.ndarray of shape (n_cells,), optional
        Initial conditions. If None, generated via initialize_gs(seed).
    seed : int
        Random seed if initial conditions are not provided.
    record_every : int
        Save state every `record_every` steps. The initial state is always saved.
    verbose : bool
        Print progress every 1000 steps.
    n_seeds : int
        Number of perturbation seed locations (passed to initialize_gs).
    perturbation_fraction : float
        Total perturbation fraction (passed to initialize_gs).

    Returns
    -------
    trajectory : np.ndarray of shape (n_frames, n_cells, 2)
    """
    if u0 is None or v0 is None:
        u, v = initialize_gs(
            region, seed=seed, n_seeds=n_seeds,
            perturbation_fraction=perturbation_fraction,
        )
    else:
        u, v = u0.copy(), v0.copy()

    n_frames = n_steps // record_every + 1
    trajectory = np.zeros((n_frames, region.n_cells, 2))
    trajectory[0, :, 0] = u
    trajectory[0, :, 1] = v

    frame_idx = 1
    for step in range(1, n_steps + 1):
        u, v = gray_scott_step(u, v, region, params)
        if step % record_every == 0:
            if frame_idx < n_frames:
                trajectory[frame_idx, :, 0] = u
                trajectory[frame_idx, :, 1] = v
                frame_idx += 1
        if verbose and step % 1000 == 0:
            print(f"  step {step}/{n_steps}: u range [{u.min():.3f}, {u.max():.3f}], "
                  f"v range [{v.min():.3f}, {v.max():.3f}]")

    return trajectory


def predict_trajectory_handcrafted(
    region: H3Region,
    params: GrayScottParams,
    u_init: np.ndarray,
    v_init: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Baseline B1: hand-crafted finite-difference prediction.

    Given an initial state, integrate forward using the known dynamics.
    This is the upper bound on prediction accuracy: a model with perfect
    knowledge of the dynamics.

    Returns
    -------
    prediction : np.ndarray of shape (n_steps, n_cells, 2)
        Predicted future states (excluding the given initial state).
    """
    u, v = u_init.copy(), v_init.copy()
    prediction = np.zeros((n_steps, region.n_cells, 2))
    for t in range(n_steps):
        u, v = gray_scott_step(u, v, region, params)
        prediction[t, :, 0] = u
        prediction[t, :, 1] = v
    return prediction


if __name__ == "__main__":
    print("Gray-Scott on H3: smoke test")
    print("=" * 60)

    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    print(f"Region: {region}")

    for regime_name in ["alpha", "gamma"]:
        print(f"\n--- Regime: {regime_name} ({GS_REGIMES[regime_name]['description']}) ---")
        params = GrayScottParams.from_regime(regime_name)
        print(f"Parameters: F={params.F}, k={params.k}, D_u={params.D_u}, D_v={params.D_v}")

        # Run a short trajectory
        traj = integrate_trajectory(
            region=region, params=params, n_steps=2000,
            seed=42, record_every=200, verbose=True,
        )
        print(f"Trajectory shape: {traj.shape}")
        print(f"Final u: range [{traj[-1, :, 0].min():.3f}, {traj[-1, :, 0].max():.3f}], "
              f"std={traj[-1, :, 0].std():.4f}")
        print(f"Final v: range [{traj[-1, :, 1].min():.3f}, {traj[-1, :, 1].max():.3f}], "
              f"std={traj[-1, :, 1].std():.4f}")
