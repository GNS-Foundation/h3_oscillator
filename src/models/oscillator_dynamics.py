"""
src/models/oscillator_dynamics.py

H3OscillatorDynamics: LTC (Liquid Time-Constant) closed-form dynamics layer
on H3 hex grid with C6-equivariant operators.

Hidden state h has shape (batch, n_cells, F, 6) — F regular features per cell,
each is a 6-component vector that cyclically permutes under hex rotations.

Continuous-time ODE (Hasani 2021):
    dh/dt = -[1/τ + g(x, h)] · h + g(x, h) · A
    g(x, h) = tanh(W_h · h + W_x · x)

Closed-form CfC integrator (treating g as constant over [t, t+Δt]):
    h(t+Δt) = (h(t) - A) · exp(-Δt · [1/τ + g]) + A

Equivariance:
  - W_h is R2R (C6-equivariant by construction, Step 1)
  - W_x is S2R (C6-equivariant by construction, Step 1)
  - A, τ are per-channel scalars broadcast to all 6 directions (C6-trivial)
  - tanh, exp are element-wise (preserve equivariance)
  - Sums and element-wise products preserve equivariance
  → The full LTC update is C6-equivariant.

The block-circulant R2R weight is the architectural constraint that B5 will
ablate (replaced with an unconstrained directional hex kernel from B4).

Stability: 1/τ + g must be positive for h to decay toward A. Since g = tanh
∈ [-1, 1], we need 1/τ > 1, i.e., τ < 1. The default init log_tau = log(0.5)
gives 1/τ = 2 at start. Training is unconstrained and may push log_tau higher;
gradient clipping + small dt should keep it stable in practice.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.models.equivariant_conv import (
    ScalarToRegular, RegularToRegular,
)


class H3OscillatorDynamics(nn.Module):
    """One LTC step with C6-equivariant operators.

    Args:
        F_dim: regular-feature multiplicity (cell hidden state is F_dim × 6)
        n_features: number of scalar input channels (e.g., 2 for u, v)
        log_tau_init: initial value of log(τ). Default log(0.5) = -0.693 gives
                      τ=0.5, 1/τ=2 at init, leaving margin for stability since
                      g = tanh ∈ [-1, 1] so 1/τ + g ∈ [1, 3] > 0 at start.
    """

    def __init__(
        self,
        F_dim: int = 8,
        n_features: int = 2,
        log_tau_init: float = math.log(0.5),
    ):
        super().__init__()
        self.F_dim = F_dim
        self.n_features = n_features

        # State-to-state operator: C6-equivariant R2R (block-circulant per Step 1).
        # This is the constraint B5 will ablate.
        self.W_h = RegularToRegular(F_dim, F_dim)

        # Input-to-state operator: C6-equivariant S2R
        self.W_x = ScalarToRegular(n_features, F_dim)

        # Per-channel scalars, broadcast across all 6 directions (C6-trivial subspace)
        self.A = nn.Parameter(torch.zeros(F_dim))             # asymptote
        self.log_tau = nn.Parameter(torch.full((F_dim,), log_tau_init))

        # No additional bias here — W_h.bias and W_x.bias together provide the
        # "b" term in g = tanh(W_h h + W_x x + b).

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        region_tensors,
        dt: float = 0.25,
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, n_cells, F_dim, 6) — hidden regular features
            x: (batch, n_cells, n_features) — scalar input (same x reused at each step)
            region_tensors: RegionTensors with dir_neighbor_idx, dir_valid_mask
            dt: integration step size

        Returns:
            (batch, n_cells, F_dim, 6) — updated hidden regular features
        """
        # g = tanh(W_h h + W_x x)
        Wh_h = self.W_h(h, region_tensors)              # (B, N, F, 6)
        Wx_x = self.W_x(x, region_tensors)              # (B, N, F, 6)
        g = torch.tanh(Wh_h + Wx_x)

        # Per-channel scalars, broadcast across 6 directions (C6-trivial)
        inv_tau = torch.exp(-self.log_tau)              # (F,) — always > 0
        inv_tau = inv_tau[None, None, :, None]          # (1, 1, F, 1)
        A = self.A[None, None, :, None]                 # (1, 1, F, 1)

        # CfC closed-form update.
        # decay = exp(-dt * (1/τ + g)). Stable if 1/τ + g > 0, otherwise can grow.
        decay = torch.exp(-dt * (inv_tau + g))
        h_next = (h - A) * decay + A

        return h_next

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
