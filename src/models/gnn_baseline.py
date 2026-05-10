"""
Baseline B3: standard GNN on H3 hex adjacency.

Architecture: GraphSAGE-mean style message passing.
  - Embed (u, v) state per cell into a hidden representation
  - Several rounds of message passing: aggregate neighbor features (mean),
    concat with self, transform via MLP
  - Decode hidden representation back to (u, v) prediction

This is a single-step predictor: given state at time t, produces state at
time t+1. Multi-step prediction (the experiment's actual task) is done via
autoregressive rollout at inference time.

The GNN uses no equivariance constraints — it tests whether explicit graph
structure (with H3 hex adjacency) plus expressive message passing is enough
to learn Gray-Scott dynamics. Comparison against M1 (H3-Oscillator) tells us
whether the equivariance constraint adds value beyond what a flexible GNN
provides.

Implementation: directly gathers from precomputed neighbor_indices, no
PyTorch Geometric dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexGNNLayer(nn.Module):
    """One round of message passing on the H3 hex adjacency.

    For each cell c with neighbors N(c):
        msg(c) = mean({h(c') for c' in N(c) if c' is in-region})
        h_new(c) = gelu(W [h(c), msg(c)])

    Boundary cells use only their valid (in-region) neighbors. The mask
    handles -1 sentinels in neighbor_indices (which mark invalid neighbors).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,             # (batch, n_cells, hidden)
        neighbor_idx: torch.Tensor,  # (n_cells, 6), int64; -1 marks missing
        valid_mask: torch.Tensor,    # (n_cells, 6), bool
        n_valid: torch.Tensor,       # (n_cells,), int — count of valid neighbors per cell
    ) -> torch.Tensor:
        # safe_idx: replace -1 with 0 so gather works; we'll mask invalid contributions
        safe_idx = neighbor_idx.clamp(min=0)  # (n_cells, 6)
        # h: (batch, n_cells, hidden); safe_idx is global cell-index lookup
        # Gather neighbors: index along dim=1
        # We need h[:, safe_idx, :] but safe_idx is (n_cells, 6), so we get (batch, n_cells, 6, hidden)
        nbr_h = h[:, safe_idx, :]  # (batch, n_cells, 6, hidden)

        # Mask invalid neighbors (those with -1 sentinel) by zero
        nbr_h = nbr_h * valid_mask[None, :, :, None].float()

        # Mean aggregation over the valid neighbors
        nbr_sum = nbr_h.sum(dim=2)  # (batch, n_cells, hidden)
        # Avoid division by zero (no isolated cells expected, but safe)
        denom = n_valid.clamp(min=1).float()  # (n_cells,)
        msg = nbr_sum / denom[None, :, None]  # (batch, n_cells, hidden)

        # Update: combine self and message
        combined = torch.cat([h, msg], dim=-1)  # (batch, n_cells, 2*hidden)
        return F.gelu(self.update(combined))


class HexGNN(nn.Module):
    """Single-step state predictor on H3 hex grid.

    Input  : state at time t,   shape (batch, n_cells, n_features)
    Output : state at time t+1, shape (batch, n_cells, n_features)

    Architecture:
      embed -> [GNN layer + residual]*n_layers -> decode
    """

    def __init__(
        self,
        n_features: int = 2,    # (u, v) for Gray-Scott
        hidden_dim: int = 32,
        n_layers: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.residual = residual

        self.embed = nn.Linear(n_features, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            HexGNNLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.decode = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """Predict state at next timestep using GNN message passing.

        Reads `neighbor_idx`, `valid_mask`, `n_valid` from `region_tensors`.

        Returns the *delta* from input applied — i.e., the model learns to
        predict the residual (next state - current state), which we add back.
        This is a standard trick for dynamical-system prediction; it makes
        the identity map easy to learn and improves stability.
        """
        nbr = region_tensors.neighbor_idx
        valid = region_tensors.valid_mask
        n_valid = region_tensors.n_valid

        h = self.embed(x)  # (batch, n_cells, hidden)
        for layer in self.gnn_layers:
            h_new = layer(h, nbr, valid, n_valid)
            h = h + h_new if self.residual else h_new

        delta = self.decode(h)  # (batch, n_cells, n_features)
        return x + delta  # residual prediction: x_{t+1} = x_t + delta

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: tiny model, random data, one forward.
    from dataclasses import dataclass

    torch.manual_seed(0)

    n_cells = 100
    batch = 4
    n_features = 2
    n_neighbors = 6

    # Fake neighbor structure
    neighbor_idx = torch.randint(0, n_cells, (n_cells, n_neighbors))
    neighbor_idx[0, 5] = -1
    neighbor_idx[1, 4] = -1
    valid_mask = neighbor_idx >= 0
    n_valid = valid_mask.sum(dim=1)

    # Build a minimal RegionTensors stub
    @dataclass
    class _RT:
        neighbor_idx: torch.Tensor
        valid_mask: torch.Tensor
        n_valid: torch.Tensor
        dir_neighbor_idx: torch.Tensor
        dir_valid_mask: torch.Tensor

    rt = _RT(
        neighbor_idx=neighbor_idx, valid_mask=valid_mask, n_valid=n_valid,
        dir_neighbor_idx=neighbor_idx, dir_valid_mask=valid_mask,
    )

    model = HexGNN(n_features=2, hidden_dim=32, n_layers=3)
    print(f"Model: {model}")
    print(f"Parameters: {model.n_parameters()}")

    x = torch.randn(batch, n_cells, n_features)
    y = model(x, rt)
    print(f"Single-step input  : {x.shape}")
    print(f"Single-step output : {y.shape}")
