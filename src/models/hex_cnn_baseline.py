"""
Baseline B4: static hex CNN with direction-aware kernels.

Architecture: HexagDLy-style hex convolution. Each layer has 7 free weights
per (in_channel, out_channel) pair:
  - 1 weight for the center cell
  - 6 weights, one for each hex direction (0°, 60°, 120°, 180°, 240°, 300°)

Unlike B3 (GNN with mean aggregation), B4 distinguishes between neighbors at
different compass directions — meaning the kernel can learn anisotropic
features. This makes B4 strictly more expressive than B3 at matched per-layer
parameter count.

What B4 does NOT have (in contrast to M1):
  - No C6 equivariance (kernel weights are independent across the 6 directions)
  - No dynamics layer (no continuous-time bounded ODE)
  - No iterative state evolution within a forward pass

This is the cleanest "hex convolution alone" baseline. If M1 outperforms B4
significantly, that tells us the dynamics + equivariance are doing real work.
If B4 matches or beats M1, the elegance isn't paying off.

Implementation: directly gathers from H3Region.direction_sorted_neighbor_indices
which is precomputed during region setup. No HexagDLy or external library
dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexConv(nn.Module):
    """One hex convolution layer with direction-aware kernels.

    Kernel structure: 7 free weights per (in_ch, out_ch) pair.
      - W_center: applies to the cell's own value
      - W_0..W_5: applies to the cell's neighbors at directions 0°..300°

    For boundary cells with fewer than 6 valid neighbors, missing directional
    slots contribute zero (handled by valid_mask).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Center weight (with bias)
        self.center = nn.Linear(in_channels, out_channels, bias=True)
        # 6 directional weights (no biases — all bias goes through center)
        self.directional = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(6)
        ])

    def forward(
        self,
        x: torch.Tensor,                  # (batch, n_cells, in_channels)
        dir_neighbor_idx: torch.Tensor,   # (n_cells, 6) — direction-sorted, -1 for missing
        dir_valid_mask: torch.Tensor,     # (n_cells, 6) bool — True where direction has neighbor
    ) -> torch.Tensor:
        # Center contribution
        out = self.center(x)  # (batch, n_cells, out_channels)

        # Directional contributions
        for k in range(6):
            nbr_idx = dir_neighbor_idx[:, k]  # (n_cells,)
            mask = dir_valid_mask[:, k]       # (n_cells,)
            safe_idx = nbr_idx.clamp(min=0)   # replace -1 with 0
            nbr_x = x[:, safe_idx]            # (batch, n_cells, in_channels)
            # Zero out invalid (boundary) neighbor contributions
            nbr_x = nbr_x * mask[None, :, None].float()
            # Apply directional weight
            out = out + self.directional[k](nbr_x)

        return out


class HexCNN(nn.Module):
    """Static hex CNN: single-step state predictor, no dynamics layer.

    Architecture:
        embed (linear)
        -> [HexConv + GELU + (residual)] * n_layers
        -> decode (linear)
        -> residual prediction: x_{t+1} = x_t + delta

    Same training API as B3: forward(x, region_tensors). Reads
    `dir_neighbor_idx` and `dir_valid_mask` from region_tensors for the
    direction-aware kernels.
    """

    def __init__(
        self,
        n_features: int = 2,
        hidden_dim: int = 20,
        n_layers: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.residual = residual

        self.embed = nn.Linear(n_features, hidden_dim)
        self.hex_layers = nn.ModuleList([
            HexConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.decode = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """Predict state at next timestep using static hex convolution.

        Reads `dir_neighbor_idx` (direction-sorted indices) and
        `dir_valid_mask` from `region_tensors`.
        """
        dir_idx = region_tensors.dir_neighbor_idx
        dir_mask = region_tensors.dir_valid_mask

        h = self.embed(x)
        for layer in self.hex_layers:
            h_new = F.gelu(layer(h, dir_idx, dir_mask))
            h = h + h_new if self.residual else h_new

        delta = self.decode(h)
        return x + delta  # residual prediction

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: tiny model, random data, one forward.
    from dataclasses import dataclass

    torch.manual_seed(0)

    n_cells = 100
    batch = 4
    n_features = 2

    # Fake direction-sorted neighbor structure
    dir_idx = torch.randint(0, n_cells, (n_cells, 6))
    dir_idx[0, 5] = -1  # one missing direction
    dir_mask = dir_idx >= 0

    @dataclass
    class _RT:
        neighbor_idx: torch.Tensor
        valid_mask: torch.Tensor
        n_valid: torch.Tensor
        dir_neighbor_idx: torch.Tensor
        dir_valid_mask: torch.Tensor

    rt = _RT(
        neighbor_idx=dir_idx, valid_mask=dir_mask, n_valid=dir_mask.sum(dim=1),
        dir_neighbor_idx=dir_idx, dir_valid_mask=dir_mask,
    )

    model = HexCNN(n_features=2, hidden_dim=20, n_layers=3)
    print(f"Model: {model}")
    print(f"Parameters: {model.n_parameters()}")

    x = torch.randn(batch, n_cells, n_features)
    y = model(x, rt)
    print(f"Single-step input  : {x.shape}")
    print(f"Single-step output : {y.shape}")
