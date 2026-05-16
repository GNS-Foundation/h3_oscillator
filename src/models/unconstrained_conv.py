"""
src/models/unconstrained_conv.py

UnconstrainedRegularToRegular: a non-equivariant version of RegularToRegular,
used for the B5 ablation (Phase 2 / M1 Step 5).

Purpose: B5 = M1 with the block-circulant C6 constraint REMOVED from the
dynamics layer's recurrent operator W_h. This layer provides that
unconstrained drop-in replacement.

Same interface as RegularToRegular:
    forward(x, region_tensors) where x has shape (batch, n_cells, in_ch, 6)
    returns (batch, n_cells, out_ch, 6)

Parameterization:
    - 7 spatial positions (1 center + 6 directional slots)
    - At each position: a free 6×6 weight matrix (no constraints)
    - No circulant constraint on center (vs M1's R2R: 6 params)
    - No cyclic conjugation tying directional positions (vs M1's R2R: 36 params)

Free params per (in_ch, out_ch) pair: 7 * 6 * 6 = 252
    (vs RegularToRegular: 42 — i.e. 6× more capacity)

By construction this layer is NOT C6-equivariant: removing the constraint is
the point. The verification at the bottom of this file confirms equivariance
fails (large error), which is the desired behavior for B5.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

NUM_DIRS = 6


class UnconstrainedRegularToRegular(nn.Module):
    """B5 ablation operator: NOT C6-equivariant.

    For each (in_ch, out_ch) pair: 7 free 6×6 matrices (one per spatial position).
    Center matrix: NOT required to be circulant.
    Directional matrices: NOT tied via cyclic conjugation.

    Total free params per (in_ch, out_ch) pair: 252
    (M1's R2R has 42 — B5 has 6× more capacity per pair)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialization: fan-in equivalent to RegularToRegular
        # Effective fan-in: in_ch × 6 (input dirs) × 7 (positions)
        std = 1.0 / math.sqrt(in_channels * NUM_DIRS * 7)

        # 7 spatial positions × in_ch × out_ch × 6_out_dir × 6_in_dir
        # Position 0 is center, positions 1-6 are the 6 directional slots
        self.weight = nn.Parameter(
            torch.randn(7, in_channels, out_channels, NUM_DIRS, NUM_DIRS) * std
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Input  x: (batch, n_cells, in_ch, 6) — regular features
        Output  : (batch, n_cells, out_ch, 6) — regular features
        """
        dir_idx = region_tensors.dir_neighbor_idx
        dir_mask = region_tensors.dir_valid_mask

        # Gather neighbors at directional slots
        safe_idx = dir_idx.clamp(min=0)
        x_nbr = x[:, safe_idx]      # (batch, n_cells, 6_slots, in_ch, 6_in_dir)
        x_nbr = x_nbr * dir_mask[None, :, :, None, None].to(x_nbr.dtype)

        # Split: position 0 = center, positions 1-6 = directional slots
        center_kernel = self.weight[0]      # (in_ch, out_ch, 6_d_out, 6_e_in)
        dir_kernels = self.weight[1:]       # (6_slots, in_ch, out_ch, 6_d_out, 6_e_in)

        # Center: sum over in_ch, in_dir
        center_contrib = torch.einsum("bcie,ijde->bcjd", x, center_kernel)

        # Directional: sum over slots, in_ch, in_dir
        nbr_contrib = torch.einsum("bckie,kijde->bcjd", x_nbr, dir_kernels)

        # Bias broadcast across all 6 output directions (note: not C6-trivial,
        # but bias-only scalar; doesn't reintroduce equivariance)
        return center_contrib + nbr_contrib + self.bias[None, None, :, None]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Confirm NON-equivariance (sanity check for the ablation)
# ============================================================================

def _verify_non_equivariance(seed: int = 0) -> float:
    """Confirm that UnconstrainedRegularToRegular is NOT C6-equivariant.

    For an ablation to be meaningful, we must verify the constraint is actually
    removed. This test should yield O(1) equivariance error, confirming the
    layer breaks rotation symmetry (which is the entire point of B5).
    """
    from src.models.equivariant_conv import _build_synthetic_hex_test_setup

    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)

    layer = UnconstrainedRegularToRegular(2, 3).double()
    x = torch.randn(2, 7, 2, NUM_DIRS, dtype=torch.float64)
    x_rot = x[:, perm_inv].roll(shifts=1, dims=-1)

    y = layer(x, rt)
    y_rot = layer(x_rot, rt)

    # If this layer were equivariant: y_rot[:, 0] would equal P · y[:, 0]
    # We expect this to FAIL with O(1) error
    expected = y[:, perm_inv].roll(shifts=1, dims=-1)
    diff = (y_rot[:, 0] - expected[:, 0]).abs().max().item()
    return diff


if __name__ == "__main__":
    print("=" * 64)
    print("UnconstrainedRegularToRegular: NON-equivariance sanity check")
    print("=" * 64)
    print("\nWe REMOVE the C6 constraint, so equivariance must FAIL.")
    print("If equivariance error < 1.0, something is wrong (ablation is fake).\n")

    for seed in [0, 1, 2, 3, 42]:
        diff = _verify_non_equivariance(seed)
        status = "✓ NOT EQUIVARIANT (correct)" if diff > 0.1 else "✗ UNEXPECTEDLY EQUIVARIANT"
        print(f"  seed {seed}: equivariance error = {diff:.4f}   {status}")

    # Parameter count comparison
    print("\n--- Parameter count (B5 vs M1's R2R) ---")
    from src.models.equivariant_conv import RegularToRegular
    for (in_ch, out_ch) in [(8, 8), (4, 4), (16, 16)]:
        r2r = RegularToRegular(in_ch, out_ch)
        r2r_u = UnconstrainedRegularToRegular(in_ch, out_ch)
        ratio = r2r_u.n_parameters() / r2r.n_parameters()
        print(f"  ({in_ch}→{out_ch}):  M1 R2R = {r2r.n_parameters():5d}  |  "
              f"B5 R2R_U = {r2r_u.n_parameters():5d}  ({ratio:.2f}× larger)")
