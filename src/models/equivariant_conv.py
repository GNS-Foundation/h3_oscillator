"""
C6-equivariant convolution primitives for the H3-Oscillator architecture.

Three layer types, following Cohen et al. 2019:

  ScalarToRegular (S2R):
      Input:  scalar field (1 channel per cell)
      Output: regular feature field (6 channels per cell)
      Free params per (in_ch, out_ch) pair: 7  (1 center + 6 directional)

  RegularToRegular (R2R):
      Input:  regular feature field (6 channels per cell)
      Output: regular feature field (6 channels per cell)
      Free params per (in_ch, out_ch) pair: 42 (6 center circulant + 36 one
      directional matrix; the other 5 directional positions are determined
      by cyclic conjugation)

  RegularToScalar (R2S):
      Input:  regular feature field (6 channels per cell)
      Output: scalar field (1 channel per cell)
      Free params per (in_ch, out_ch) pair: 7  (1 center + 6 directional)

Convention for the C6 group action:
  - On scalar fields: g acts only by permuting cells.
  - On regular features: g permutes cells AND cyclically shifts the 6
    directional components. We use the convention that 60° CCW rotation
    sends component d to component d+1 (mod 6).
  - On directional neighbor slots: 60° CCW rotation sends the cell at
    slot k of cell c to slot k+1 of the rotated cell (slots are sorted
    by global compass angle, so they re-index under rotation).

The kernel parameterizations below enforce equivariance under this group
action by construction. A numerical verification test is included in the
__main__ block.

Implementation notes:
  - All tensors use the convention (batch, n_cells, channels, ...) where
    regular features have an extra trailing dim of size 6.
  - We rely on `region_tensors.dir_neighbor_idx` (direction-sorted) and
    `dir_valid_mask` (boundary handling) from src/training.py's RegionTensors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _gather_neighbors_scalar(
    x: torch.Tensor,             # (batch, n_cells, in_ch)
    dir_idx: torch.Tensor,       # (n_cells, 6) int64; -1 for missing
    dir_mask: torch.Tensor,      # (n_cells, 6) bool
) -> torch.Tensor:
    """Gather scalar inputs from each cell's 6 direction-sorted neighbors.

    Returns: (batch, n_cells, 6, in_ch). Missing neighbors are zeroed.
    """
    safe_idx = dir_idx.clamp(min=0)              # (n_cells, 6)
    nbr = x[:, safe_idx, :]                       # (batch, n_cells, 6, in_ch)
    return nbr * dir_mask[None, :, :, None].float()


def _gather_neighbors_regular(
    x: torch.Tensor,             # (batch, n_cells, in_ch, 6)
    dir_idx: torch.Tensor,       # (n_cells, 6)
    dir_mask: torch.Tensor,      # (n_cells, 6)
) -> torch.Tensor:
    """Gather regular-feature inputs from each cell's 6 neighbors.

    Returns: (batch, n_cells, 6, in_ch, 6). Missing neighbors are zeroed.
    The middle dim 6 indexes the *neighbor slot* (which slot of the cell);
    the trailing dim 6 indexes the *feature direction* of the regular feature.
    """
    safe_idx = dir_idx.clamp(min=0)
    nbr = x[:, safe_idx, :, :]                    # (batch, n_cells, 6, in_ch, 6)
    return nbr * dir_mask[None, :, :, None, None].float()


# -------------------------------------------------------------------------
# Scalar to Regular (S2R)
# -------------------------------------------------------------------------

class ScalarToRegular(nn.Module):
    """C6-equivariant scalar-to-regular convolution.

    Free params per (in_ch, out_ch) pair: 7 (1 center + 6 directional).

    Formula:
        out[c, d] = w_center * x[c]
                  + Σ_k w_{(k - d) mod 6 + 1} * x[neighbor_k(c)]

    where index 0 is the center and indices 1..6 are directional weights.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # weight[..., 0] = center, weight[..., 1..6] = the 6 directional
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 7))
        # One bias per output channel; broadcast across the 6 feature directions.
        # (For regular outputs, a "scalar bias" lives in the trivial subspace
        # of the regular representation, which is 1-dim — matches our scalar
        # parameterization.)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

        # Precompute the (d, k) -> directional weight index lookup.
        # weight_idx[d, k] = (k - d) mod 6, then we use weight[..., 1 + idx].
        self.register_buffer(
            "_weight_idx",
            torch.tensor([[(k - d) % 6 for k in range(6)] for d in range(6)],
                         dtype=torch.long),
        )

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, in_ch) — scalar field
            region_tensors: RegionTensors with dir_neighbor_idx and dir_valid_mask
        Returns:
            (batch, n_cells, out_ch, 6) — regular feature field
        """
        dir_idx = region_tensors.dir_neighbor_idx
        dir_mask = region_tensors.dir_valid_mask

        # Center contribution: same for all 6 output directions
        # weight[..., 0] : (out_ch, in_ch)
        # x: (batch, n_cells, in_ch) -> output_center: (batch, n_cells, out_ch)
        out_center = torch.einsum("bnc,oc->bno", x, self.weight[..., 0])

        # Directional contributions
        # Gather: (batch, n_cells, 6, in_ch) where dim 2 = neighbor slot k
        nbr = _gather_neighbors_scalar(x, dir_idx, dir_mask)

        # Build per-(d, k) weight tensor: K[d, k, o, c] = weight[o, c, 1 + (k - d) mod 6]
        # weight_dir: (out_ch, in_ch, 6) — last dim is the 6 directional weights
        weight_dir = self.weight[..., 1:]                # (out_ch, in_ch, 6)
        # K: (out_ch, in_ch, 6, 6) where the last two dims are (d, k)
        K = weight_dir[..., self._weight_idx]            # (out_ch, in_ch, 6, 6)
        # K[o, c, d, k] = weight_dir[o, c, _weight_idx[d, k]]

        # Compute output[b, n, o, d] = sum_{c, k} K[o, c, d, k] * nbr[b, n, k, c]
        out_dir = torch.einsum("ocdk,bnkc->bnod", K, nbr)

        # Combine: center is shared across d, directional varies by d
        out = out_center.unsqueeze(-1) + out_dir + self.bias[None, None, :, None]
        # out: (batch, n_cells, out_ch, 6)
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Regular to Regular (R2R)
# -------------------------------------------------------------------------

class RegularToRegular(nn.Module):
    """C6-equivariant regular-to-regular convolution.

    Free params per (in_ch, out_ch) pair: 42
        - 6 center circulant params (commutator constraint)
        - 36 free params for direction-0 matrix (other 5 directions determined
          by cyclic conjugation: K_k = P^k @ K_0 @ P^{-k})

    Formula at cell c:
        out[c, d_out] = Σ_{d_in} K_center[d_out, d_in] * x[c, d_in]
                      + Σ_k Σ_{d_in} K_k[d_out, d_in] * x[neighbor_k(c), d_in]

    where:
        K_center[i, j] = c[(i - j) mod 6]      (circulant, parameterized by c ∈ R^6)
        K_k[i, j] = K_0[(i - k) mod 6, (j - k) mod 6]    (cyclic conjugation)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Center circulant: 6 params per (out_ch, in_ch) pair
        self.center_w = nn.Parameter(torch.empty(out_channels, in_channels, 6))
        # Direction-0 full 6x6 matrix per (out_ch, in_ch) pair
        self.dir0_w = nn.Parameter(torch.empty(out_channels, in_channels, 6, 6))
        # Bias (broadcast across feature directions: must be C6-trivial)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Initialize with reduced gain to compensate for 7-position summation
        gain_factor = 1.0 / math.sqrt(7.0)
        nn.init.kaiming_normal_(self.center_w, nonlinearity='linear')
        nn.init.kaiming_normal_(self.dir0_w, nonlinearity='linear')
        with torch.no_grad():
            self.center_w.mul_(gain_factor)
            self.dir0_w.mul_(gain_factor)

        # Precompute index for circulant construction: circ_idx[i, j] = (i - j) mod 6
        idx = torch.arange(6)
        self.register_buffer("_circ_idx", (idx[:, None] - idx[None, :]) % 6)

        # Precompute index for directional matrix construction:
        # K_k[i, j] = K_0[(i - k) mod 6, (j - k) mod 6]
        # We'll build via advanced indexing: shifted_idx[k, i] = (i - k) mod 6
        ks = torch.arange(6)
        self.register_buffer("_shift_idx", (idx[None, :] - ks[:, None]) % 6)
        # _shift_idx[k, i] = (i - k) mod 6, shape (6, 6)

    def _build_kernel(self) -> torch.Tensor:
        """Build the full kernel tensor (out_ch, in_ch, 7, 6, 6).

        Position 0 is center, positions 1..6 are the 6 directional slots.
        kernel[..., p, i, j] gives the weight from input direction j at position p
        to output direction i at the central cell.
        """
        # Center: (out_ch, in_ch, 6, 6) circulant
        center_kernel = self.center_w[..., self._circ_idx]
        # center_kernel[o, c, i, j] = center_w[o, c, (i - j) mod 6]

        # Directional: 6 matrices, each is dir0 with rows AND cols both shifted
        # by k positions. We build all 6 at once via advanced indexing.
        # K_k[i, j] = dir0[(i-k) mod 6, (j-k) mod 6]
        shift = self._shift_idx  # (6, 6)
        # For each k, we want dir0 indexed by (shift[k, :], shift[k, :])
        # dir0: (out_ch, in_ch, 6, 6) - (o, c, row, col)
        # We use einsum-like indexing: dir_kernel[k, i, j] = dir0[shift[k, i], shift[k, j]]
        # Build via gather:
        dir_kernel_all = self.dir0_w[..., shift[:, :, None], shift[:, None, :]]
        # Indexing self.dir0_w[..., A, B] where A: (6, 6, 1), B: (6, 1, 6)
        # broadcasts to (6, 6, 6) for the last 3 dims. Result: (out_ch, in_ch, 6, 6, 6)
        # where the leading 6 indexes k, and the (6, 6) indexes (i, j).

        # Concatenate center + 6 directional into a 7-position kernel
        kernel = torch.cat([
            center_kernel.unsqueeze(2),  # (out_ch, in_ch, 1, 6, 6)
            dir_kernel_all,              # (out_ch, in_ch, 6, 6, 6)
        ], dim=2)
        return kernel  # (out_ch, in_ch, 7, 6, 6)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, in_ch, 6) — regular feature field
            region_tensors: RegionTensors with dir_neighbor_idx and dir_valid_mask
        Returns:
            (batch, n_cells, out_ch, 6) — regular feature field
        """
        dir_idx = region_tensors.dir_neighbor_idx
        dir_mask = region_tensors.dir_valid_mask

        kernel = self._build_kernel()  # (out_ch, in_ch, 7, 6, 6)
        center_kernel = kernel[..., 0, :, :]   # (out_ch, in_ch, 6, 6) = (o, c, i, j)
        dir_kernel = kernel[..., 1:, :, :]     # (out_ch, in_ch, 6, 6, 6) = (o, c, k, i, j)

        # Center contribution: out_center[b, n, o, i] = sum_{c, j} center[o, c, i, j] * x[b, n, c, j]
        out_center = torch.einsum("ocij,bncj->bnoi", center_kernel, x)

        # Gather neighbors: (batch, n_cells, 6, in_ch, 6)  -- (b, n, k, c, j)
        nbr = _gather_neighbors_regular(x, dir_idx, dir_mask)

        # Directional: out_dir[b, n, o, i] = sum_{c, k, j} dir_kernel[o, c, k, i, j] * nbr[b, n, k, c, j]
        out_dir = torch.einsum("ockij,bnkcj->bnoi", dir_kernel, nbr)

        out = out_center + out_dir + self.bias[None, None, :, None]
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Regular to Scalar (R2S)
# -------------------------------------------------------------------------

class RegularToScalar(nn.Module):
    """C6-equivariant regular-to-scalar convolution.

    Free params per (in_ch, out_ch) pair: 7 (1 center + 6 directional).

    Formula:
        out[c] = w_0 * Σ_{d_in} x[c, d_in]
               + Σ_k Σ_{d_in} w_{(k - d_in) mod 6 + 1} * x[neighbor_k(c), d_in]

    Note: The center's single weight is *applied to the sum* over input directions
    (which is the C6-trivial component). The directional structure couples
    neighbor slot k with input direction d_in via shifted indexing.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # weight[..., 0] = center, weight[..., 1..6] = directional
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 7))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

        # weight_idx[k, d_in] = (k - d_in) mod 6
        self.register_buffer(
            "_weight_idx",
            torch.tensor([[(k - d) % 6 for d in range(6)] for k in range(6)],
                         dtype=torch.long),
        )

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, in_ch, 6) — regular feature field
            region_tensors: RegionTensors with dir_neighbor_idx and dir_valid_mask
        Returns:
            (batch, n_cells, out_ch) — scalar field
        """
        dir_idx = region_tensors.dir_neighbor_idx
        dir_mask = region_tensors.dir_valid_mask

        # Center contribution: w_0 * sum_{d_in} x[c, d_in]
        # weight[..., 0]: (out_ch, in_ch)
        # x.sum(dim=-1): (batch, n_cells, in_ch)
        x_sum = x.sum(dim=-1)
        out_center = torch.einsum("bnc,oc->bno", x_sum, self.weight[..., 0])

        # Directional contribution
        # Gather neighbors: (batch, n_cells, 6, in_ch, 6) = (b, n, k, c, d_in)
        nbr = _gather_neighbors_regular(x, dir_idx, dir_mask)

        # Build per-(k, d_in) weight tensor: K[k, d_in, o, c] = weight[o, c, 1 + (k-d_in) mod 6]
        weight_dir = self.weight[..., 1:]                    # (out_ch, in_ch, 6)
        # K: (out_ch, in_ch, 6, 6) where last two dims are (k, d_in)
        K = weight_dir[..., self._weight_idx]                 # (out_ch, in_ch, 6, 6)

        # out_dir[b, n, o] = sum_{c, k, d_in} K[o, c, k, d_in] * nbr[b, n, k, c, d_in]
        out_dir = torch.einsum("ockj,bnkcj->bno", K, nbr)

        out = out_center + out_dir + self.bias[None, None, :]
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Numerical equivariance verification (run as __main__)
# -------------------------------------------------------------------------

def _build_synthetic_hex_test_setup(seed: int = 0):
    """Build a 7-cell hex test region: 1 center + 6 outer cells in a ring.

    The 6 outer cells are arranged at directions 0..5 (slots 0..5 of cell 0).
    The "C6 rotation" we test is the 60° CCW rotation of the field.

    Key insight: adjacency is determined by physical positions. Cells don't
    move under rotation — field values rotate. So pre- and post-rotation
    use the SAME adjacency tensor.

    Under 60° CCW rotation, the value at position p_k post-rotation equals
    the value at position p_{k-1} pre-rotation. In cell labels (cell j+1
    is at position p_j for j=0..5):
        x_post[cell k+1] = x_pre[cell at p_{k-1}] = x_pre[cell ((k-1) mod 6) + 1]
        x_post[cell 0] = x_pre[cell 0]

    The corresponding `perm_inv` (where x_post[c] = x_pre[perm_inv[c]]) is:
        perm_inv = [0, 6, 1, 2, 3, 4, 5]

    Returns:
        region_tensors  — adjacency (fixed across rotation)
        perm_inv        — cell-permutation index for the input rotation
    """
    n_cells = 7
    # Adjacency: cell 0's slots 0..5 -> cells 1..6.
    # Outer cells (1..6) each have cell 0 at their "opposite" slot (slot 3 by
    # convention, since cell 0 is to the West of cell 1, which is at the East).
    dir_idx = torch.full((n_cells, 6), -1, dtype=torch.long)
    dir_idx[0] = torch.arange(1, 7)
    for k in range(6):
        dir_idx[k + 1, (k + 3) % 6] = 0
    dir_mask = dir_idx >= 0

    from dataclasses import dataclass
    @dataclass
    class _RT:
        neighbor_idx: torch.Tensor
        valid_mask: torch.Tensor
        n_valid: torch.Tensor
        dir_neighbor_idx: torch.Tensor
        dir_valid_mask: torch.Tensor

    rt = _RT(
        neighbor_idx=dir_idx, valid_mask=dir_mask,
        n_valid=dir_mask.sum(dim=1),
        dir_neighbor_idx=dir_idx, dir_valid_mask=dir_mask,
    )

    # Permutation: x_post[cell j+1] = x_pre[cell ((j-1) mod 6) + 1]
    # perm_inv[j] = source cell for value at cell j post-rotation
    perm_inv = torch.tensor([0, 6, 1, 2, 3, 4, 5], dtype=torch.long)

    return rt, perm_inv


def _verify_s2r_equivariance(seed: int = 0, tol: float = 1e-5):
    """Verify ScalarToRegular satisfies C6 equivariance.

    Equivariance condition for our 60° CCW rotation g:
        K(g · x)[c, d] = (g · K(x))[c, d] = K(x)[perm_inv[c], (d-1) mod 6]
    """
    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)
    layer = ScalarToRegular(in_channels=1, out_channels=2)

    x_pre = torch.randn(1, 7, 1)
    x_post = x_pre[:, perm_inv, :]  # rotated input (cells permuted)

    y_pre = layer(x_pre, rt)
    y_post = layer(x_post, rt)

    # Expected: y_post[c, o, d] = y_pre[perm_inv[c], o, (d-1) mod 6]
    expected = y_pre[:, perm_inv, :, :].roll(shifts=1, dims=-1)

    # Check at cell 0 (only cell with full adjacency)
    diff_cell0 = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    return diff_cell0 < tol, diff_cell0


def _verify_r2r_equivariance(seed: int = 0, tol: float = 1e-5):
    """Verify R2R equivariance. Input is regular feature: rotation acts on
    cells AND on the 6 feature directions.
    """
    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)
    layer = RegularToRegular(in_channels=2, out_channels=2)

    x_pre = torch.randn(1, 7, 2, 6)
    # Rotate: permute cells AND cyclic-shift feature directions
    x_post = x_pre[:, perm_inv, :, :].roll(shifts=1, dims=-1)

    y_pre = layer(x_pre, rt)
    y_post = layer(x_post, rt)

    expected = y_pre[:, perm_inv, :, :].roll(shifts=1, dims=-1)

    diff_cell0 = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    return diff_cell0 < tol, diff_cell0


def _verify_r2s_equivariance(seed: int = 0, tol: float = 1e-5):
    """Verify R2S equivariance. Output is scalar — only cell permutation, no
    feature shift.
    """
    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)
    layer = RegularToScalar(in_channels=2, out_channels=1)

    x_pre = torch.randn(1, 7, 2, 6)
    x_post = x_pre[:, perm_inv, :, :].roll(shifts=1, dims=-1)

    y_pre = layer(x_pre, rt)
    y_post = layer(x_post, rt)

    expected = y_pre[:, perm_inv, :]  # no feature shift (scalar output)

    diff_cell0 = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    return diff_cell0 < tol, diff_cell0


if __name__ == "__main__":
    print("=" * 60)
    print("C6-equivariant convolution: numerical equivariance test")
    print("=" * 60)

    print("\nTest setup: 7-cell hex region (1 center + 6 ring) with C6 rotation")
    print("as cyclic permutation of the 6 outer cells.")

    for seed in [0, 1, 2, 3, 42]:
        print(f"\n--- Seed {seed} ---")
        ok_s2r, diff_s2r = _verify_s2r_equivariance(seed)
        ok_r2r, diff_r2r = _verify_r2r_equivariance(seed)
        ok_r2s, diff_r2s = _verify_r2s_equivariance(seed)
        print(f"  S2R: max diff = {diff_s2r:.2e}  {'✓ EQUIVARIANT' if ok_s2r else '✗ FAIL'}")
        print(f"  R2R: max diff = {diff_r2r:.2e}  {'✓ EQUIVARIANT' if ok_r2r else '✗ FAIL'}")
        print(f"  R2S: max diff = {diff_r2s:.2e}  {'✓ EQUIVARIANT' if ok_r2s else '✗ FAIL'}")

    # Parameter counts
    print("\n--- Parameter counts ---")
    s2r = ScalarToRegular(1, 8)
    r2r = RegularToRegular(8, 8)
    r2s = RegularToScalar(8, 1)
    print(f"  S2R(1, 8):  {s2r.n_parameters()} params  (expected: 1*8*7 + 8 = 64)")
    print(f"  R2R(8, 8):  {r2r.n_parameters()} params  (expected: 8*8*42 + 8 = 2696)")
    print(f"  R2S(8, 1):  {r2s.n_parameters()} params  (expected: 8*1*7 + 1 = 57)")
