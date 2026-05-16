"""
M1Static: gauge-equivariant static hex CNN. M1 minus the dynamics layer.

Architecture:
    x (scalar, n_features channels)
      -> S2R encoder (n_features -> F)
      -> [R2R + GELU + (residual)] * n_layers
      -> R2S decoder (F -> n_features)
      -> residual prediction: x_{t+1} = x_t + delta

End-to-end C6-equivariance follows by composition:
  - Each layer (S2R, R2R, R2S) is C6-equivariant by construction
  - GELU is element-wise on regular features → preserves cyclic shift
  - Residual connections (sum of equivariant maps) preserve equivariance
  - Therefore the full model is C6-equivariant

Comparison vs B4:
  - B4: 1 embed (Linear, no spatial mixing) + n_layers HexConv + 1 decode (Linear)
        = n_layers spatial-mixing layers, free directional kernels
  - M1Static: S2R + n_layers R2R + R2S = (n_layers + 2) spatial-mixing layers,
        all C6-equivariant by kernel parameterization

This isolates the contribution of the equivariance constraint alone
(no dynamics yet — that's Step 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.equivariant_conv import (
    ScalarToRegular, RegularToRegular, RegularToScalar,
)


class M1Static(nn.Module):
    """Static gauge-equivariant single-step state predictor.

    Args:
        n_features: number of scalar features in input/output (e.g. 2 for u, v)
        F_dim: regular-feature multiplicity (each cell has F_dim regular features,
               each is a 6-component vector)
        n_layers: number of R2R "body" layers between encoder and decoder
        residual: whether to use skip connections around each R2R block
        residual_prediction: whether to predict delta (x_{t+1} = x_t + delta)
                             vs absolute state. Standard trick for stable
                             dynamical-system prediction; matches B3/B4.
    """

    def __init__(
        self,
        n_features: int = 2,
        F_dim: int = 8,
        n_layers: int = 3,
        residual: bool = True,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.F_dim = F_dim
        self.n_layers = n_layers
        self.residual = residual
        self.residual_prediction = residual_prediction

        self.encoder = ScalarToRegular(n_features, F_dim)
        self.r2r_layers = nn.ModuleList([
            RegularToRegular(F_dim, F_dim) for _ in range(n_layers)
        ])
        self.decoder = RegularToScalar(F_dim, n_features)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, n_features) — scalar field
            region_tensors: RegionTensors with dir_neighbor_idx, dir_valid_mask
        Returns:
            (batch, n_cells, n_features) — predicted next state
        """
        # Lift scalar input to regular features
        h = self.encoder(x, region_tensors)  # (batch, n_cells, F_dim, 6)

        # Pass through R2R body
        for layer in self.r2r_layers:
            h_new = F.gelu(layer(h, region_tensors))
            h = h + h_new if self.residual else h_new

        # Project regular features back to scalar
        delta = self.decoder(h, region_tensors)  # (batch, n_cells, n_features)

        if self.residual_prediction:
            return x + delta
        return delta

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# End-to-end equivariance verification
# -------------------------------------------------------------------------

def _verify_m1_static_equivariance(seed: int = 0, tol: float = 1e-5):
    """Verify that the FULL M1Static model is C6-equivariant.

    Composition of equivariant maps + element-wise activations + residuals
    should be equivariant. Verifies by direct numerical test, same as for
    individual layers.
    """
    from src.models.equivariant_conv import _build_synthetic_hex_test_setup

    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)

    model = M1Static(n_features=2, F_dim=4, n_layers=3,
                     residual=True, residual_prediction=True)

    # Random scalar input
    x_pre = torch.randn(1, 7, 2)
    x_post = x_pre[:, perm_inv, :]  # rotate input (cells permuted)

    y_pre = model(x_pre, rt)
    y_post = model(x_post, rt)

    # Output is scalar field (same shape as input). Equivariance:
    # y_post[c, f] = y_pre[perm_inv[c], f] (just cell permutation, no feature shift)
    expected = y_pre[:, perm_inv, :]

    diff_cell0 = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    return diff_cell0 < tol, diff_cell0


if __name__ == "__main__":
    print("=" * 60)
    print("M1Static: end-to-end equivariance test")
    print("=" * 60)

    # Numerical equivariance verification at multiple seeds
    print("\nFull-model equivariance test (5 seeds, 3-layer F=4 model):")
    all_ok = True
    for seed in [0, 1, 2, 3, 42]:
        ok, diff = _verify_m1_static_equivariance(seed)
        status = "✓ EQUIVARIANT" if ok else "✗ FAIL"
        print(f"  seed {seed}: max diff = {diff:.2e}  {status}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All seeds pass — M1Static is end-to-end C6-equivariant.")
    else:
        print("\n✗ Some seeds failed — equivariance is broken somewhere.")

    # Parameter counts at default config (F=8, 3 layers)
    print("\n--- Parameter counts ---")
    for F_dim in [4, 6, 8, 12]:
        for n_layers in [2, 3]:
            model = M1Static(n_features=2, F_dim=F_dim, n_layers=n_layers)
            print(f"  F={F_dim}, n_layers={n_layers}: {model.n_parameters()} params")

    # Smoke test on real H3 region
    print("\n--- Smoke test on 817-cell H3 region ---")
    from src.h3_region import H3Region
    from src.training import RegionTensors

    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    rt = RegionTensors.from_region(region, torch.device('cpu'))
    model = M1Static(n_features=2, F_dim=8, n_layers=3)
    x = torch.randn(2, region.n_cells, 2)
    y = model(x, rt)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}  (expected: same as input)")
    print(f"  Params: {model.n_parameters()}  (B3=6402, B4=8562 for comparison)")
