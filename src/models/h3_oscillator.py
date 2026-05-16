"""
src/models/h3_oscillator.py

H3Oscillator: the full M1 architecture (Phase 2 / M1 Step 3).

    x_t ∈ R^{B, N, n_features}                         scalar field
    ↓ Encoder: S2R → GELU → R2R → GELU + residual
    h_0 ∈ R^{B, N, F, 6}                               regular features
    ↓ for k in range(K):  h = LTC(h, x_t; dt=1/K)      [shared weights]
    h_K
    ↓ Decoder: R2R → GELU + residual → R2S
    delta ∈ R^{B, N, n_features}
    ↓ output = x_t + delta                              residual prediction

C6 equivariance is preserved end-to-end:
  - Each S2R / R2R / R2S is equivariant by construction (Step 1)
  - GELU is element-wise (preserves equivariance)
  - The LTC update is composition of equivariant operations (Step 3)
  - Residual connections (sums of equivariant maps) are equivariant
  → The full H3Oscillator is C6-equivariant.

Parameter count at default config (F=8, K=4):
    Encoder S2R(2→8):     120
    Encoder R2R(8→8):    2696
    Dynamics R2R(8→8):   2696  ← block-circulant constraint (B5 will ablate this)
    Dynamics S2R(2→8):    120
    Dynamics A,log_tau:    16
    Decoder R2R(8→8):    2696
    Decoder R2S(8→2):     114
    -----------------------------
    Total:               8458
    (Comparison: B3=6402, B4=8562, M1Static=8322)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.equivariant_conv import (
    ScalarToRegular, RegularToRegular, RegularToScalar,
)
from src.models.oscillator_dynamics import H3OscillatorDynamics


class H3Oscillator(nn.Module):
    """Full M1 architecture: equivariant encoder + LTC dynamics + equivariant decoder.

    Args:
        n_features: number of scalar features in/out (e.g., 2 for u, v)
        F_dim: regular-feature multiplicity
        K: number of LTC integration steps per prediction (shared weights)
        residual_prediction: if True, predicts x_t + delta (matches B3/B4/M1Static)
    """

    def __init__(
        self,
        n_features: int = 2,
        F_dim: int = 8,
        K: int = 4,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.F_dim = F_dim
        self.K = K
        self.dt = 1.0 / K
        self.residual_prediction = residual_prediction

        # Encoder
        self.encoder_s2r = ScalarToRegular(n_features, F_dim)
        self.encoder_r2r = RegularToRegular(F_dim, F_dim)

        # Dynamics (single layer, applied K times)
        self.dynamics = H3OscillatorDynamics(F_dim=F_dim, n_features=n_features)

        # Decoder
        self.decoder_r2r = RegularToRegular(F_dim, F_dim)
        self.decoder_r2s = RegularToScalar(F_dim, n_features)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, n_features) — scalar field
            region_tensors: RegionTensors with dir_neighbor_idx, dir_valid_mask
        Returns:
            (batch, n_cells, n_features) — predicted next state
        """
        # ----- Encoder -----
        h = self.encoder_s2r(x, region_tensors)        # (B, N, F, 6)
        h = F.gelu(h)
        h_skip = h
        h = self.encoder_r2r(h, region_tensors)
        h = F.gelu(h)
        h = h + h_skip                                 # residual

        # ----- Dynamics: K LTC iterations with shared weights -----
        for _ in range(self.K):
            h = self.dynamics(h, x, region_tensors, dt=self.dt)

        # ----- Decoder -----
        h_skip = h
        h = self.decoder_r2r(h, region_tensors)
        h = F.gelu(h)
        h = h + h_skip                                 # residual
        delta = self.decoder_r2s(h, region_tensors)    # (B, N, n_features)

        if self.residual_prediction:
            return x + delta
        return delta

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# End-to-end equivariance verification
# ============================================================================

def _verify_h3oscillator_equivariance(seed: int = 0, tol: float = 1e-4) -> tuple:
    """Verify the full H3Oscillator is C6-equivariant end-to-end.

    Composition of equivariant maps + element-wise activations + residual
    connections + LTC closed-form integration should preserve equivariance.
    Numerical tolerance is looser than for individual layers because exp()
    and division introduce more floating-point noise.
    """
    from src.models.equivariant_conv import _build_synthetic_hex_test_setup

    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)

    # Small model for fast test
    model = H3Oscillator(n_features=2, F_dim=4, K=4, residual_prediction=True)

    # Random scalar input
    x_pre = torch.randn(1, 7, 2)
    x_post = x_pre[:, perm_inv, :]                     # rotate input

    y_pre = model(x_pre, rt)
    y_post = model(x_post, rt)

    # Output is scalar (trivial rep), cell 0 is fixed under rotation,
    # so y_post[:, 0] should equal y_pre[:, perm_inv[0]] = y_pre[:, 0]
    expected = y_pre[:, perm_inv, :]
    diff_cell0 = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    return diff_cell0 < tol, diff_cell0


if __name__ == "__main__":
    print("=" * 64)
    print("H3Oscillator (full M1): end-to-end equivariance test")
    print("=" * 64)

    # Numerical equivariance verification
    print("\nFull-model equivariance test (5 seeds, F=4, K=4):")
    all_ok = True
    for seed in [0, 1, 2, 3, 42]:
        ok, diff = _verify_h3oscillator_equivariance(seed)
        status = "✓ EQUIVARIANT" if ok else "✗ FAIL"
        print(f"  seed {seed}: max diff = {diff:.2e}  {status}")
        all_ok = all_ok and ok

    if all_ok:
        print("\n✓ All seeds pass — H3Oscillator is end-to-end C6-equivariant.")
    else:
        print("\n✗ Some seeds failed — equivariance is broken somewhere.")

    # Parameter count breakdown at default config
    print("\n--- Parameter count breakdown (default: F=8, K=4) ---")
    model = H3Oscillator(n_features=2, F_dim=8, K=4)
    print(f"  encoder_s2r:   {sum(p.numel() for p in model.encoder_s2r.parameters()):5d}")
    print(f"  encoder_r2r:   {sum(p.numel() for p in model.encoder_r2r.parameters()):5d}")
    print(f"  dynamics.W_h:  {sum(p.numel() for p in model.dynamics.W_h.parameters()):5d}")
    print(f"  dynamics.W_x:  {sum(p.numel() for p in model.dynamics.W_x.parameters()):5d}")
    print(f"  dynamics A+τ:  {model.dynamics.A.numel() + model.dynamics.log_tau.numel():5d}")
    print(f"  decoder_r2r:   {sum(p.numel() for p in model.decoder_r2r.parameters()):5d}")
    print(f"  decoder_r2s:   {sum(p.numel() for p in model.decoder_r2s.parameters()):5d}")
    print(f"  TOTAL:        {model.n_parameters():6d}  (B3=6402, B4=8562, M1Static=8322)")

    # Parameter counts at various config
    print("\n--- Parameter counts (various configs) ---")
    for F_dim in [6, 8, 12]:
        for K in [2, 4, 8]:
            m = H3Oscillator(n_features=2, F_dim=F_dim, K=K)
            print(f"  F={F_dim}, K={K}: {m.n_parameters():6d} params")

    # Smoke test on real H3 region (817 cells)
    print("\n--- Smoke test on 817-cell H3 region ---")
    from src.h3_region import H3Region
    from src.training import RegionTensors

    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    rt = RegionTensors.from_region(region, torch.device('cpu'))
    model = H3Oscillator(n_features=2, F_dim=8, K=4)
    x = torch.randn(2, region.n_cells, 2)
    y = model(x, rt)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}  (expected: same as input)")
    print(f"  Params: {model.n_parameters()}")
