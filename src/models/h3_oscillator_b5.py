"""
src/models/h3_oscillator_b5.py

B5: M1 architecture with the C6 block-circulant constraint REMOVED from the
dynamics layer's recurrent state-to-state operator W_h.

This is the Phase 2 / M1 Step 5 architectural Acid Test:

    M1 dynamics: W_h = RegularToRegular (C6-equivariant by construction)
    B5 dynamics: W_h = UnconstrainedRegularToRegular (no C6 constraint)

Everything else is identical between M1 and B5:
    - Same encoder (S2R + R2R, equivariant)
    - Same decoder (R2R + R2S, equivariant)
    - Same W_x in dynamics (S2R, equivariant)
    - Same A, log_τ scalars
    - Same closed-form CfC update
    - Same K=4 iteration count
    - Same training script, optimizer, dataset, evaluation

Parameter accounting (default F=8):
    M1 dynamics W_h (R2R, equivariant):       2,696 params
    B5 dynamics W_h (R2R_U, unconstrained):  16,136 params  (5.99× more)
    M1 total:    8,458 params
    B5 total:   21,898 params  (2.59× more)

The asymmetry is deliberate: removing a constraint inherently adds free
parameters. If B5 with 2.6× more capacity still doesn't beat M1, the
constraint is doing work that capacity alone can't replace — a strong
positive result for the C6 prior. If B5 beats M1, the result is harder to
interpret (could be capacity, could be wrong constraint).

End-to-end equivariance: should FAIL by construction (verify below). The
dynamics layer breaks rotation symmetry through its unconstrained W_h, even
though the encoder/decoder/W_x remain equivariant.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.equivariant_conv import (
    ScalarToRegular, RegularToRegular, RegularToScalar,
)
from src.models.unconstrained_conv import UnconstrainedRegularToRegular


# ============================================================================
# B5 Dynamics layer
# ============================================================================

class H3OscillatorDynamicsB5(nn.Module):
    """B5's LTC dynamics: same as H3OscillatorDynamics but with W_h unconstrained.

    Only difference from H3OscillatorDynamics:
        self.W_h = UnconstrainedRegularToRegular(...)   # was RegularToRegular

    Everything else identical: W_x is S2R (equivariant), A and log_tau are
    per-channel scalars broadcast across 6 dirs, closed-form CfC update.
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

        # *** THE ONLY DIFFERENCE FROM M1 ***
        # Recurrent operator W_h: unconstrained (no C6 block-circulant prior)
        self.W_h = UnconstrainedRegularToRegular(F_dim, F_dim)

        # Everything below is identical to H3OscillatorDynamics (M1):
        self.W_x = ScalarToRegular(n_features, F_dim)
        self.A = nn.Parameter(torch.zeros(F_dim))
        self.log_tau = nn.Parameter(torch.full((F_dim,), log_tau_init))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        region_tensors,
        dt: float = 0.25,
    ) -> torch.Tensor:
        """Same forward signature as H3OscillatorDynamics."""
        Wh_h = self.W_h(h, region_tensors)
        Wx_x = self.W_x(x, region_tensors)
        g = torch.tanh(Wh_h + Wx_x)

        inv_tau = torch.exp(-self.log_tau)
        inv_tau = inv_tau[None, None, :, None]
        A = self.A[None, None, :, None]

        decay = torch.exp(-dt * (inv_tau + g))
        h_next = (h - A) * decay + A
        return h_next

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# B5 full model
# ============================================================================

class H3OscillatorB5(nn.Module):
    """Full B5 architecture: M1 with the C6 constraint removed from dynamics W_h only.

    Encoder, decoder, and W_x in dynamics remain C6-equivariant.
    Only W_h (the state-to-state operator in dynamics) is unconstrained.
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

        # Equivariant encoder (same as M1)
        self.encoder_s2r = ScalarToRegular(n_features, F_dim)
        self.encoder_r2r = RegularToRegular(F_dim, F_dim)

        # B5 dynamics: unconstrained W_h, everything else equivariant
        self.dynamics = H3OscillatorDynamicsB5(F_dim=F_dim, n_features=n_features)

        # Equivariant decoder (same as M1)
        self.decoder_r2r = RegularToRegular(F_dim, F_dim)
        self.decoder_r2s = RegularToScalar(F_dim, n_features)

    def forward(self, x: torch.Tensor, region_tensors) -> torch.Tensor:
        """Same signature as H3Oscillator."""
        h = self.encoder_s2r(x, region_tensors)
        h = F.gelu(h)
        h_skip = h
        h = self.encoder_r2r(h, region_tensors)
        h = F.gelu(h)
        h = h + h_skip

        for _ in range(self.K):
            h = self.dynamics(h, x, region_tensors, dt=self.dt)

        h_skip = h
        h = self.decoder_r2r(h, region_tensors)
        h = F.gelu(h)
        h = h + h_skip
        delta = self.decoder_r2s(h, region_tensors)

        if self.residual_prediction:
            return x + delta
        return delta

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Verification: B5 should be NOT equivariant (the constraint is removed)
# ============================================================================

def _verify_b5_not_equivariant(seed: int = 0, tol: float = 1e-4) -> tuple:
    """Confirm that H3OscillatorB5 is NOT C6-equivariant.

    M1 was end-to-end equivariant (Step 3 verification). B5 should fail
    equivariance: that's the ablation working. If this test 'passes'
    (equivariance error tiny), something is wrong with the ablation.
    """
    from src.models.equivariant_conv import _build_synthetic_hex_test_setup

    rt, perm_inv = _build_synthetic_hex_test_setup(seed)
    torch.manual_seed(seed)

    model = H3OscillatorB5(n_features=2, F_dim=4, K=4, residual_prediction=True)

    x_pre = torch.randn(1, 7, 2)
    x_post = x_pre[:, perm_inv, :]

    y_pre = model(x_pre, rt)
    y_post = model(x_post, rt)

    expected = y_pre[:, perm_inv, :]
    diff = (y_post[:, 0] - expected[:, 0]).abs().max().item()
    # NOTE: returns True if the layer is BROKEN-equivariant (i.e. ablation worked)
    return diff > tol, diff


if __name__ == "__main__":
    print("=" * 64)
    print("H3OscillatorB5: NON-equivariance test (the ablation working)")
    print("=" * 64)

    print("\nFull-model equivariance test (5 seeds, F=4, K=4):")
    print("Expected: equivariance FAILS (large error) because dynamics W_h is unconstrained")
    all_ok = True
    for seed in [0, 1, 2, 3, 42]:
        broken, diff = _verify_b5_not_equivariant(seed)
        status = "✓ NOT EQUIVARIANT (ablation works)" if broken else "✗ UNEXPECTEDLY EQUIVARIANT"
        print(f"  seed {seed}: equivariance error = {diff:.2e}  {status}")
        all_ok = all_ok and broken

    if all_ok:
        print("\n✓ All seeds confirm: B5 breaks C6 equivariance — ablation is genuine.")
    else:
        print("\n✗ Some seed unexpectedly preserved equivariance — check implementation.")

    # Parameter count breakdown
    print("\n--- Parameter count breakdown (default: F=8, K=4) ---")
    model = H3OscillatorB5(n_features=2, F_dim=8, K=4)
    print(f"  encoder_s2r:        {sum(p.numel() for p in model.encoder_s2r.parameters()):6d}")
    print(f"  encoder_r2r:        {sum(p.numel() for p in model.encoder_r2r.parameters()):6d}")
    print(f"  dynamics.W_h (R2R_U): {sum(p.numel() for p in model.dynamics.W_h.parameters()):6d}  ← unconstrained")
    print(f"  dynamics.W_x:       {sum(p.numel() for p in model.dynamics.W_x.parameters()):6d}")
    print(f"  dynamics A+log_τ:    {model.dynamics.A.numel() + model.dynamics.log_tau.numel():6d}")
    print(f"  decoder_r2r:        {sum(p.numel() for p in model.decoder_r2r.parameters()):6d}")
    print(f"  decoder_r2s:        {sum(p.numel() for p in model.decoder_r2s.parameters()):6d}")
    print(f"  TOTAL B5:           {model.n_parameters():6d}  (M1: 8458)")

    # Smoke test on real 817-cell region
    print("\n--- Smoke test on 817-cell H3 region ---")
    from src.h3_region import H3Region
    from src.training import RegionTensors

    region = H3Region(center_lat=45.0, center_lon=0.0, resolution=5, k_ring=16)
    rt = RegionTensors.from_region(region, torch.device('cpu'))
    model = H3OscillatorB5(n_features=2, F_dim=8, K=4)
    x = torch.randn(2, region.n_cells, 2)
    y = model(x, rt)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}  (expected: same as input)")
    print(f"  Params: {model.n_parameters()}")
