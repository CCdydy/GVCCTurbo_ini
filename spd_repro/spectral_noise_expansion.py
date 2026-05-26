"""
spectral_noise_expansion.py — SPD Sec 4.1 spectral noise expansion.

Algorithm (3 steps per SPD Sec 4.1):

Given a low-resolution flow-matching state x^{s_i}_{t_i} at timestep t_i,
spatial shape (s_i H, s_i W), expand to scale s_{i+1} > s_i:

  (i)   ξ^{s_i}_{t_i}     = T_Φ ( x^{s_i}_{t_i} )            (DCT)
  (ii)  ξ^{s_{i+1}}_{t_i} = embed ξ^{s_i} in lower-freq part of (s_{i+1}H, s_{i+1}W) grid
                            fill Ω_{s_{i+1}} \ Ω_{s_i}  with  t_i · N(0,1)
  (iii) x^{s_{i+1}}_{t_i} = T_Φ^{-1}( ξ^{s_{i+1}}_{t_i} )    (IDCT)

The orthonormal DCT identifies spectral coefficients at the same bin index
across resolutions with the same physical frequency, so the embedding is a
direct copy (no per-bin rescaling). The global resolution-ratio scaling
needed to recover the next-stage flow-matching distribution is applied
afterward via timestep alignment (see `timestep_alignment.py`, Eqs 5–6),
which multiplies the result by κ_i = r / (1 + (r-1) t_i).

This module implements step (i)–(iii) only. Use together with
timestep_alignment.align(t_i, r) to get the final state and re-indexed
timestep.
"""

import os
import sys
from typing import Optional

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from dct_utils import dct_2d, idct_2d


def spectral_noise_expand(
    x_lo: torch.Tensor,
    target_h: int,
    target_w: int,
    t_i: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """SPD Sec 4.1 steps (i)-(iii): expand a low-res state to higher resolution.

    Operates on the last two dimensions. Leading dims (channel, frame, batch)
    are preserved.

    Args:
        x_lo:     (..., H_lo, W_lo) — low-res state at timestep t_i
        target_h: target H (must satisfy target_h >= H_lo)
        target_w: target W (must satisfy target_w >= W_lo)
        t_i:      current timestep in [0, 1] — noise level used to fill new freqs
        generator: torch.Generator for reproducible noise (optional)

    Returns:
        x_hi: (..., target_h, target_w) — expanded state at the SAME timestep t_i,
              before timestep alignment.  Call timestep_alignment.align(t_i, r) to
              get \tilde{t}_i and κ_i, then multiply this output by κ_i.
    """
    *prefix, H_lo, W_lo = x_lo.shape
    assert target_h >= H_lo and target_w >= W_lo, \
        f"spectral expansion must grow resolution; got {H_lo}x{W_lo} -> {target_h}x{target_w}"

    if target_h == H_lo and target_w == W_lo:
        return x_lo.clone()  # nothing to do

    # Step (i): forward DCT-II on the last two dims
    xi_lo = dct_2d(x_lo)  # (..., H_lo, W_lo)

    # Step (ii): build a (target_h, target_w) zero-init spectrum, copy ξ_lo into
    # the upper-left block (lower freqs at the same bin indices across scales
    # under the orthonormal identification used by SPD).
    xi_hi = x_lo.new_zeros(*prefix, target_h, target_w)
    xi_hi[..., :H_lo, :W_lo] = xi_lo

    # Fill Ω_{s_{i+1}} \ Ω_{s_i} with t_i * N(0, 1). The complementary mask
    # is the union of the right strip and bottom strip of the (target_h, target_w) grid.
    if t_i > 0:
        if generator is not None:
            noise = torch.randn(
                *xi_hi.shape, generator=generator,
                device=xi_hi.device, dtype=xi_hi.dtype,
            )
        else:
            noise = torch.randn_like(xi_hi)
        # right strip (columns W_lo:target_w, rows 0:target_h)
        xi_hi[..., :, W_lo:] = noise[..., :, W_lo:] * t_i
        # bottom strip (rows H_lo:target_h, columns 0:W_lo)  — non-overlapping with right
        xi_hi[..., H_lo:, :W_lo] = noise[..., H_lo:, :W_lo] * t_i

    # Step (iii): inverse DCT-II back to spatial domain at the new resolution
    x_hi = idct_2d(xi_hi)
    return x_hi


# ============================================================================
# Convenience: combined expansion + timestep alignment (one call)
# ============================================================================

def spectral_noise_expand_aligned(
    x_lo: torch.Tensor,
    target_h: int,
    target_w: int,
    t_i: float,
    scale_ratio: float,
    generator: Optional[torch.Generator] = None,
):
    """Run spectral_noise_expand + timestep alignment (Eqs 5-6) in one call.

    Returns:
        x_hi_aligned: (..., target_h, target_w) — flow-matching state at the
                      new resolution, ready to continue denoising at \tilde{t}_i
        t_aligned:    \tilde{t}_i to use as the model's input timestep
    """
    from timestep_alignment import align
    x_hi = spectral_noise_expand(x_lo, target_h, target_w, t_i, generator=generator)
    t_aligned, kappa = align(t_i, scale_ratio)
    return x_hi * kappa, t_aligned


# ============================================================================
# Self-test
# ============================================================================

def _test_identity_when_no_growth():
    """If target_h == H_lo and target_w == W_lo, expansion is the identity."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 24)
    y = spectral_noise_expand(x, 16, 24, t_i=0.5)
    err = (y - x).abs().max().item()
    assert err < 1e-5, f"identity grow failed: max err {err}"
    print("✓ identity when target shape == source")


def _test_low_freq_preserved():
    """After expansion, the low-frequency DCT coefficients of the new state
    should match the original low-res DCT coefficients (the SPD copy step).
    """
    torch.manual_seed(1)
    x_lo = torch.randn(1, 16, 24).double()
    target_h, target_w = 32, 48
    # With t_i = 0, the high-freq slots are zero, so the IDCT-then-DCT should
    # recover exactly the embedded coefficients.
    x_hi = spectral_noise_expand(x_lo, target_h, target_w, t_i=0.0)
    xi_hi = dct_2d(x_hi)
    err_low = (xi_hi[..., :16, :24] - dct_2d(x_lo)).abs().max().item()
    print(f"  low-freq coeffs after expand (t=0) max err vs source DCT: {err_low:.3e}")
    assert err_low < 1e-9, f"low-freq coeffs not preserved: {err_low}"
    # High-freq slots should be zero
    high_strip_R = xi_hi[..., :, 24:].abs().max().item()
    high_strip_B = xi_hi[..., 16:, :24].abs().max().item()
    print(f"  new high-freq strips (t=0) magnitudes: right={high_strip_R:.3e}, bottom={high_strip_B:.3e}")
    assert high_strip_R < 1e-9 and high_strip_B < 1e-9
    print("✓ low-frequency coefficients preserved exactly when t_i = 0")


def _test_noise_variance_at_high_freq():
    """For t_i > 0, the new high-freq DCT coefficients should have variance ≈ t_i^2."""
    torch.manual_seed(2)
    x_lo = torch.zeros(8, 32, 32).double()  # zero low-res so only noise is non-zero
    target_h, target_w = 64, 64
    t_i = 0.7
    x_hi = spectral_noise_expand(x_lo, target_h, target_w, t_i=t_i)
    xi_hi = dct_2d(x_hi)
    # New high-freq slots: union of right + bottom strips
    right = xi_hi[..., :, 32:].reshape(-1)
    bottom = xi_hi[..., 32:, :32].reshape(-1)
    new_coeffs = torch.cat([right, bottom])
    var = new_coeffs.var().item()
    print(f"  high-freq noise variance: {var:.4f}  (expected ≈ t_i^2 = {t_i**2:.4f})")
    assert abs(var - t_i**2) / (t_i**2) < 0.10, "variance too far from t_i^2"
    print("✓ high-freq noise has variance ≈ t_i^2")


def _test_timestep_alignment_via_combined():
    """spectral_noise_expand_aligned should multiply by κ and return aligned t."""
    torch.manual_seed(3)
    x_lo = torch.randn(1, 4, 16, 24)
    r = 2.0
    t_i = 0.5

    from timestep_alignment import align
    expected_t, expected_kappa = align(t_i, r)

    x_unaligned = spectral_noise_expand(x_lo, 32, 48, t_i=t_i,
                                         generator=torch.Generator().manual_seed(99))
    x_aligned, t_aligned = spectral_noise_expand_aligned(
        x_lo, 32, 48, t_i=t_i, scale_ratio=r,
        generator=torch.Generator().manual_seed(99),
    )
    err = (x_aligned - x_unaligned * expected_kappa).abs().max().item()
    assert abs(t_aligned - expected_t) < 1e-9
    assert err < 1e-5, f"alignment scaling mismatch: {err}"
    print(f"✓ combined call returns t_aligned={t_aligned:.4f}, kappa={expected_kappa:.4f}")


if __name__ == "__main__":
    _test_identity_when_no_growth()
    _test_low_freq_preserved()
    _test_noise_variance_at_high_freq()
    _test_timestep_alignment_via_combined()
    print("\nAll spectral noise expansion tests passed.")
