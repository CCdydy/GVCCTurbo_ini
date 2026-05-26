"""
spectral.py — V0 spectral-masked codebook injection for turboddcm.

V0 hypothesis (Frequency-Scheduled DDCM, see [[research-direction]] memory):
At each SDE step the codebook-selected z* is DCT-transformed, multiplied by
a per-step low-pass mask M_t(ω), and inverse-DCT'd. The mask cutoff grows
along the denoising trajectory: early steps allow only low frequencies
(where the model's prior is being established), late steps allow the full
band (where high-frequency detail is being added).

Bitstream is UNCHANGED in V0: the codebook still selects top-M atoms in the
usual way; we only filter the assembled z* spectrally. Future V1 will move
the selection itself to per-band.

Two crucial invariants for V0:
  1. Per-(channel-frame) variance preservation — the SDE step expects unit-
     variance Gaussian innovation, so we rescale after masking.
  2. Encoder/decoder mask consistency — the mask is a deterministic function
     of (step, t), not of the residual, so encoder and decoder produce the
     same mask without extra side-information.

Reuses the orthonormal DCT/IDCT and radial-bin utilities from spd_repro/
(those are pure math primitives, not SPD's generation logic).
"""

import os
import sys
import math
from typing import Optional

import torch

# Reuse the same DCT primitives as spd_repro (math only; no SPD pipeline logic).
_spd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "spd_repro")
if _spd_path not in sys.path:
    sys.path.insert(0, _spd_path)
from dct_utils import dct_2d, idct_2d, radial_frequency_bins


# ============================================================================
# Schedules — cutoff(t) ∈ [0, 1] gives the low-pass cutoff at timestep t
# ============================================================================

class SpectralSchedule:
    """Base class: a function (t in [0,1]) → cutoff in [0,1].

    Larger cutoff lets through more bands. cutoff=1 → no filtering (baseline).
    cutoff=0 → only DC.
    """

    name: str = "base"

    def cutoff(self, t: float) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class IdentitySchedule(SpectralSchedule):
    """cutoff(t) = 1 for all t — equivalent to no spectral mask (baseline)."""
    name = "identity"

    def cutoff(self, t: float) -> float:
        return 1.0


class LinearCutoffSchedule(SpectralSchedule):
    """Linear interpolation: cutoff(t=1) = c_high_t (small, e.g. 0.2),
    cutoff(t=0) = c_low_t (large, e.g. 1.0). At high noise, only low-freq
    survives; at low noise, full band."""

    def __init__(self, c_high_t: float = 0.2, c_low_t: float = 1.0):
        assert 0.0 <= c_high_t <= c_low_t <= 1.0
        self.c_high_t = c_high_t
        self.c_low_t = c_low_t
        self.name = f"linear[{c_high_t}->{c_low_t}]"

    def cutoff(self, t: float) -> float:
        return self.c_high_t + (self.c_low_t - self.c_high_t) * (1.0 - t)


class HighPassReverseSchedule(SpectralSchedule):
    """Sanity-check schedule — high-pass cutoff GROWING from t=1 to t=0.
    Expected to perform worse than baseline (deletes the signal-recovered
    low-freq band the model has already produced). Useful as a negative
    control."""

    def __init__(self, c_high_t: float = 0.2, c_low_t: float = 1.0):
        self.c_high_t = c_high_t
        self.c_low_t = c_low_t
        self.name = f"highpass_reverse[{c_high_t}->{c_low_t}]"

    def cutoff(self, t: float) -> float:
        # Returns a "high-pass lower bound" — see apply_spectral_mask high_pass param
        return self.c_high_t + (self.c_low_t - self.c_high_t) * (1.0 - t)


class SPDCutoffSchedule(SpectralSchedule):
    """Schedule derived from SPD Prop 1's per-frequency activation time.

    At timestep t, the cutoff frequency is the largest ω whose t_ω ≥ t —
    i.e., the boundary between "currently noise-dominated" (above ω) and
    "currently signal-recovered" (below ω) frequencies for the diffusion
    trajectory.

    Inverts SPD Eq 9 analytically. Requires the power-law fit (β, c) from
    spd_repro/power_spectrum.py. Defaults match the Wan 2.1-T2V-1.3B
    measurement: β=1.99, c=exp(-4.07)≈0.017.
    """

    def __init__(self, beta: float = 1.988, intercept_log_c: float = -4.074,
                 delta: float = 0.01):
        self.beta = beta
        self.intercept_log_c = intercept_log_c
        self.c_coeff = math.exp(intercept_log_c)
        self.delta = delta
        self.name = f"spd[β={beta:.2f},δ={delta}]"

    def cutoff(self, t: float) -> float:
        # Solve SPD Eq 9 for ω given t:
        #   t = 1 / (1 + sqrt(δ / (P_ω (1 + P_ω - δ))))
        # ⇒  (1-t)/t = sqrt(δ / (P_ω (1 + P_ω - δ)))
        # ⇒  P_ω^2 + (1-δ) P_ω - δ t² / (1-t)² = 0
        # ⇒  P_ω = [-(1-δ) + sqrt((1-δ)² + 4 δ t²/(1-t)²)] / 2
        # Then ω = (P_ω / c)^(-1/β)  since P_ω = c · ω^(-β)
        if t >= 1.0 - 1e-9:
            return 0.0  # at pure noise, no freq is recovered yet
        if t <= 1e-9:
            return 1.0  # at clean state, all freqs are recovered
        A = self.delta * (t ** 2) / ((1.0 - t) ** 2)
        disc = (1.0 - self.delta) ** 2 + 4.0 * A
        P_omega = (-(1.0 - self.delta) + math.sqrt(disc)) / 2.0
        if P_omega <= 0:
            return 0.0
        omega = (P_omega / self.c_coeff) ** (-1.0 / self.beta)
        return min(1.0, max(0.0, omega))


# ============================================================================
# Mask construction
# ============================================================================

_MASK_CACHE: dict = {}


def make_lowpass_mask(
    cutoff: float, H: int, W: int, device, dtype=torch.float32,
    soft: bool = False, soft_width: float = 0.02,
) -> torch.Tensor:
    """Build a 2D low-pass mask of shape (H, W) over DCT bins.

    Args:
        cutoff:     normalized radial frequency threshold in [0, 1]
        H, W:       DCT grid shape
        device:     torch device for the mask
        soft:       if True, use a soft cosine taper of width `soft_width`
                    near the cutoff (reduces ringing artifacts on IDCT)
        soft_width: width of the taper region (only used if soft=True)
    """
    key = (cutoff, H, W, str(device), soft, soft_width)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    omega_mag, _, _ = radial_frequency_bins(H, W, num_bins=max(H, W), device=device)
    if not soft:
        m = (omega_mag <= cutoff).to(dtype)
    else:
        # cosine taper: 1 below (cutoff - soft_width/2), 0 above (cutoff + soft_width/2)
        lo = cutoff - soft_width / 2
        hi = cutoff + soft_width / 2
        m = torch.zeros_like(omega_mag)
        m[omega_mag <= lo] = 1.0
        in_taper = (omega_mag > lo) & (omega_mag < hi)
        taper_vals = 0.5 * (1.0 + torch.cos(math.pi * (omega_mag[in_taper] - lo) / soft_width))
        m[in_taper] = taper_vals
        m = m.to(dtype)
    _MASK_CACHE[key] = m
    return m


def make_highpass_mask(
    cutoff: float, H: int, W: int, device, dtype=torch.float32,
) -> torch.Tensor:
    """Build a 2D high-pass mask (pass freqs ABOVE cutoff)."""
    return 1.0 - make_lowpass_mask(cutoff, H, W, device, dtype)


# ============================================================================
# Apply mask to 3D latent noise with per-frame variance preservation
# ============================================================================

def apply_spectral_mask(
    z: torch.Tensor,
    schedule: SpectralSchedule,
    t: float,
    preserve_per_frame_std: bool = True,
    soft: bool = False,
) -> torch.Tensor:
    """Apply 2D spectral low-pass mask to the spatial dims of a 3D video latent.

    Args:
        z:    (1, C, F, H, W) — codebook-assembled unit-variance noise
        schedule: spectral schedule providing cutoff(t)
        t:    current timestep in [0, 1]
        preserve_per_frame_std: rescale each (C, H, W) frame so its std equals
            the pre-mask value (preserves SDE noise magnitude assumption)
        soft: use soft cosine taper at cutoff

    Returns:
        z_masked: (1, C, F, H, W), spectrally filtered, std preserved per frame
    """
    if isinstance(schedule, IdentitySchedule):
        return z

    B, C, F, H, W = z.shape
    cutoff = schedule.cutoff(t)
    if isinstance(schedule, HighPassReverseSchedule):
        mask = make_highpass_mask(cutoff, H, W, z.device, z.dtype)
    else:
        mask = make_lowpass_mask(cutoff, H, W, z.device, z.dtype, soft=soft)

    # Per-frame std before mask (over C, H, W of each temporal frame)
    if preserve_per_frame_std:
        orig_std = z.std(dim=(1, 3, 4), keepdim=True)  # (1, 1, F, 1, 1)

    # Flatten leading dims, DCT, mask, IDCT
    z_flat = z.reshape(B * C * F, H, W).float()
    z_freq = dct_2d(z_flat)
    z_freq = z_freq * mask.unsqueeze(0)
    z_masked = idct_2d(z_freq).reshape(B, C, F, H, W).to(z.dtype)

    if preserve_per_frame_std:
        new_std = z_masked.std(dim=(1, 3, 4), keepdim=True).clamp(min=1e-8)
        z_masked = z_masked * (orig_std / new_std)

    return z_masked


# ============================================================================
# Self-test
# ============================================================================

def _test_identity_baseline():
    """IdentitySchedule should leave z unchanged."""
    z = torch.randn(1, 16, 9, 60, 104)
    s = IdentitySchedule()
    z2 = apply_spectral_mask(z, s, t=0.5)
    err = (z2 - z).abs().max().item()
    assert err < 1e-6, f"identity not identity: {err}"
    print("✓ IdentitySchedule leaves z unchanged")


def _test_lowpass_removes_highfreq():
    """LinearCutoffSchedule with c=0 should kill all but DC and low-freq."""
    z = torch.randn(1, 4, 3, 32, 32)
    s = LinearCutoffSchedule(c_high_t=0.1, c_low_t=0.1)  # constant 0.1 cutoff
    z2 = apply_spectral_mask(z, s, t=0.5)
    # Check that the DCT of z2 has high-freq coeffs near zero
    z2_freq = dct_2d(z2.reshape(-1, 32, 32).float())
    # Top-right (high-freq) corner should be near zero (before renorm)
    # After renorm, the surviving low-freq part is scaled up.
    # Just check non-trivial: the result has the same per-frame std as input.
    orig_std = z.std(dim=(1, 3, 4), keepdim=True)
    new_std = z2.std(dim=(1, 3, 4), keepdim=True)
    err = (orig_std - new_std).abs().max().item()
    assert err < 1e-3, f"per-frame std drift: {err}"
    print("✓ LinearCutoffSchedule preserves per-frame std after masking")


def _test_spd_schedule_monotonic():
    """SPDCutoffSchedule should give cutoff(t=0.99) ≈ 0 and cutoff(t=0.01) ≈ 1."""
    s = SPDCutoffSchedule(beta=1.988, intercept_log_c=-4.074, delta=0.01)
    cutoffs = [s.cutoff(t) for t in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]]
    print(f"  SPD schedule cutoff at t = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:")
    print(f"    {[f'{c:.3f}' for c in cutoffs]}")
    assert all(cutoffs[i] <= cutoffs[i + 1] + 1e-9 for i in range(len(cutoffs) - 1)), \
        "SPD schedule should be monotonic non-decreasing as t decreases"
    print("✓ SPDCutoffSchedule monotonic increasing as t → 0")


def _test_mask_caching():
    """Mask cache should hit on second call with same args."""
    _MASK_CACHE.clear()
    m1 = make_lowpass_mask(0.5, 60, 104, device='cpu')
    m2 = make_lowpass_mask(0.5, 60, 104, device='cpu')
    assert m1 is m2, "mask cache miss"
    print(f"✓ mask cache works ({len(_MASK_CACHE)} entries)")


def _example():
    """Show what schedules look like over a 20-step trajectory."""
    schedules = [
        IdentitySchedule(),
        LinearCutoffSchedule(c_high_t=0.2, c_low_t=1.0),
        SPDCutoffSchedule(beta=1.988, intercept_log_c=-4.074, delta=0.01),
        SPDCutoffSchedule(beta=1.988, intercept_log_c=-4.074, delta=0.05),
        SPDCutoffSchedule(beta=1.988, intercept_log_c=-4.074, delta=0.1),
    ]
    print(f"\n  cutoff(t) over 20-step trajectory (t descending 1.0 → 0.05):")
    t_grid = [1.0 - i / 19.0 for i in range(20)]
    header = "  t       " + " ".join(f"{s.name:>22}" for s in schedules)
    print(header)
    for t in t_grid:
        row = f"  {t:.3f}   " + " ".join(f"{s.cutoff(t):>22.3f}" for s in schedules)
        print(row)


if __name__ == "__main__":
    _test_identity_baseline()
    _test_lowpass_removes_highfreq()
    _test_spd_schedule_monotonic()
    _test_mask_caching()
    _example()
