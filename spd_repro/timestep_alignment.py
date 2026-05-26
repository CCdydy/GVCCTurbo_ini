"""
timestep_alignment.py — SPD Sec 4.1 timestep alignment.

After spectral noise expansion from scale s_i to scale s_{i+1},
the enlarged state x^{s_{i+1}}_{t_i} has a higher overall noise
magnitude than the pretrained model expects at timestep t_i. We
rescale the state and re-index the timestep so the result is a
valid flow-matching state at the new resolution.

Per SPD Eqs 5–6 (Sec 4.1):

    \tilde{t}_i = (r * t_i) / (1 + (r - 1) * t_i)              [Eq 6]
    \tilde{x}^{s_{i+1}}_{\tilde{t}_i} = kappa_i * x^{s_{i+1}}_{t_i}   [Eq 5]
    kappa_i = (s_{i+1}/s_i) / (1 + ((s_{i+1}/s_i) - 1) * t_i)

where r = s_{i+1} / s_i is the per-scale resolution ratio.

NB: this formula is mathematically identical to the SD3 shift_t
reparameterisation, which is suggestive — SPD's progressive
resolution growth is in a sense an SD3-shifted timestep family.
"""

import math
from typing import Tuple


def aligned_timestep(t_i: float, scale_ratio: float) -> float:
    """SPD Eq 6: rescale a timestep after resolution expansion.

    Args:
        t_i: current timestep (in [0, 1])
        scale_ratio: r = s_{i+1} / s_i (>= 1, since SPD expands resolution)

    Returns:
        \tilde{t}_i: aligned timestep at the new resolution
    """
    return (scale_ratio * t_i) / (1.0 + (scale_ratio - 1.0) * t_i)


def state_scale_factor(t_i: float, scale_ratio: float) -> float:
    """SPD Eq 5: state rescaling factor kappa_i.

    Args:
        t_i: current timestep (in [0, 1])
        scale_ratio: r = s_{i+1} / s_i

    Returns:
        kappa_i: scalar to multiply the noise-expanded state by
    """
    return scale_ratio / (1.0 + (scale_ratio - 1.0) * t_i)


def align(t_i: float, scale_ratio: float) -> Tuple[float, float]:
    """Return both (aligned_t, state_scale) at once."""
    return aligned_timestep(t_i, scale_ratio), state_scale_factor(t_i, scale_ratio)


# ============================================================================
# Identities to sanity-check the formulas
# ============================================================================

def _verify_endpoints():
    """Spot-check boundary behavior:
       - At t_i = 0:   aligned_t = 0,        kappa = scale_ratio (pure noise upscaling)
       - At t_i = 1:   aligned_t = 1,        kappa = 1            (pure noise; no shift)
    """
    for r in [1.5, 2.0, 3.0]:
        # t=0 endpoint
        t_a = aligned_timestep(0.0, r); k = state_scale_factor(0.0, r)
        assert abs(t_a - 0.0) < 1e-12, f"r={r}: aligned_t(0)={t_a} != 0"
        assert abs(k - r) < 1e-12, f"r={r}: kappa(0)={k} != r"
        # t=1 endpoint
        t_a = aligned_timestep(1.0, r); k = state_scale_factor(1.0, r)
        assert abs(t_a - 1.0) < 1e-12, f"r={r}: aligned_t(1)={t_a} != 1"
        assert abs(k - 1.0) < 1e-12, f"r={r}: kappa(1)={k} != 1"
    print("✓ Endpoint identities hold for r ∈ {1.5, 2.0, 3.0}")


def _verify_against_sd3_shift():
    """SPD's Eq 6 is functionally identical to SD3's shifted-time reparam
    with shift=scale_ratio. SD3:  τ = shift * t / (1 + (shift-1)*t).
    SPD:  \tilde t = r * t / (1 + (r-1)*t).
    """
    import random
    random.seed(0)
    for _ in range(10):
        t = random.random()
        r = random.uniform(1.0, 5.0)
        spd_t = aligned_timestep(t, r)
        sd3_t = r * t / (1 + (r - 1) * t)
        assert abs(spd_t - sd3_t) < 1e-12
    print("✓ SPD aligned_timestep ≡ SD3 shifted-time with shift=r (10 random samples)")


def _identity_at_r_equals_1():
    """When scale_ratio = 1 (no resolution change), the alignment is identity."""
    import random
    random.seed(1)
    for _ in range(10):
        t = random.random()
        t_a, k = align(t, 1.0)
        assert abs(t_a - t) < 1e-12 and abs(k - 1.0) < 1e-12
    print("✓ r=1 gives identity alignment")


if __name__ == "__main__":
    _verify_endpoints()
    _verify_against_sd3_shift()
    _identity_at_r_equals_1()

    print("\nExample schedule (scale_ratio=2 doubles resolution):")
    print(f"  {'t_i':>6}  {'tilde_t_i':>10}  {'kappa':>8}")
    for t in [0.95, 0.8, 0.6, 0.4, 0.2, 0.05]:
        t_a, k = align(t, 2.0)
        print(f"  {t:6.3f}  {t_a:10.4f}  {k:8.4f}")
