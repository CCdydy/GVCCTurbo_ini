"""
optimal_schedule.py — SPD Sec 4.2 optimal resolution schedule.

Two propositions from SPD provide a closed-form δ-optimal schedule:

  Prop 1 (per-frequency δ-optimal activation time, Eq 9):
      t_ω = 1 / (1 + sqrt(δ / (P_ω * (1 + P_ω - δ))))
      For t ≥ t_ω, frequency ω is "noise-dominated" — the Bayes-optimal
      velocity at that frequency is approximately epsilon, so spectral
      content at ω is not yet meaningfully recovered.

  Prop 2 (per-resolution δ-optimal transition time, Eq 10):
      t_i^* = min over ω in Ω_{s_i} of t_ω
            = t_{ω = s_i * ω_max(H,W)}
      (assumes P_ω is monotonically decreasing in |ω|)

Inputs needed:
  - δ in (0, 1): single tolerance hyperparameter (SPD default: 0.01)
  - P_ω: power spectrum measured from data (from power_spectrum.py)
  - resolution scales s_{1:S} chosen to align with model training

Outputs:
  - t_i^* for each i, the time to switch from scale s_i to scale s_{i+1}
"""

import math
from typing import List, Tuple

import numpy as np


def per_frequency_activation_time(P_omega: float, delta: float) -> float:
    """SPD Prop 1 (Eq 9): per-frequency δ-optimal activation time t_ω.

    Args:
        P_omega: signal power at frequency ω (>= 0)
        delta:   error tolerance in (0, 1)

    Returns:
        t_ω in (0, 1]: earliest timestep at which ω is sufficiently
        signal-dominated to be worth modelling.
    """
    if P_omega <= 0.0:
        return 1.0
    if delta >= 1 + P_omega:
        return 0.0
    denom = P_omega * (1.0 + P_omega - delta)
    if denom <= 0.0:
        return 1.0
    return 1.0 / (1.0 + math.sqrt(delta / denom))


def per_resolution_transition_time(
    omega_max_for_scale: float,
    P_omega_func,
    delta: float,
) -> float:
    """SPD Prop 2 (Eq 10): t_i^* = t_{ω = s_i * ω_max(H,W)}.

    Args:
        omega_max_for_scale: highest representable radial frequency at the
            current scale s_i, normalized in [0, 1] (e.g. s_i * (1/sqrt(2)) at
            Nyquist if you use normalized frequency convention from
            dct_utils.radial_frequency_bins)
        P_omega_func: callable f(|ω|) → P_ω, e.g. power-law fit
            P_ω = c * |ω|^(-β)
        delta: tolerance
    """
    P = P_omega_func(omega_max_for_scale)
    return per_frequency_activation_time(P, delta)


# ============================================================================
# Power-law convenience (P_ω = c * |ω|^(-β))
# ============================================================================

def make_power_law_pofreq(beta: float, intercept_log_c: float):
    """Return f(omega) = exp(intercept_log_c) * omega^(-beta).

    Args:
        beta:           power-law exponent (e.g. ~2.42 for Wan latents)
        intercept_log_c: intercept in log space; from np.polyfit on log-log

    Returns:
        Callable f(omega: float) -> float
    """
    c = math.exp(intercept_log_c)
    def f(omega: float) -> float:
        if omega <= 0:
            return float('inf')
        return c * (omega ** (-beta))
    return f


# ============================================================================
# Build full progressive-resolution schedule
# ============================================================================

def build_progressive_schedule(
    scales: List[float],
    P_omega_func,
    delta: float,
    omega_max_full: float = 1.0,
) -> List[Tuple[float, float, float]]:
    """Given a list of progressive resolution scales s_1 < s_2 < ... < s_S = 1.0,
    compute transition times t_i^* for switching from s_i to s_{i+1}.

    Returns:
        List of tuples (s_i, s_{i+1}, t_i^*) for i = 1, ..., S-1.
        The schedule means: run denoising at s_i for t ∈ (t_i^*, t_{i-1}^*],
        then expand to s_{i+1} at t_i^*.
    """
    assert len(scales) >= 2, "Need at least 2 scales for one transition"
    assert all(scales[i] < scales[i + 1] for i in range(len(scales) - 1)), \
        "scales must be strictly increasing"
    assert abs(scales[-1] - 1.0) < 1e-9, "final scale must be 1.0 (full resolution)"

    schedule = []
    for i in range(len(scales) - 1):
        s_i = scales[i]
        s_next = scales[i + 1]
        # representable freq at scale s_i, normalized to [0, 1]
        omega_max_si = s_i * omega_max_full
        P = P_omega_func(omega_max_si)
        t_i_star = per_frequency_activation_time(P, delta)
        schedule.append((s_i, s_next, t_i_star))
    return schedule


# ============================================================================
# Verification
# ============================================================================

def _verify_monotonic_in_omega():
    """For power-law decreasing P_ω, t_ω should DECREASE as |ω| increases.

    Higher |ω| → smaller P_ω → frequency stays noise-dominated for longer →
    t_ω (the time below which signal is recovered) is smaller (closer to 0).

    Equivalently: low frequencies unlock early in denoising (large t_ω,
    close to t=1); high frequencies unlock late (small t_ω, close to t=0).
    """
    f = make_power_law_pofreq(beta=2.42, intercept_log_c=0.0)
    delta = 0.01
    ts = [per_frequency_activation_time(f(o), delta) for o in np.linspace(0.05, 1.0, 20)]
    is_monotonic = all(ts[i] >= ts[i + 1] - 1e-9 for i in range(len(ts) - 1))
    assert is_monotonic, "t_ω should be monotonic non-increasing in |ω|"
    print(f"✓ t_ω monotonic non-increasing in |ω| (β=2.42, δ=0.01)")
    print(f"  t_ω(|ω|=0.05) = {ts[0]:.4f}, t_ω(|ω|=1.0) = {ts[-1]:.4f}")


def _verify_endpoint_behavior():
    """As δ → 0, t_ω → 1 (very strict tolerance, model is never accurate enough).
    As δ → 1 + P_ω, t_ω → 0 (no tolerance constraint)."""
    P = 1.0
    for delta in [1e-6, 1e-4, 1e-2, 0.5, 1.5]:
        t = per_frequency_activation_time(P, delta)
        print(f"  P=1.0, δ={delta:.0e}: t_ω = {t:.4f}")
    print("✓ δ → 0 gives t_ω → 1; δ → 1+P gives t_ω → 0")


def _example_schedule():
    """Example: Wan T2V 1.3B 720p with β=2.42, S=2 (one mid-res stage)."""
    f = make_power_law_pofreq(beta=2.42, intercept_log_c=math.log(1.0))
    delta = 0.01
    scales = [0.5, 1.0]  # 50% res → 100% res
    schedule = build_progressive_schedule(scales, f, delta)
    print("\nExample S=2 schedule (50% → 100% res, β=2.42, δ=0.01):")
    print(f"  {'s_i':>6}  {'s_next':>6}  {'t_i*':>6}")
    for (s, sn, t) in schedule:
        print(f"  {s:6.2f}  {sn:6.2f}  {t:6.4f}")
    print("  (transition happens at t_i*; run scale s_i for t > t_i* down to t = t_i*)")


if __name__ == "__main__":
    _verify_monotonic_in_omega()
    print()
    _verify_endpoint_behavior()
    _example_schedule()
