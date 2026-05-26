"""
dct_utils.py — Orthonormal DCT/IDCT and radial frequency binning.

DCT-II (orthonormal) implementation matches scipy.fftpack.dct(type=2, norm='ortho').
Used throughout SPD reproduction for spectral noise expansion (Sec 4.1)
and power spectrum analysis (Sec 3.2 / Fig 2a).

DCT chosen as default transform per SPD Sec 5.4 (DCT ≈ DWT, both > FFT).

Algorithm: zh217/torch-dct based.
"""

import math
from typing import Optional

import torch
import numpy as np


# ============================================================================
# 1D DCT-II / DCT-III (raw, no normalization)
# ============================================================================

def _dct1_raw(x: torch.Tensor) -> torch.Tensor:
    """Raw DCT-II along last dim, matches scipy.fftpack.dct(x, type=2, norm=None)."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i
    V = 2 * V

    return V.view(*x_shape)


def _idct1_raw(X: torch.Tensor) -> torch.Tensor:
    """Raw inverse of _dct1_raw: matches scipy.fftpack.idct(X, type=2, norm=None)."""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N) / 2

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    Vc = torch.complex(V_r, V_i)
    v = torch.fft.irfft(Vc, n=N, dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


# ============================================================================
# Orthonormal wrappers
# ============================================================================

def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Type-II DCT along last dim.
    Matches scipy.fftpack.dct(x, type=2, norm='ortho').
    """
    N = x.shape[-1]
    Y = _dct1_raw(x)
    Y = Y.clone()
    Y[..., 0] *= 1.0 / (2 * math.sqrt(N))
    Y[..., 1:] *= 1.0 / math.sqrt(2 * N)
    return Y


def idct_1d(X: torch.Tensor) -> torch.Tensor:
    """Inverse of dct_1d (orthonormal Type-III DCT).
    Matches scipy.fftpack.idct(X, type=2, norm='ortho').
    """
    N = X.shape[-1]
    X = X.clone()
    X[..., 0] *= 2 * math.sqrt(N)
    X[..., 1:] *= math.sqrt(2 * N)
    return _idct1_raw(X)


# ============================================================================
# 2D DCT-II / IDCT-III (orthonormal)
# ============================================================================

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D Type-II DCT along the last two dims.

    Args:
        x: (..., H, W)

    Returns:
        X: (..., H, W)
    """
    X = dct_1d(x)
    X = dct_1d(X.transpose(-1, -2)).transpose(-1, -2)
    return X


def idct_2d(X: torch.Tensor) -> torch.Tensor:
    """Inverse of dct_2d."""
    x = idct_1d(X)
    x = idct_1d(x.transpose(-1, -2)).transpose(-1, -2)
    return x


# ============================================================================
# Radial frequency binning (for power spectrum)
# ============================================================================

def radial_frequency_bins(H: int, W: int, num_bins: Optional[int] = None, device='cpu'):
    """Return radial frequency magnitudes for an HxW DCT grid and bin
    assignments for radial averaging.

    Args:
        H, W: spatial dims
        num_bins: number of radial bins. Defaults to min(H,W)//2.
        device: torch device

    Returns:
        omega_mag: (H, W) normalized magnitude in [0,1]
        bin_idx:   (H, W) integer bin assignment in [0, num_bins-1]
        bin_centers: (num_bins,) approximate freq magnitude of each bin
    """
    if num_bins is None:
        num_bins = min(H, W) // 2

    fy = torch.arange(H, device=device, dtype=torch.float32) / H
    fx = torch.arange(W, device=device, dtype=torch.float32) / W

    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    omega_mag = torch.sqrt(fy_grid ** 2 + fx_grid ** 2) / math.sqrt(2)
    # Normalized so the (1,1) corner -> 1.0

    bin_idx = (omega_mag * num_bins).clamp(0, num_bins - 1).long()
    bin_centers = (torch.arange(num_bins, device=device, dtype=torch.float32) + 0.5) / num_bins
    return omega_mag, bin_idx, bin_centers


def radial_average(X_power: torch.Tensor, bin_idx: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Average per-bin energy across radial frequency rings.

    Args:
        X_power: (..., H, W) per-bin energy (e.g. |X|^2 of DCT coeffs)
        bin_idx: (H, W) bin assignment
        num_bins: number of bins

    Returns:
        radial_avg: (..., num_bins)
    """
    *prefix, H, W = X_power.shape
    flat_power = X_power.reshape(-1, H * W)
    flat_bin = bin_idx.reshape(H * W)

    out = flat_power.new_zeros(flat_power.shape[0], num_bins)

    for b in range(num_bins):
        mask = (flat_bin == b)
        if mask.sum() > 0:
            out[:, b] = flat_power[:, mask].mean(dim=-1)

    out = out.reshape(*prefix, num_bins)
    return out


# ============================================================================
# Verification
# ============================================================================

def _verify_against_scipy():
    """Verify DCT/IDCT against scipy reference."""
    try:
        from scipy.fftpack import dct as sp_dct, idct as sp_idct, dctn, idctn
    except ImportError:
        print("scipy not available; skipping verification")
        return

    torch.manual_seed(0)

    x = torch.randn(4, 64).double()
    X_ours = dct_1d(x)
    X_scipy = torch.from_numpy(sp_dct(x.numpy(), type=2, norm='ortho', axis=-1))
    print(f"1D DCT max err vs scipy: {(X_ours - X_scipy).abs().max().item():.3e}")

    x_rec = idct_1d(X_ours)
    print(f"1D DCT round-trip max err: {(x_rec - x).abs().max().item():.3e}")

    X_ours_scipy_inv = torch.from_numpy(sp_idct(X_ours.numpy(), type=2, norm='ortho', axis=-1))
    print(f"1D IDCT max err vs scipy: {(x_rec - X_ours_scipy_inv).abs().max().item():.3e}")

    x2 = torch.randn(2, 16, 24).double()
    X_ours = dct_2d(x2)
    X_scipy = torch.from_numpy(dctn(x2.numpy(), type=2, norm='ortho', axes=(-2, -1)))
    print(f"2D DCT max err vs scipy: {(X_ours - X_scipy).abs().max().item():.3e}")

    x2_rec = idct_2d(X_ours)
    print(f"2D DCT round-trip max err: {(x2_rec - x2).abs().max().item():.3e}")

    # GPU test
    if torch.cuda.is_available():
        x_gpu = torch.randn(2, 60, 104, device='cuda').float()
        X_gpu = dct_2d(x_gpu)
        x_gpu_rec = idct_2d(X_gpu)
        print(f"2D DCT GPU round-trip max err (float32): {(x_gpu_rec - x_gpu).abs().max().item():.3e}")

    omega_mag, bin_idx, bin_centers = radial_frequency_bins(64, 64, num_bins=16)
    print(f"Radial bins: bin_idx range [{bin_idx.min().item()}, {bin_idx.max().item()}]")
    print(f"             bin_centers[0,8,15] = {bin_centers[[0,8,15]].tolist()}")


if __name__ == "__main__":
    _verify_against_scipy()
