"""
power_spectrum.py — measure Wan 2.1 latent power spectrum.

Reproduces SPD Fig 2a: per-frequency power P_omega ∝ |omega|^(-β)
on Wan 2.1-T2V-1.3B latent space, fitting β.

SPD reports β ≈ 2.42 for Wan 2.1 video latents (and ~1.92 for FLUX
image latents). We expect similar values on our backbone.
"""

import os
import sys
import math
import argparse
import json
from pathlib import Path
import re

import torch
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from dct_utils import dct_2d, radial_frequency_bins, radial_average


# --- UVG loader (mini, no dependency on the GVCC uvg_data module) ---

def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    import cv2
    match = re.search(r'(\d+)x(\d+)', os.path.basename(yuv_path))
    W, H = int(match.group(1)), int(match.group(2))
    frame_size = H * W * 3 // 2
    frames = []
    with open(yuv_path, 'rb') as f:
        f.seek(start_frame * frame_size)
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:H * W].reshape(H, W)
            u = yuv[H * W:H * W + H * W // 4].reshape(H // 2, W // 2)
            v = yuv[H * W + H * W // 4:].reshape(H // 2, W // 2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            yuv_img = np.stack([y, u, v], axis=-1)
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames


def encode_to_latent(vae, frames, height, width, device, dtype=torch.float32):
    """VAE-encode a list of PIL frames to a normalized latent.
    Returns: (C, F_lat, H_lat, W_lat) tensor on CPU
    """
    processed = []
    for frame in frames:
        f = frame.resize((width, height), Image.LANCZOS)
        arr = np.array(f).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        processed.append(t)
    video = torch.stack(processed, dim=1).unsqueeze(0)  # (1, 3, F, H, W)
    video = 2.0 * video - 1.0
    video = video.to(device=device, dtype=dtype)

    with torch.no_grad():
        posterior = vae.encode(video).latent_dist
        raw_z = posterior.mode()

    mean = torch.tensor(vae.config.latents_mean, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    z = (raw_z - mean) / std
    return z.squeeze(0).cpu()  # (C, F_lat, H_lat, W_lat)


def compute_power_spectrum(latent: torch.Tensor, num_bins: int = 32, device='cuda'):
    """For a latent (C, F_lat, H_lat, W_lat), compute radial-averaged power.

    Averages across channels and temporal frames.

    Returns:
        bin_centers: (num_bins,)
        P_omega:     (num_bins,)
    """
    C, F, H, W = latent.shape

    # 2D DCT per (channel, frame), then |X|^2
    z = latent.to(device).float()  # (C, F, H, W)
    z_flat = z.reshape(C * F, H, W)
    X = dct_2d(z_flat)  # (CF, H, W)
    P = X.pow(2)  # (CF, H, W)

    omega_mag, bin_idx, bin_centers = radial_frequency_bins(H, W, num_bins=num_bins, device=device)
    # Mean power per radial bin per (channel, frame); then mean over (channel, frame)
    radial = radial_average(P, bin_idx, num_bins)  # (CF, num_bins)
    P_omega = radial.mean(dim=0)  # (num_bins,)
    return bin_centers.cpu(), P_omega.cpu()


def fit_power_law(bin_centers: torch.Tensor, P_omega: torch.Tensor,
                  fit_low_frac=0.1, fit_high_frac=0.9):
    """Fit log P = a - beta * log omega over a middle portion of the curve
    (avoid DC and Nyquist where finite-grid effects dominate).

    Returns:
        beta, intercept_a, fit_range (low_freq, high_freq)
    """
    centers = bin_centers.numpy()
    P = P_omega.numpy()

    N = len(centers)
    lo_idx = max(1, int(N * fit_low_frac))
    hi_idx = min(N - 1, int(N * fit_high_frac))

    x = np.log(centers[lo_idx:hi_idx] + 1e-12)
    y = np.log(P[lo_idx:hi_idx] + 1e-12)

    # Linear fit y = a + slope * x → slope = -beta
    slope, intercept = np.polyfit(x, y, 1)
    beta = -slope
    return float(beta), float(intercept), (float(centers[lo_idx]), float(centers[hi_idx - 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_ckpt",
                        default="/home/rog/Desktop/gvcc_turbo/checkpoints/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--data_dir",
                        default="/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg")
    parser.add_argument("--sequences", nargs="*",
                        default=["Beauty", "Bosphorus", "HoneyBee", "Jockey",
                                 "ReadySteadyGo", "ShakeNDry", "YachtRide"])
    parser.add_argument("--num_frames", type=int, default=33,
                        help="Frames per sequence (one GOP equivalent)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_bins", type=int, default=32)
    parser.add_argument("--output_dir", default=os.path.join(_here, "results"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load VAE (only — no DiT needed for power spectrum)
    print(f"Loading Wan VAE from {args.wan_ckpt}/vae ...")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.wan_ckpt, "vae"),
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False).to("cuda")
    print(f"  z_dim={vae.config.z_dim}, base_dim={vae.config.base_dim}")

    # Find sequences
    yuv_files = {}
    for p in Path(args.data_dir).rglob("*.yuv"):
        name = p.name.split('_')[0]
        if name in args.sequences:
            yuv_files[name] = p

    if not yuv_files:
        print(f"No yuv sequences found in {args.data_dir}; aborting")
        return

    # Accumulate spectra per sequence and overall
    all_curves = {}
    centers_ref = None

    for seq in args.sequences:
        if seq not in yuv_files:
            print(f"  Skipping {seq}: yuv not found")
            continue
        print(f"\n=== {seq} ===")
        frames = load_yuv420_frames(str(yuv_files[seq]), args.num_frames)
        z = encode_to_latent(vae, frames, args.height, args.width, "cuda")
        print(f"  latent: {tuple(z.shape)}")

        centers, P_omega = compute_power_spectrum(z, num_bins=args.num_bins)
        beta, intercept, fit_range = fit_power_law(centers, P_omega)
        print(f"  beta = {beta:.3f}  (fit range omega ∈ [{fit_range[0]:.3f}, {fit_range[1]:.3f}])")
        all_curves[seq] = {
            "bin_centers": centers.tolist(),
            "P_omega": P_omega.tolist(),
            "beta": beta,
            "intercept": intercept,
            "fit_range": fit_range,
        }
        centers_ref = centers

        # Free
        del z
        torch.cuda.empty_cache()

    # Overall: average across sequences
    Ps = torch.stack([torch.tensor(c["P_omega"]) for c in all_curves.values()], dim=0)
    P_avg = Ps.mean(dim=0)
    beta_avg, intercept_avg, fit_range_avg = fit_power_law(centers_ref, P_avg)
    print(f"\n=== Aggregate ===")
    print(f"  beta (avg curve) = {beta_avg:.3f}  (fit range {fit_range_avg})")

    # SPD reports beta ≈ 2.42 for Wan video latents
    print(f"\n  Reference: SPD reports beta ≈ 2.42 for Wan 2.1-T2V latent (Fig 2a)")
    print(f"  Our measurement: beta_avg = {beta_avg:.3f}")
    if abs(beta_avg - 2.42) < 0.5:
        print(f"  ✓ Within ±0.5 of SPD report — spectral autoregression confirmed")
    else:
        print(f"  ⚠ Deviation from SPD report > 0.5 — investigate basis or VAE")

    # Save JSON
    summary = {
        "config": vars(args),
        "per_sequence": all_curves,
        "aggregate": {
            "bin_centers": centers_ref.tolist(),
            "P_omega": P_avg.tolist(),
            "beta": beta_avg,
            "intercept": intercept_avg,
            "fit_range": fit_range_avg,
        },
        "reference_beta_spd": 2.42,
    }
    json_path = out / "power_spectrum.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # Optional matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for seq, c in all_curves.items():
            ax.loglog(c["bin_centers"], c["P_omega"], alpha=0.5, label=f"{seq} (β={c['beta']:.2f})")
        ax.loglog(centers_ref, P_avg.tolist(), 'k-', lw=2.5, label=f"avg (β={beta_avg:.2f})")
        # Reference line: β=2.42
        x_ref = np.array([0.05, 1.0])
        y_ref = np.exp(intercept_avg) * x_ref ** (-2.42)
        ax.loglog(x_ref, y_ref, 'r--', label="SPD β=2.42 reference")
        ax.set_xlabel(r"normalized radial frequency $|\omega|$")
        ax.set_ylabel(r"$P_\omega$")
        ax.set_title(f"Wan 2.1-T2V latent power spectrum ({args.height}p) — fit β={beta_avg:.2f}")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plot_path = out / "power_spectrum.png"
        fig.savefig(plot_path, dpi=150)
        print(f"  Saved: {plot_path}")
    except ImportError:
        print("  matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
