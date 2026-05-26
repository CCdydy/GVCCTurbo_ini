"""
run_spd_wan.py — entry point for SPD reproduction on Wan 2.1-T2V.

Runs baseline vs SPD-accelerated generation on the same prompt+seed,
reports wall-clock speedup, and writes both videos for visual comparison.
Optionally invokes power_spectrum.py first to measure β/intercept on Wan
latents and feeds those into the SPD schedule.

Targets SPD paper Table 3 (Wan 2.1-T2V-1.3B 720p, 50-step baseline):
  Wan 2.1 (50 steps): 1.00× speedup baseline
  SPD (S=2):          2.03× speedup, VBench ≈ baseline

Usage:
  python run_spd_wan.py --measure_beta --prompt "..." --num_steps 50
"""

import os
import sys
import json
import math
import time
import argparse
import subprocess
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import imageio

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from spd_pipeline import SpectralProgressiveWan


# ----------------------------------------------------------------------------
# Baseline Wan inference (no SPD)
# ----------------------------------------------------------------------------

class WanBaseline:
    """Standard Wan 2.1-T2V inference at fixed resolution (no progressive growth)."""

    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(self, model_dir: str, flow_shift: float = 3.0):
        self.model_dir = os.path.abspath(model_dir)
        self.flow_shift = flow_shift
        self._loaded = False

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from diffusers import WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
        from transformers import UMT5EncoderModel, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_dir, "tokenizer"))
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.model_dir, "text_encoder"), torch_dtype=dtype
        ).eval().requires_grad_(False)
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.model_dir, "vae"), torch_dtype=torch.float32
        ).eval().requires_grad_(False).to(self.device)
        self.transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(self.model_dir, "transformer"), torch_dtype=dtype
        ).eval().requires_grad_(False).to(self.device)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.model_dir, subfolder="scheduler")

        mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        self.latent_mean = mean.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_std = std.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_channels = self.vae.config.z_dim
        self._loaded = True

    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_length: int = 226):
        self.text_encoder.to(self.device)
        inputs = self.tokenizer(
            [prompt], padding="max_length", max_length=max_length,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.to(self.device)
        hidden = self.text_encoder(ids, attention_mask=mask).last_hidden_state.to(self.dtype)
        seq_len = mask.gt(0).sum(dim=1).long()
        trimmed = hidden[0, : seq_len[0]]
        padded = torch.cat([
            trimmed,
            trimmed.new_zeros(max_length - trimmed.size(0), trimmed.size(1)),
        ]).unsqueeze(0)
        return padded

    @torch.no_grad()
    def generate(self, prompt, num_frames=33, height=720, width=1280,
                 num_steps=50, seed=42, return_timing=False):
        assert self._loaded
        prompt_embeds = self.encode_prompt(prompt)

        f_lat = (num_frames - 1) // 4 + 1
        h_lat = height // 8; h_lat += h_lat % 2
        w_lat = width // 8;  w_lat += w_lat % 2

        gen = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(
            (1, self.latent_channels, f_lat, h_lat, w_lat),
            generator=gen, device=self.device, dtype=torch.float32,
        )

        self.scheduler.set_timesteps(num_inference_steps=num_steps, device=self.device)

        t0 = time.time()
        for t in self.scheduler.timesteps:
            t_in = t.reshape(1).to(device=self.device, dtype=self.dtype)
            with torch.cuda.amp.autocast(dtype=self.dtype):
                v = self.transformer(
                    hidden_states=x.to(self.dtype),
                    timestep=t_in,
                    encoder_hidden_states=prompt_embeds.to(self.dtype),
                    return_dict=False,
                )[0]
            v = v.float()
            x = self.scheduler.step(v, t, x, return_dict=False)[0]
        wall = time.time() - t0

        # Decode
        raw_z = x.float().to(self.device) * self.latent_std + self.latent_mean
        video = self.vae.decode(raw_z, return_dict=False)[0].squeeze(0)
        video = (video / 2.0 + 0.5).clamp(0, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()
        frames = [Image.fromarray((v * 255).astype(np.uint8)) for v in video]

        if return_timing:
            return frames, {"wall": wall, "num_steps": num_steps}
        return frames


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def save_video(frames, path, fps=25):
    path = str(path)
    arrays = [np.array(f) for f in frames]
    imageio.mimwrite(path, arrays, fps=fps, codec='libx264', quality=8)


def maybe_measure_beta(output_dir, wan_ckpt, data_dir, height, width):
    """Run power_spectrum.py and read back β, intercept from its JSON output."""
    print("\n=== Measuring β on Wan latents (power_spectrum.py) ===")
    cmd = [
        sys.executable,
        os.path.join(_here, "power_spectrum.py"),
        "--wan_ckpt", wan_ckpt,
        "--data_dir", data_dir,
        "--height", str(height), "--width", str(width),
        "--output_dir", str(output_dir),
    ]
    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    js_path = os.path.join(output_dir, "power_spectrum.json")
    with open(js_path) as f:
        js = json.load(f)
    beta = js["aggregate"]["beta"]
    intercept = js["aggregate"]["intercept"]
    print(f"  → Wan latent β = {beta:.3f}, intercept = {intercept:.3f}")
    return beta, intercept


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_ckpt",
                        default="/home/rog/Desktop/gvcc_turbo/checkpoints/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--data_dir",
                        default="/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg")
    parser.add_argument("--output_dir", default=os.path.join(_here, "results"))
    parser.add_argument("--prompt", default="A beautiful butterfly flying over a meadow at sunset, cinematic.")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--measure_beta", action="store_true",
                        help="If set, run power_spectrum.py to fit β; else use --beta/--beta_intercept")
    parser.add_argument("--beta", type=float, default=2.42)
    parser.add_argument("--beta_intercept", type=float, default=0.0)
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline run (only SPD)")
    parser.add_argument("--skip_spd", action="store_true",
                        help="Skip SPD run (only baseline)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Optionally measure β empirically
    beta = args.beta
    intercept = args.beta_intercept
    if args.measure_beta:
        beta, intercept = maybe_measure_beta(out, args.wan_ckpt, args.data_dir, 480, 832)
        # ^ measure on the 480p latent (cheap), use for all schedules

    results = {"config": vars(args), "beta_used": beta, "intercept_used": intercept}

    # -------- baseline --------
    if not args.skip_baseline:
        print("\n=== Baseline Wan 2.1 inference ===")
        baseline = WanBaseline(args.wan_ckpt, flow_shift=args.flow_shift)
        baseline.load("cuda", torch.bfloat16)
        frames_b, timing_b = baseline.generate(
            prompt=args.prompt,
            num_frames=args.num_frames, height=args.height, width=args.width,
            num_steps=args.num_steps, seed=args.seed, return_timing=True,
        )
        save_video(frames_b, out / "baseline.mp4")
        print(f"  baseline wall = {timing_b['wall']:.2f}s ({args.num_steps} steps)")
        results["baseline"] = timing_b
        del baseline
        torch.cuda.empty_cache()

    # -------- SPD --------
    if not args.skip_spd:
        print("\n=== SPD inference ===")
        spd = SpectralProgressiveWan(
            model_dir=args.wan_ckpt,
            delta=args.delta, scales=args.scales,
            beta=beta, beta_intercept=intercept,
            flow_shift=args.flow_shift,
        )
        spd.load("cuda", torch.bfloat16)
        frames_s, timing_s = spd.generate(
            prompt=args.prompt,
            num_frames=args.num_frames, height=args.height, width=args.width,
            num_steps=args.num_steps, seed=args.seed, return_timings=True,
        )
        save_video(frames_s, out / "spd.mp4")
        print(f"  SPD wall = {timing_s['total_wall']:.2f}s, stages = {timing_s['schedule']}")
        print(f"  transition_times = {timing_s['transition_times']}")
        for s in timing_s["stages"]:
            print(f"    stage {s['stage']} (scale {s['scale']}, steps={s['steps']}): {s['wall']:.2f}s")
        results["spd"] = timing_s
        del spd
        torch.cuda.empty_cache()

    if not args.skip_baseline and not args.skip_spd:
        speedup = results["baseline"]["wall"] / results["spd"]["total_wall"]
        print(f"\n=== SPEEDUP = {speedup:.2f}× ===")
        print(f"  (SPD paper Table 3 reports 2.03× for Wan 2.1-T2V-1.3B 720p, S=2)")
        results["speedup"] = speedup

    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
