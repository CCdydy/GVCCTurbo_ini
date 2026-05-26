"""
run_mr_experiment.py — V1 MR-GVCC Experiment

Compares multi-resolution GVCC variants against the full-resolution baseline
on UVG using Wan2.1-T2V-1.3B at 720p (full) / 480p (low stage).

Methods (--method):
  full_res    Original single-resolution GVCC (baseline)
  mr_random   MR-GVCC with random high-freq transition [SPD-style]
  mr_oracle   MR-GVCC with oracle high-freq transition [encode-only upper bound]
  mr_codebook MR-GVCC with codebook high-freq transition [target-aware, main]

Quick single-sequence smoke test:
  python exp_v1_mr/run_mr_experiment.py \
      --method mr_random \
      --sequences Beauty \
      --num_gops 1

Full ablation (all 4 methods × UVG 7 seq × 3 GOP):
  for m in full_res mr_random mr_codebook; do
    python exp_v1_mr/run_mr_experiment.py --method $m --output_dir exp_v1_mr/results_$m
  done
"""

import sys
import os
import re
import time
import json
import gc
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from sde_rf_wan.wan_t2v_wrapper import WanT2VDiffusersWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from sde_rf_wan.mr_pipeline import MRTurboDDCMPipeline
from uvg_data import find_uvg_sequences


# ==================================================================
# UVG loading helpers (copied from run_t2v_experiment.py)
# ==================================================================

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


def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]


# ==================================================================
# Metrics
# ==================================================================

def frames_to_tensor(frames):
    return torch.stack([
        torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1)
        for f in frames
    ])


def compute_psnr(orig, recon):
    mse = ((orig - recon) ** 2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item(), psnr


def compute_msssim(orig, recon):
    try:
        from pytorch_msssim import ms_ssim
        vals = []
        for i in range(0, orig.shape[0], 4):
            v = ms_ssim(orig[i:i + 4], recon[i:i + 4],
                        data_range=1.0, size_average=False)
            vals.extend(v.cpu().tolist())
        return sum(vals) / len(vals)
    except ImportError:
        return None


def compute_lpips(orig, recon, device="cuda"):
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(device)
    orig_lp = (2.0 * orig - 1.0).to(device)
    recon_lp = (2.0 * recon - 1.0).to(device)
    vals = []
    for i in range(0, orig_lp.shape[0], 4):
        with torch.no_grad():
            d = loss_fn(orig_lp[i:i + 4], recon_lp[i:i + 4])
        vals.extend(d.flatten().cpu().tolist())
    del loss_fn
    torch.cuda.empty_cache()
    return sum(vals) / len(vals)


def save_video_mp4(frames, path, fps=16):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()


# ==================================================================
# Per-GOP encode + decode (handles both pipeline types)
# ==================================================================

def run_gop(pipe, gop_frames, args, is_mr: bool, prev_z_last=None):
    """Run encode + decode for one GOP.

    Returns (recon_frames, t_enc, t_dec, this_z_last, boundary_used).
    For non-MR or when boundary_link is off, this_z_last is None.

    boundary_link uses the prediction-error formulation:
        r = z_first_GT − z_first_pred^{n+1}   (decoder's own first-frame prediction)
    Encoder does decode once (no correction) to obtain z_first_pred, encodes
    r via the boundary codebook, applies correction to x_T_pred, and runs a
    second VAE decode to produce the final corrected frames. SDE is run once.
    """
    print(f"  Encoding...")
    t0 = time.time()
    if is_mr:
        step_data_per_stage, trans_atoms, _ = pipe.encode(gop_frames, prompt="")
    else:
        step_data, _ = pipe.encode(gop_frames, prompt="")
    t_enc = time.time() - t0

    print(f"  Decoding...")
    t0 = time.time()
    boundary_used = False
    if is_mr:
        # First decode pass: no correction. Capture x_T_pred for boundary use.
        frames_recon, x_T_pred = pipe.decode(
            step_data_per_stage, trans_atoms, prompt="",
            latent_correction=None,
            return_final_latent=True,
        )
        this_z_last = x_T_pred[0, :, -1, :, :].detach().clone()

        # Apply boundary-link correction (skip for GOP 0 — no anchor needed
        # for prediction-error form, but we gate on prev_z_last to keep it
        # off for the first GOP; first GOP's prediction is presumably fine).
        if getattr(pipe, "boundary_link", False) and prev_z_last is not None:
            gt_lat = pipe.model.encode_video(
                gop_frames[:1], pipe.full_h, pipe.full_w,
            )
            z_first_gt = gt_lat[0, :, 0, :, :].detach()
            z_first_pred = x_T_pred[0, :, 0, :, :].detach()
            atoms = pipe.encode_boundary(z_first_gt, z_first_pred)
            H_lat, W_lat = pipe.full_lat[2], pipe.full_lat[3]
            r_recon = pipe.decode_boundary(atoms, (H_lat, W_lat))
            correction = pipe.build_boundary_correction(r_recon)
            x_T_corr = x_T_pred + correction.to(x_T_pred.device, dtype=x_T_pred.dtype)
            frames_recon = pipe.model.decode_latent(x_T_corr)
            this_z_last = x_T_corr[0, :, -1, :, :].detach().clone()
            boundary_used = True
    else:
        frames_recon = pipe.decode(step_data, prompt="")
        this_z_last = None
    t_dec = time.time() - t0

    return frames_recon, t_enc, t_dec, this_z_last, boundary_used


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V1 MR-GVCC Experiment")
    parser.add_argument("--data_dir",
                        default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt",
                        default=os.path.join(_project_root, "exp_t2v",
                                             "Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--output_dir", default="./exp_v1_mr/results")
    parser.add_argument("--method", default="mr_random",
                        choices=["full_res", "mr_random", "mr_oracle",
                                 "mr_codebook", "mr_prior_codebook",
                                 "mr_dlc1_pure", "mr_dlc1_codebook",
                                 "mr_dlc2_subband"],
                        help="Which pipeline variant to run")
    parser.add_argument("--dlc2_rank_frac", type=float, default=0.5,
                        help="DLC2 sub-band fraction of high-freq dim (0,1]")
    parser.add_argument("--low_target_from_full", action="store_true",
                        help="Override low-stage x0 with T_L(x0_full) — DCT-truncated "
                             "low-freq projection of full-res VAE encode. Tests fix "
                             "for VAE-mismatch ceiling.")
    parser.add_argument("--low_target_align", default="none",
                        choices=["none", "std", "norm"],
                        help="Optional scale alignment for --low_target_from_full "
                             "against the native low-res VAE latent.")
    parser.add_argument("--vectorized_atom_gen", action="store_true",
                        help="Vectorize codebook atom generation (saves ~2.3x encode "
                             "time via reduced kernel launches; identical quality + BPP).")
    parser.add_argument("--model_call_skip",
                        choices=["none", "x0_cache", "adaptive_x0_cache"],
                        default="none",
                        help="Decoder skip mode. x0_cache = fixed period (see "
                             "--model_call_period). adaptive_x0_cache = encoder "
                             "always runs Wan, computes per-step u_teacher vs "
                             "u_fast error, transmits 1-bit/step skip mask "
                             "(see --adaptive_skip_threshold).")
    parser.add_argument("--model_call_period", type=int, default=2,
                        help="Skip when (i - stage_start_i) %% period != 0 within a stage.")
    parser.add_argument("--model_call_skip_stages", default="all",
                        help="\"all\" (default) or comma-separated stage indices "
                             "to enable skipping in (e.g. \"0\" = only low stage, "
                             "\"1\" = only full stage).")
    parser.add_argument("--transition_warmup", action="store_true",
                        help="Force an extra Wan call right after every transition "
                             "(i.e. don't skip on i = transition_step + 1).")
    parser.add_argument("--adaptive_skip_threshold", type=float, default=0.1,
                        help="adaptive_x0_cache mode: skip when "
                             "||u_teacher - u_fast|| / ||u_teacher|| < threshold.")
    parser.add_argument("--max_consecutive_skips", type=str, default="1",
                        help="adaptive_x0_cache mode: hard cap on consecutive "
                             "skips. Either a single int (applied to all stages, "
                             "e.g. \"1\" = at most one skip in a row, matching fixed "
                             "period=2's staleness bound) or comma-separated per-stage "
                             "(e.g. \"1,2\" for S=2 means cap=1 in stage 0, cap=2 in "
                             "stage 1). \"999\" effectively disables.")
    parser.add_argument("--boundary_link", action="store_true",
                        help="GOP-link latent correction: encoder transmits a tiny "
                             "boundary residual (z_first_GT - prev_z_last) per GOP "
                             "boundary; decoder applies fade-in over first "
                             "--boundary_fade_k latent frames before VAE decode.")
    parser.add_argument("--boundary_M", type=int, default=64,
                        help="boundary_link: M atoms per boundary residual.")
    parser.add_argument("--boundary_fade_k", type=int, default=4,
                        help="boundary_link: linear fade-in over first k latent "
                             "frames of each GOP after the first.")
    parser.add_argument("--boundary_lowpass", type=int, default=1,
                        help="boundary_link: spatial downsample factor (1=full-res, "
                             ">1 enables variant 3 low-frequency-only correction).")
    # Video config
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=3)
    parser.add_argument("--full_height", type=int, default=720)
    parser.add_argument("--full_width", type=int, default=1280)
    parser.add_argument("--low_height", type=int, default=480)
    parser.add_argument("--low_width", type=int, default=832)
    # Codebook
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384,
                        help="Codebook size for full_res baseline")
    parser.add_argument("--K_low", type=int, default=4096,
                        help="Codebook size for MR low-res stage")
    parser.add_argument("--K_full", type=int, default=16384,
                        help="Codebook size for MR full-res stage")
    # Schedule
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--transition_step", type=int, default=8,
                        help="S=2 only: step index where low→full transition occurs")
    parser.add_argument("--transition_align", default="variance",
                        choices=["variance", "timestep", "none"])
    # Multi-stage (S>=2). When --resolutions is set, overrides --low_*/--full_*
    # and --transition_step / --K_low / --K_full.
    parser.add_argument("--resolutions", default=None,
                        help="Comma-sep ascending list 'HxW,HxW,...', "
                             "e.g. '368x640,480x832,720x1280' for S=3")
    parser.add_argument("--transition_steps", default=None,
                        help="Comma-sep list, length S-1, e.g. '6,11' for S=3")
    parser.add_argument("--K_per_stage", default=None,
                        help="Comma-sep list, length S, e.g. '4096,8192,16384'")
    # SDE / model
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) / args.method
    out.mkdir(parents=True, exist_ok=True)

    is_mr = args.method != "full_res"
    hf_mode = {
        "mr_random":         "random",
        "mr_oracle":         "oracle",
        "mr_codebook":       "codebook",
        "mr_prior_codebook": "prior_codebook",
        "mr_dlc1_pure":      "dlc1_pure",
        "mr_dlc1_codebook":  "dlc1_codebook",
        "mr_dlc2_subband":   "dlc2_subband",
    }.get(args.method, "random")

    # Parse multi-stage spec if provided
    resolutions_list = None
    transition_steps_list = None
    K_per_stage_list = None
    if args.resolutions is not None:
        resolutions_list = [
            tuple(int(x) for x in r.split("x"))
            for r in args.resolutions.split(",")
        ]
        assert all(len(r) == 2 for r in resolutions_list)
        # Top stage = last entry; override full_height/full_width
        args.full_height, args.full_width = resolutions_list[-1]
        # Bottom stage → low_*
        args.low_height, args.low_width = resolutions_list[0]
        S = len(resolutions_list)
        if args.transition_steps is None:
            raise ValueError("--transition_steps required when --resolutions is set")
        transition_steps_list = [int(s) for s in args.transition_steps.split(",")]
        assert len(transition_steps_list) == S - 1
        if args.K_per_stage is not None:
            K_per_stage_list = [int(k) for k in args.K_per_stage.split(",")]
            assert len(K_per_stage_list) == S
        else:
            # default: power-of-2 ramp ending at 16384
            K_per_stage_list = [2 ** (14 - (S - 1 - i)) for i in range(S)]

    HEIGHT, WIDTH = args.full_height, args.full_width
    FPG = args.num_frames

    print("=" * 70)
    print(f"V1 MR-GVCC Experiment — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Method: {args.method}")
    print(f"  Full res: {WIDTH}×{HEIGHT}  Low res: {args.low_width}×{args.low_height}")
    print(f"  GOPs: {args.num_gops} × {FPG} frames")
    print(f"  M={args.M}, steps={args.steps}, ddim_tail={args.ddim_tail}")
    if is_mr:
        if resolutions_list is not None:
            print(f"  resolutions={resolutions_list}")
            print(f"  transition_steps={transition_steps_list}")
            print(f"  K_per_stage={K_per_stage_list}")
        else:
            print(f"  transition_step={args.transition_step}, align={args.transition_align}")
            print(f"  K_low={args.K_low}, K_full={args.K_full}")
    else:
        print(f"  K={args.K}")
    print("=" * 70)

    # Load sequences
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    # Load model
    model = WanT2VDiffusersWrapper(args.wan_ckpt, flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    all_seq_results = []

    for seq_idx, (seq_name, seq_path) in enumerate(all_seqs):
        print(f"\n{'='*70}")
        print(f"  [{seq_idx+1}/{len(all_seqs)}] {seq_name}  ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*70}")

        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Load frames at full resolution
        raw_frames = load_yuv420_frames(seq_path, FPG * args.num_gops)
        frames_full = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames
        save_video_mp4(frames_full[:FPG * args.num_gops], seq_dir / "original.mp4")

        gop_results = []
        prev_z_last = None  # for boundary_link: last decoded latent frame of GOP n-1

        for g in range(args.num_gops):
            gop_frames = frames_full[g * FPG: (g + 1) * FPG]
            if len(gop_frames) < FPG:
                print(f"  Warning: not enough frames for GOP {g}, skipping")
                break

            print(f"\n  --- {seq_name} GOP {g}  ({datetime.now().strftime('%H:%M:%S')}) ---")

            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)

            if is_mr:
                pipe = MRTurboDDCMPipeline(
                    model,
                    resolutions=resolutions_list,
                    transition_steps_list=transition_steps_list,
                    K_per_stage=K_per_stage_list,
                    low_height=args.low_height,
                    low_width=args.low_width,
                    full_height=HEIGHT,
                    full_width=WIDTH,
                    K_low=args.K_low,
                    K_full=args.K_full,
                    M=args.M,
                    num_steps=args.steps,
                    transition_step=args.transition_step,
                    num_ddim_tail=args.ddim_tail,
                    guidance_scale=args.guidance_scale,
                    g_scale=args.g_scale,
                    num_frames=FPG,
                    seed=args.seed,
                    high_freq_mode=hf_mode,
                    transition_align=args.transition_align,
                    dlc2_rank_frac=args.dlc2_rank_frac,
                    low_target_from_full=args.low_target_from_full,
                    vectorized_atom_gen=args.vectorized_atom_gen,
                    low_target_align=args.low_target_align,
                    model_call_skip=args.model_call_skip,
                    model_call_period=args.model_call_period,
                    model_call_skip_stages=args.model_call_skip_stages,
                    transition_warmup=args.transition_warmup,
                    adaptive_skip_threshold=args.adaptive_skip_threshold,
                    boundary_link=args.boundary_link,
                    boundary_M=args.boundary_M,
                    boundary_fade_k=args.boundary_fade_k,
                    boundary_lowpass=args.boundary_lowpass,
                    max_consecutive_skips=args.max_consecutive_skips,
                )
            else:
                pipe = TurboDDCMWanPipeline(
                    model,
                    K=args.K, M=args.M,
                    num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                    guidance_scale=args.guidance_scale, g_scale=args.g_scale,
                    num_frames=FPG,
                    height=HEIGHT, width=WIDTH,
                    seed=args.seed,
                    vectorized_atom_gen=args.vectorized_atom_gen,
                )

            frames_recon, t_enc, t_dec, this_z_last, boundary_used = run_gop(
                pipe, gop_frames, args, is_mr, prev_z_last=prev_z_last,
            )
            prev_z_last = this_z_last  # carry forward to next GOP

            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")

            # Metrics (at full resolution)
            n = min(len(gop_frames), len(frames_recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec)

            total_bits = pipe._total_codebook_bits
            if boundary_used:
                total_bits += pipe._boundary_bits_per_gop
            total_pixels = len(gop_frames) * HEIGHT * WIDTH
            bpp = total_bits / total_pixels
            bitrate_kbps = total_bits / (len(gop_frames) / 16.0) / 1000.0

            result = {
                "sequence": seq_name,
                "gop": g,
                "method": args.method,
                "PSNR_dB": round(mean_psnr, 2),
                "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
                "LPIPS": round(mean_lpips, 4),
                "BPP": round(bpp, 6),
                "total_bits": total_bits,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "encode_s": round(t_enc, 1),
                "decode_s": round(t_dec, 1),
                "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
            }
            if is_mr:
                result["high_freq_mode"] = hf_mode
                result["transition_align"] = args.transition_align
                result["dlc2_rank_frac"] = args.dlc2_rank_frac if hf_mode == "dlc2_subband" else None
                result["low_target_from_full"] = args.low_target_from_full
                result["low_target_align"] = args.low_target_align if args.low_target_from_full else "n/a"
                if resolutions_list is not None:
                    result["resolutions"] = resolutions_list
                    result["transition_steps"] = transition_steps_list
                    result["K_per_stage"] = K_per_stage_list
                    result["S"] = len(resolutions_list)
                else:
                    result["transition_step"] = args.transition_step
                    result["S"] = 2

            gop_results.append(result)
            with open(gop_dir / "metrics.json", "w") as fp:
                json.dump(result, fp, indent=2)

            ms_str = f"{mean_msssim:.4f}" if mean_msssim else "N/A"
            print(f"  PSNR={mean_psnr:.2f} dB  MS-SSIM={ms_str}  LPIPS={mean_lpips:.4f}")
            print(f"  BPP={bpp:.6f}  {bitrate_kbps:.2f} kbps")
            print(f"  Time: enc={t_enc:.0f}s  dec={t_dec:.0f}s")

            del pipe, t_gt, t_rec
            gc.collect()
            torch.cuda.empty_cache()

        if gop_results:
            avg = lambda key: float(np.mean([r[key] for r in gop_results]))
            all_seq_results.append({
                "sequence": seq_name,
                "gop_results": gop_results,
                "avg_PSNR": round(avg("PSNR_dB"), 2),
                "avg_LPIPS": round(avg("LPIPS"), 4),
                "avg_BPP": round(avg("BPP"), 6),
                "avg_encode_s": round(avg("encode_s"), 1),
                "avg_decode_s": round(avg("decode_s"), 1),
            })

        del frames_full
        gc.collect()

    # Summary
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not all_seq_results:
        print("\nNo results.")
        return

    print(f"\n{'='*70}")
    print(f"SUMMARY — {args.method}  M={args.M}")
    print(f"{'='*70}")
    hdr = "{:<12} {:>4} {:>8} {:>7} {:>8} {:>8} {:>8}"
    row = "{:<12} {:>4} {:>7.2f} {:>7.4f} {:>8} {:>7.0f} {:>7.0f}"
    print(hdr.format("Seq", "GOP", "PSNR", "LPIPS", "BPP", "Enc_s", "Dec_s"))
    print("-" * 64)

    for sr in all_seq_results:
        for r in sr["gop_results"]:
            print(row.format(
                r["sequence"], r["gop"], r["PSNR_dB"], r["LPIPS"],
                f"{r['BPP']:.6f}", r["encode_s"], r["decode_s"],
            ))

    print("-" * 64)
    overall_psnr = float(np.mean([sr["avg_PSNR"] for sr in all_seq_results]))
    overall_lpips = float(np.mean([sr["avg_LPIPS"] for sr in all_seq_results]))
    overall_bpp = float(np.mean([sr["avg_BPP"] for sr in all_seq_results]))
    overall_enc = float(np.mean([sr["avg_encode_s"] for sr in all_seq_results]))
    overall_dec = float(np.mean([sr["avg_decode_s"] for sr in all_seq_results]))
    print(
        f"OVERALL       PSNR={overall_psnr:.2f} dB  "
        f"LPIPS={overall_lpips:.4f}  BPP={overall_bpp:.6f}"
    )
    print(f"  avg enc={overall_enc:.0f}s  avg dec={overall_dec:.0f}s")

    summary = {
        "experiment": "V1 MR-GVCC",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "method": args.method,
            "wan_ckpt": args.wan_ckpt,
            "high_freq_mode": hf_mode if is_mr else "n/a",
            "transition_align": args.transition_align if is_mr else "n/a",
            "dlc2_rank_frac": args.dlc2_rank_frac if (is_mr and hf_mode == "dlc2_subband") else None,
            "low_target_from_full": args.low_target_from_full if is_mr else False,
            "low_target_align": args.low_target_align if (is_mr and args.low_target_from_full) else "n/a",
            "full_resolution": f"{WIDTH}×{HEIGHT}",
            "low_resolution": f"{args.low_width}×{args.low_height}" if is_mr else "n/a",
            "transition_step": args.transition_step if is_mr else "n/a",
            "frames_per_gop": FPG,
            "M": args.M,
            "K": args.K if not is_mr else f"K_low={args.K_low},K_full={args.K_full}",
            "steps": args.steps,
            "ddim_tail": args.ddim_tail,
            "g_scale": args.g_scale,
            "S": (len(resolutions_list) if resolutions_list is not None else (2 if is_mr else 1)),
            "resolutions": resolutions_list,
            "transition_steps": transition_steps_list,
            "K_per_stage": K_per_stage_list,
        },
        "sequences": all_seq_results,
        "overall": {
            "PSNR_dB": round(overall_psnr, 2),
            "LPIPS": round(overall_lpips, 4),
            "BPP": round(overall_bpp, 6),
            "avg_encode_s": round(overall_enc, 1),
            "avg_decode_s": round(overall_dec, 1),
        },
    }
    with open(out / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
