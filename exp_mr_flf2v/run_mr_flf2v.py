"""run_mr_flf2v.py — Multi-Resolution FLF2V GVCC.

Combines:
  - Wan2.1-FLF2V-14B-720P as backbone (FLF2V conditioning via clip_fea + y dict)
  - MultiResTurboDDCMWanPipeline (S-stage progressive resolution from T2V MR FINAL)
  - Per-stage FLF2V conditioning re-encoded at each stage's resolution
  - x0_cache p=2 + transition_warmup (the T2V MR FINAL acceleration stack)

Default config matches FLF2V FINAL convention:
  - GT boundary frames (free)
  - 33-frame GOP @ 720P top stage, 480P low stage
  - vec + x0_cache p=2
"""
import sys, os, re, time, gc, argparse, json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper
from sde_rf_wan.mr_pipeline import MRTurboDDCMPipeline
from uvg_data import find_uvg_sequence
from exp_flf2v.run_flf2v_experiment import (
    load_yuv420_frames, compress_boundary_frame, frames_to_tensor,
    compute_psnr, compute_msssim, compute_lpips, save_video_mp4,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    ap.add_argument("--wan_ckpt",
                    default="/home/rog/Desktop/gvcc_turbo/checkpoints/Wan2.1-FLF2V-14B-720P")
    ap.add_argument("--output_dir", default="./exp_mr_flf2v/results")
    ap.add_argument("--sequences", nargs="*", default=["Jockey"])
    ap.add_argument("--num_gops", type=int, default=1)
    ap.add_argument("--num_frames_per_gop", type=int, default=33)
    # Multi-stage settings — default S=2 like T2V FINAL
    ap.add_argument("--resolutions", default="480x854,720x1280",
                    help="comma-separated H×W per stage, ascending")
    ap.add_argument("--transition_steps", default="8",
                    help="step indices where stage transitions occur (S-1 values)")
    ap.add_argument("--K_per_stage", default="16384,16384",
                    help="K (codebook size) per stage")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ddim_tail", type=int, default=3)
    ap.add_argument("--g_scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flow_shift", type=float, default=5.0)
    # T2V FINAL accel stack
    ap.add_argument("--method", default="mr_dlc1_codebook",
                    help="MR method (mr_dlc1_codebook matches T2V FINAL lp_raw)")
    ap.add_argument("--low_target_from_full", action="store_true", default=True,
                    help="lp_raw fix — T2V FINAL uses this. Default ON.")
    ap.add_argument("--vectorized_atom_gen", action="store_true", default=True)
    ap.add_argument("--model_call_skip", default="x0_cache",
                    choices=["none", "u_cache", "x0_cache"])
    ap.add_argument("--model_call_period", type=int, default=2)
    ap.add_argument("--transition_warmup", action="store_true", default=True)
    ap.add_argument("--ref_codec", default="gt",
                    choices=["compressai", "webp", "gt"])
    ap.add_argument("--ref_quality", type=int, default=4)
    args = ap.parse_args()

    # Parse resolutions
    resolutions = [tuple(int(x) for x in r.split("x")) for r in args.resolutions.split(",")]
    transition_steps = [int(x) for x in args.transition_steps.split(",") if x] if args.transition_steps else []
    K_per_stage = [int(x) for x in args.K_per_stage.split(",")]
    S = len(resolutions)
    assert len(transition_steps) == S - 1
    assert len(K_per_stage) == S

    FPG = args.num_frames_per_gop
    HEIGHT, WIDTH = resolutions[-1]  # top stage = final output res
    frames_per_gop_excl_first = FPG - 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MR-FLF2V GVCC — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stages (S={S}): {resolutions}")
    print(f"  Transition steps: {transition_steps}")
    print(f"  K per stage: {K_per_stage}, M={args.M}, steps={args.steps}")
    print(f"  Method: {args.method}  lp_raw={args.low_target_from_full}")
    print(f"  Accel: vec={args.vectorized_atom_gen}  "
          f"skip={args.model_call_skip}/p={args.model_call_period}  "
          f"warmup={args.transition_warmup}")
    print("=" * 70)

    # ----- Load FLF2V model -----
    model = WanFLF2VWrapper(checkpoint_dir=args.wan_ckpt, flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    # ----- Process each sequence -----
    all_results = []
    for seq_name in args.sequences:
        seq_dir = out_dir / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        yuv = find_uvg_sequence(args.data_dir, seq_name)
        assert yuv, f"{seq_name} not found in {args.data_dir}"
        # num_gops=0 means "all available frames"
        if args.num_gops <= 0:
            raw_all = load_yuv420_frames(yuv, 99999)  # load all
            actual_gops = (len(raw_all) - 1) // frames_per_gop_excl_first
            total_unique = frames_per_gop_excl_first * actual_gops + 1
            raw = raw_all[:total_unique]
        else:
            actual_gops = args.num_gops
            total_unique = frames_per_gop_excl_first * actual_gops + 1
            raw = load_yuv420_frames(yuv, total_unique)
        frames = [f.resize((WIDTH, HEIGHT), Image.LANCZOS) for f in raw]
        print(f"\n[{seq_name}] {len(frames)} frames @ {WIDTH}×{HEIGHT}  ({actual_gops} GOPs)")

        # GOP splits (first frame shared across adjacent GOPs)
        gops = []
        for g in range(actual_gops):
            start = g * frames_per_gop_excl_first
            gops.append(frames[start:start+FPG])

        # ----- Boundary frame compression (shared boundaries) -----
        boundary_indices = [g * frames_per_gop_excl_first for g in range(actual_gops + 1)]
        boundary_indices = [min(idx, len(frames)-1) for idx in boundary_indices]
        boundary_compressed = {}
        total_boundary_bytes = 0
        for idx in boundary_indices:
            gt = frames[idx]
            if args.ref_codec == "gt":
                boundary_compressed[idx] = (gt, 0)
            else:
                dec, nb = compress_boundary_frame(gt, args.ref_codec, args.ref_quality)
                boundary_compressed[idx] = (dec, nb)
                total_boundary_bytes += nb

        # ----- Per-GOP -----
        gop_results = []
        all_recon = []
        for g in range(actual_gops):
            gop_frames = gops[g]
            first_idx = g * frames_per_gop_excl_first
            last_idx = (g + 1) * frames_per_gop_excl_first
            first_decoded, first_bytes = boundary_compressed[first_idx]
            last_decoded, last_bytes = boundary_compressed[last_idx]
            gop_first_bytes = 0 if g > 0 else first_bytes  # AR sharing
            print(f"\n--- {seq_name} GOP {g}/{actual_gops-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            print(f"  first={'GT' if args.ref_codec=='gt' else args.ref_codec}({first_bytes}B "
                  f"{'reused' if g>0 else 'new'})  last={'GT' if args.ref_codec=='gt' else args.ref_codec}({last_bytes}B)")

            # Build pipeline ONCE per GOP (reuses model for velocity)
            pipe = MRTurboDDCMPipeline(
                model,
                resolutions=resolutions,
                transition_steps_list=transition_steps,
                K_per_stage=K_per_stage,
                M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=1.0, g_scale=args.g_scale,
                num_frames=FPG, seed=args.seed,
                high_freq_mode=args.method.replace("mr_", ""),
                low_target_from_full=args.low_target_from_full,
                vectorized_atom_gen=args.vectorized_atom_gen,
                model_call_skip=args.model_call_skip,
                model_call_period=args.model_call_period,
                transition_warmup=args.transition_warmup,
            )

            # ----- Per-stage FLF2V conditioning (re-encode first/last at each stage's res) -----
            # FLF2V wrapper's encode_first_last_frames takes (first, last, F, H, W).
            # We need to call it per stage because the latent shape depends on resolution.
            t0 = time.time()
            i2v_per_stage = []
            for (h, w) in resolutions:
                # Resize first/last to this stage's resolution
                f1 = first_decoded.resize((w, h), Image.LANCZOS)
                f2 = last_decoded.resize((w, h), Image.LANCZOS)
                cond = model.encode_first_last_frames(f1, f2, FPG, h, w)
                i2v_per_stage.append(cond)
            t_cond = time.time() - t0
            print(f"  Built per-stage FLF2V cond ({len(i2v_per_stage)} stages, {t_cond:.1f}s)")

            # ----- Encode -----
            t0 = time.time()
            step_data, trans_atoms, x0_enc = pipe.encode(
                gop_frames, prompt="",
                precomputed_i2v_cond_per_stage=i2v_per_stage,
            )
            t_enc = time.time() - t0

            # ----- Decode -----
            t0 = time.time()
            recon = pipe.decode(
                step_data, trans_atoms, prompt="",
                latent_correction=None,
                precomputed_i2v_cond_per_stage=i2v_per_stage,
            )
            t_dec = time.time() - t0

            # Metrics
            n = min(len(gop_frames), len(recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(recon[:n])
            psnr_mean, per_psnr = compute_psnr(t_gt, t_rec)
            msssim = compute_msssim(t_gt, t_rec)
            lpips_v = compute_lpips(t_gt, t_rec)

            # Bitrate
            cb_bits = pipe._total_codebook_bits
            cb_bytes = cb_bits // 8
            gop_total_bytes = cb_bytes + gop_first_bytes + last_bytes
            total_pixels = len(gop_frames) * HEIGHT * WIDTH
            bpp = gop_total_bytes * 8 / total_pixels
            kbps = gop_total_bytes * 8 / (FPG / 16.0) / 1000.0

            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(recon, gop_dir / "reconstructed.mp4")
            first_decoded.save(gop_dir / "first_used.png")
            last_decoded.save(gop_dir / "last_used.png")

            all_recon.extend(recon if g == 0 else recon[1:])

            print(f"  PSNR={psnr_mean:.2f} dB  MS-SSIM={msssim:.4f}  LPIPS={lpips_v:.4f}")
            print(f"  cb={cb_bytes}B + bound={gop_first_bytes+last_bytes}B = "
                  f"{gop_total_bytes}B  BPP={bpp:.6f}  {kbps:.2f} kbps")
            print(f"  Time: cond={t_cond:.1f}s + enc={t_enc:.1f}s + dec={t_dec:.1f}s")

            gop_results.append({
                "gop": g, "PSNR": psnr_mean, "MS_SSIM": msssim, "LPIPS": lpips_v,
                "cb_bytes": cb_bytes, "boundary_bytes": gop_first_bytes + last_bytes,
                "total_bytes": gop_total_bytes, "BPP": bpp, "kbps": kbps,
                "cond_s": t_cond, "enc_s": t_enc, "dec_s": t_dec,
            })

            del pipe, step_data, trans_atoms, recon, t_gt, t_rec
            gc.collect()
            torch.cuda.empty_cache()

        # Save full video
        save_video_mp4(all_recon, seq_dir / "recon_full.mp4")
        all_results.extend([{**r, "sequence": seq_name} for r in gop_results])

    # Summary
    print("\n" + "=" * 70)
    print("MR-FLF2V SUMMARY")
    print("=" * 70)

    # Per-sequence aggregates
    by_seq = {}
    for r in all_results:
        by_seq.setdefault(r["sequence"], []).append(r)
    print(f"{'Sequence':<14} {'PSNR':>6} {'MS-SSIM':>8} {'LPIPS':>7} {'BPP':>8} {'kbps':>7} "
          f"{'enc(s)':>7} {'dec(s)':>7}")
    print("-" * 72)
    for seq, rs in by_seq.items():
        print(f"{seq:<14} "
              f"{np.mean([r['PSNR'] for r in rs]):>6.2f} "
              f"{np.mean([r['MS_SSIM'] for r in rs]):>8.4f} "
              f"{np.mean([r['LPIPS'] for r in rs]):>7.4f} "
              f"{np.mean([r['BPP'] for r in rs]):>8.5f} "
              f"{np.mean([r['kbps'] for r in rs]):>7.2f} "
              f"{np.mean([r['enc_s'] for r in rs]):>7.1f} "
              f"{np.mean([r['dec_s'] for r in rs]):>7.1f}")
    print("-" * 72)
    avg_psnr = np.mean([r["PSNR"] for r in all_results])
    avg_msssim = np.mean([r["MS_SSIM"] for r in all_results])
    avg_lpips = np.mean([r["LPIPS"] for r in all_results])
    avg_bpp = np.mean([r["BPP"] for r in all_results])
    avg_kbps = np.mean([r["kbps"] for r in all_results])
    avg_enc = np.mean([r["enc_s"] for r in all_results])
    avg_dec = np.mean([r["dec_s"] for r in all_results])
    print(f"{'OVERALL':<14} "
          f"{avg_psnr:>6.2f} {avg_msssim:>8.4f} {avg_lpips:>7.4f} {avg_bpp:>8.5f} "
          f"{avg_kbps:>7.2f} {avg_enc:>7.1f} {avg_dec:>7.1f}")
    print(f"Total wall-clock: enc {avg_enc:.1f}s + dec {avg_dec:.1f}s = "
          f"{avg_enc+avg_dec:.1f}s/GOP × {len(all_results)} GOPs = "
          f"{(avg_enc+avg_dec)*len(all_results)/60:.1f} min")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "args": vars(args),
            "per_gop": all_results,
            "avg_PSNR": float(avg_psnr),
            "avg_MS_SSIM": float(avg_msssim),
            "avg_LPIPS": float(avg_lpips),
            "avg_BPP": float(avg_bpp),
            "avg_kbps": float(avg_kbps),
            "avg_enc_s": float(avg_enc),
            "avg_dec_s": float(avg_dec),
        }, fh, indent=2)


if __name__ == "__main__":
    main()
