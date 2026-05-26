"""run_chained.py — autoregressive chained-GOP MobileI2V-GVCC.

GOP 0: ref = GT first frame (free)
GOP k>0: ref = decoded last frame from GOP k-1 (0 transmission bytes)

Uses the 20-step + x0_cache p=2 + full 4-bit residual config by default.
"""
import sys, os, re, time, gc, zlib, json
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_mobilei2v")
sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc")

from mobilei2v_wrapper import MobileI2VWrapper
from mobilei2v_pipeline import MobileI2VGVCCPipeline
from uvg_data import find_uvg_sequence
import lpips
from pytorch_msssim import ms_ssim
from run_smoke import (
    load_yuv420_frames, frames_to_tensor, compute_psnr, compute_msssim,
    compute_lpips, compress_tail_residual, decompress_tail_residual,
)

UVG_DIR = "/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg"
FPG = 17
HEIGHT, WIDTH = 720, 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Jockey")
    ap.add_argument("--num_gops", type=int, default=3)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ddim_tail", type=int, default=3)
    ap.add_argument("--g_scale", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--flow_score", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_call_skip", default="x0_cache",
                    choices=["none", "u_cache", "x0_cache"])
    ap.add_argument("--model_call_period", type=int, default=2)
    ap.add_argument("--no_tail_residual", action="store_true")
    ap.add_argument("--tail_residual_bits", type=int, default=4)
    ap.add_argument("--tail_residual_frames", type=int, default=2)
    ap.add_argument("--out", default="/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_mobilei2v/out_chained")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    yuv_path = find_uvg_sequence(UVG_DIR, args.seq)
    assert yuv_path, f"{args.seq} not found"

    total_frames_needed = args.num_gops * FPG
    print(f"[INFO] Loading {total_frames_needed} frames from {args.seq} "
          f"[{args.start_frame}, {args.start_frame+total_frames_needed})")
    raw = load_yuv420_frames(yuv_path, total_frames_needed, args.start_frame)
    frames_all = [f.resize((WIDTH, HEIGHT), Image.LANCZOS) for f in raw]
    print(f"[INFO] {len(frames_all)} frames @ {WIDTH}x{HEIGHT}")

    # Build wrapper + pipeline (single-load, reuse across GOPs)
    w = MobileI2VWrapper(flow_score=args.flow_score)
    pipe = MobileI2VGVCCPipeline(
        w, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
        guidance_scale=args.cfg, g_scale=args.g_scale, seed=args.seed,
        flow_score=args.flow_score,
        model_call_skip=args.model_call_skip,
        model_call_period=args.model_call_period,
    )

    # AR loop
    next_ref = None
    all_recon = []
    gop_results = []

    for g in range(args.num_gops):
        gop_frames = frames_all[g*FPG:(g+1)*FPG]
        if g == 0:
            ref_image = gop_frames[0]
            ref_src = "GT[0]"
        else:
            ref_image = next_ref
            ref_src = f"AR(decoded_last[GOP{g-1}])"

        print(f"\n--- GOP {g} ({ref_src}) ---")
        t0 = time.time()
        step_data, x0_enc = pipe.encode(gop_frames, prompt="", ref_image=ref_image)
        t_enc = time.time() - t0

        # Tail residual
        tail_bytes = 0
        correction = None
        if not args.no_tail_residual:
            x0_true = pipe._gt_latent
            n_tail = args.tail_residual_frames
            tail_res = (x0_true - x0_enc).squeeze(0)[:, -n_tail:, :, :]
            comp, tail_bytes, meta = compress_tail_residual(tail_res, args.tail_residual_bits)
            tail_dec = decompress_tail_residual(comp, meta, device="cuda")
            correction = torch.zeros_like(x0_enc.squeeze(0))
            correction[:, -n_tail:, :, :] = tail_dec
            correction = correction.unsqueeze(0)

        t0 = time.time()
        recon = pipe.decode(step_data, prompt="", ref_image=ref_image,
                            latent_correction=correction)
        t_dec = time.time() - t0

        # Crop and metrics
        recon = [f.crop((0, 0, WIDTH, HEIGHT)) if f.size != (WIDTH, HEIGHT) else f
                 for f in recon]
        next_ref = recon[-1].copy()
        all_recon.extend(recon)

        n = min(len(gop_frames), len(recon))
        t_gt = frames_to_tensor(gop_frames[:n])
        t_rec = frames_to_tensor(recon[:n])
        psnr, per_psnr = compute_psnr(t_gt, t_rec)
        msssim = compute_msssim(t_gt, t_rec)
        lp = compute_lpips(t_gt, t_rec)

        cb_bytes = pipe._total_codebook_bits // 8
        total_bytes = cb_bytes + tail_bytes
        bpp = total_bytes * 8 / (FPG * HEIGHT * WIDTH)
        kbps = total_bytes * 8 / (FPG / 16.0) / 1000.0

        print(f"  PSNR={psnr:.2f} (last={per_psnr[-1].item():.2f})  "
              f"MS-SSIM={msssim:.4f}  LPIPS={lp:.4f}")
        print(f"  Bytes: cb={cb_bytes} + tail={tail_bytes} = {total_bytes}  "
              f"BPP={bpp:.4f}  {kbps:.0f} kbps")
        print(f"  Time: enc={t_enc:.1f}s + dec={t_dec:.1f}s = {t_enc+t_dec:.1f}s")

        gop_results.append({
            "gop": g, "ref": ref_src, "PSNR": psnr, "MS_SSIM": msssim, "LPIPS": lp,
            "last_PSNR": per_psnr[-1].item(),
            "cb_bytes": cb_bytes, "tail_bytes": tail_bytes,
            "kbps": kbps, "BPP": bpp,
            "enc_s": t_enc, "dec_s": t_dec,
        })

    # Full video metrics
    gt_full = frames_all[:args.num_gops*FPG]
    psnr_all, _ = compute_psnr(frames_to_tensor(gt_full), frames_to_tensor(all_recon))
    avg_kbps = np.mean([r["kbps"] for r in gop_results])
    total_time = sum(r["enc_s"] + r["dec_s"] for r in gop_results)
    total_video_s = args.num_gops * FPG / 16.0

    print(f"\n{'='*60}")
    print(f"CHAINED {args.num_gops}-GOP — {args.seq}")
    print(f"{'='*60}")
    print(f"Avg PSNR over {args.num_gops} GOPs = {psnr_all:.2f} dB")
    print(f"Avg bitrate              = {avg_kbps:.0f} kbps")
    print(f"Total wall-clock         = {total_time:.1f}s for {total_video_s:.2f}s video")

    # Save full video
    import cv2
    def write_mp4(arr_list, path, fps=16):
        arr = np.stack([np.asarray(f) for f in arr_list])
        T, H, W, _ = arr.shape
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for frm in arr:
            vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        vw.release()

    write_mp4(gt_full, os.path.join(args.out, "gt_full.mp4"))
    write_mp4(all_recon, os.path.join(args.out, "recon_full.mp4"))

    # Side-by-side
    side = []
    for gt, rc in zip(gt_full, all_recon):
        side.append(np.concatenate([np.asarray(gt), np.asarray(rc)], axis=1))
    arr = np.stack(side)
    T, H, W2, _ = arr.shape
    vw = cv2.VideoWriter(os.path.join(args.out, "side_by_side_full.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 16, (W2, H))
    for frm in arr:
        vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
    vw.release()

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({
            "seq": args.seq, "num_gops": args.num_gops,
            "avg_PSNR": float(psnr_all), "avg_kbps": float(avg_kbps),
            "total_wall_s": total_time, "video_s": total_video_s,
            "per_gop": gop_results, "args": vars(args),
        }, fh, indent=2)
    print(f"Saved gt_full.mp4 / recon_full.mp4 / side_by_side_full.mp4 to {args.out}")


if __name__ == "__main__":
    main()
