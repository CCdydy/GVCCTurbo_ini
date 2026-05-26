"""run_chained.py — Wan2.2-Fun-5B GVCC chained multi-GOP runner.

For FLF2V, each GOP takes first+last frame from GT. We use:
  - GOP 0..k: first = GT[g*FPG], last = GT[(g+1)*FPG-1]  (independent FLF2V per GOP)

Supports tail_residual + x0_cache + chained-first-frame-reuse (AR for first frame only).
"""
import sys, os, re, time, gc, zlib, json, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_wan22_fun")
sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc")

from wan22_fun_wrapper import Wan22FunWrapper
from wan22_fun_pipeline import Wan22FunGVCCPipeline
from uvg_data import find_uvg_sequence
import lpips
from pytorch_msssim import ms_ssim
from run_smoke import load_yuv420_frames, frames_to_tensor, psnr, lpips_v

UVG_DIR = "/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg"
FPG = 33
HEIGHT, WIDTH = 704, 1280


def compress_tail_residual(residual_tail, bits=4):
    """Per-channel min/max quantize + zlib."""
    r = residual_tail.float().cpu()
    C, F, H, W = r.shape
    maxval = r.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    r_norm = r / maxval
    levels = 2 ** bits
    q = ((r_norm + 1.0) * (levels - 1) / 2.0).round().clamp(0, levels - 1)
    packed = q.to(torch.uint8).numpy().tobytes() if bits <= 8 else q.to(torch.int16).numpy().tobytes()
    comp = zlib.compress(packed, 9)
    meta = {'C': C, 'F': F, 'H': H, 'W': W, 'bits': bits, 'maxval': maxval.squeeze().tolist()}
    return comp, len(comp), meta


def decompress_tail_residual(compressed, meta, device='cuda'):
    C, F, H, W = meta['C'], meta['F'], meta['H'], meta['W']
    bits = meta['bits']
    mv = meta['maxval']
    if isinstance(mv, (int, float)): mv = [mv]
    maxval = torch.tensor(mv).float().reshape(C, 1, 1, 1)
    packed = zlib.decompress(compressed)
    if bits <= 8:
        arr = np.frombuffer(packed, dtype=np.uint8).copy()
        q = torch.from_numpy(arr).float().reshape(C, F, H, W)
    else:
        arr = np.frombuffer(packed, dtype=np.int16).copy()
        q = torch.from_numpy(arr).float().reshape(C, F, H, W)
    levels = 2 ** bits
    r_norm = q * 2.0 / (levels - 1) - 1.0
    return (r_norm * maxval).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Jockey")
    ap.add_argument("--num_gops", type=int, default=3)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ddim_tail", type=int, default=3)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--g_scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_call_skip", default="none",
                    choices=["none", "u_cache", "x0_cache"])
    ap.add_argument("--model_call_period", type=int, default=2)
    ap.add_argument("--all_frames_atom", action="store_true",
                    help="Encode atoms for ALL latent frames including frame 0 "
                         "(matches original Wan2.1-FLF2V GVCC convention)")
    ap.add_argument("--tail_residual_bits", type=int, default=0,
                    help="0 disables tail residual")
    ap.add_argument("--tail_residual_frames", type=int, default=0,
                    help="how many trailing latent frames to compress as residual")
    ap.add_argument("--chained_first_frame", action="store_true",
                    help="GOP k>0 uses decoded last of GOP k-1 as first (saves ref bytes)")
    ap.add_argument("--out", default="/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_wan22_fun/out_chained")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    yuv = find_uvg_sequence(UVG_DIR, args.seq)
    assert yuv, f"{args.seq} not found"
    total_needed = args.num_gops * FPG
    print(f"[INFO] Loading {total_needed} frames of {args.seq}")
    raw = load_yuv420_frames(yuv, total_needed, args.start_frame)
    frames_all = [f.resize((WIDTH, HEIGHT), Image.LANCZOS) for f in raw]
    print(f"[INFO] {len(frames_all)} frames @ {WIDTH}x{HEIGHT}")

    w = Wan22FunWrapper()
    pipe = Wan22FunGVCCPipeline(
        w, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
        guidance_scale=args.cfg, g_scale=args.g_scale,
        num_frames=FPG, height=HEIGHT, width=WIDTH, seed=args.seed,
        model_call_skip=args.model_call_skip,
        model_call_period=args.model_call_period,
        skip_frame0=not args.all_frames_atom,
    )

    use_tail = args.tail_residual_bits > 0 and args.tail_residual_frames > 0
    next_first = None
    all_recon = []
    gop_results = []
    total_enc, total_dec = 0.0, 0.0

    for g in range(args.num_gops):
        gop_frames = frames_all[g*FPG:(g+1)*FPG]
        if args.chained_first_frame and g > 0:
            first_image = next_first
            first_src = f"AR(decoded_last[GOP{g-1}])"
        else:
            first_image = gop_frames[0]
            first_src = "GT[0]"
        last_image = gop_frames[-1]

        print(f"\n--- GOP {g} (first={first_src}, last=GT) ---")
        t0 = time.time()
        step_data, x0_enc = pipe.encode(gop_frames, prompt="",
                                         first_image=first_image, last_image=last_image)
        t_enc = time.time() - t0
        total_enc += t_enc

        tail_bytes = 0
        correction = None
        if use_tail:
            x0_true = pipe._gt_latent
            n_tail = args.tail_residual_frames
            tail_res = (x0_true - x0_enc).squeeze(0)[:, -n_tail:, :, :]
            comp, tail_bytes, meta = compress_tail_residual(tail_res, args.tail_residual_bits)
            tail_dec = decompress_tail_residual(comp, meta, device="cuda")
            correction = torch.zeros_like(x0_enc.squeeze(0))
            correction[:, -n_tail:, :, :] = tail_dec
            correction = correction.unsqueeze(0)

        t0 = time.time()
        recon = pipe.decode(step_data, prompt="",
                             first_image=first_image, last_image=last_image,
                             latent_correction=correction)
        t_dec = time.time() - t0
        total_dec += t_dec

        recon = [f.crop((0, 0, WIDTH, HEIGHT)) if f.size != (WIDTH, HEIGHT) else f for f in recon]
        next_first = recon[-1].copy()  # for AR chain
        all_recon.extend(recon)

        n = min(len(gop_frames), len(recon))
        t_gt = frames_to_tensor(gop_frames[:n])
        t_rec = frames_to_tensor(recon[:n])
        p_mean, per_p = psnr(t_gt, t_rec)
        msssim = ms_ssim(t_gt, t_rec, data_range=1.0).item()
        lp = lpips_v(t_gt, t_rec)

        cb_bytes = pipe._total_codebook_bits // 8
        total_bytes = cb_bytes + tail_bytes
        kbps = total_bytes * 8 / (FPG / 16.0) / 1000.0
        bpp  = total_bytes * 8 / (FPG * HEIGHT * WIDTH)

        print(f"  PSNR={p_mean:.2f} (first={per_p[0].item():.2f}, last={per_p[-1].item():.2f})  "
              f"MS-SSIM={msssim:.4f}  LPIPS={lp:.4f}")
        print(f"  cb={cb_bytes} + tail={tail_bytes} = {total_bytes}B  "
              f"BPP={bpp:.4f}  {kbps:.0f} kbps")
        print(f"  Time: enc={t_enc:.1f}s + dec={t_dec:.1f}s = {t_enc+t_dec:.1f}s")
        gop_results.append({
            "gop": g, "first_src": first_src,
            "PSNR": p_mean, "first_PSNR": per_p[0].item(), "last_PSNR": per_p[-1].item(),
            "MS_SSIM": msssim, "LPIPS": lp,
            "cb_bytes": cb_bytes, "tail_bytes": tail_bytes,
            "kbps": kbps, "BPP": bpp,
            "enc_s": t_enc, "dec_s": t_dec,
        })

    # Aggregate
    gt_full = frames_all[:args.num_gops*FPG]
    p_all, _ = psnr(frames_to_tensor(gt_full), frames_to_tensor(all_recon))
    avg_kbps = np.mean([r["kbps"] for r in gop_results])
    video_s = args.num_gops * FPG / 16.0

    print(f"\n{'='*60}")
    print(f"CHAINED {args.num_gops}-GOP — {args.seq}")
    print(f"  skip={args.model_call_skip} period={args.model_call_period}  "
          f"tail={args.tail_residual_bits}bit×{args.tail_residual_frames}f  "
          f"chained_first={args.chained_first_frame}")
    print(f"{'='*60}")
    print(f"Avg PSNR over {args.num_gops} GOPs = {p_all:.2f} dB")
    print(f"Avg bitrate              = {avg_kbps:.0f} kbps")
    print(f"Total wall-clock         = {total_enc + total_dec:.1f}s for {video_s:.2f}s video")

    import cv2
    def write_mp4(arr_list, path, fps=16):
        arr = np.stack([np.asarray(f) for f in arr_list])
        T, H, W, _ = arr.shape
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for frm in arr: vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        vw.release()
    write_mp4(gt_full, os.path.join(args.out, "gt_full.mp4"))
    write_mp4(all_recon, os.path.join(args.out, "recon_full.mp4"))
    side = [np.concatenate([np.asarray(g), np.asarray(r)], axis=1)
            for g, r in zip(gt_full, all_recon)]
    write_mp4([Image.fromarray(s) for s in side], os.path.join(args.out, "side_full.mp4"))
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"seq": args.seq, "num_gops": args.num_gops,
                   "avg_PSNR": float(p_all), "avg_kbps": float(avg_kbps),
                   "total_wall_s": total_enc + total_dec, "video_s": video_s,
                   "per_gop": gop_results, "args": vars(args)}, fh, indent=2)
    print(f"Saved videos+metrics to {args.out}")


if __name__ == "__main__":
    main()
