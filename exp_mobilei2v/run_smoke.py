"""run_smoke.py — MobileI2V-GVCC smoke test on 1 UVG sequence × 1 GOP (17 frames).

Loads first 17 frames of a UVG sequence, encodes via DDCM SDE + first-frame
overwrite + (optional) tail residual, decodes, computes PSNR / MS-SSIM / LPIPS / BPP.
"""
import sys, os, re, time, gc, zlib
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

UVG_DIR = "/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg"
FPG = 17  # MobileI2V fixed
HEIGHT, WIDTH = 720, 1280


def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    import cv2
    m = re.search(r'(\d+)x(\d+)', os.path.basename(yuv_path))
    W, H = int(m.group(1)), int(m.group(2))
    frame_size = H * W * 3 // 2
    frames = []
    with open(yuv_path, 'rb') as f:
        f.seek(start_frame * frame_size)
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:H*W].reshape(H, W)
            u = yuv[H*W:H*W + H*W//4].reshape(H//2, W//2)
            v = yuv[H*W + H*W//4:].reshape(H//2, W//2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(np.stack([y, u, v], -1), cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames


def frames_to_tensor(frames):
    """List[PIL] → (T, 3, H, W) fp32 in [0, 1]."""
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def compute_psnr(a, b):
    """Per-frame PSNR. a,b are (T,3,H,W) in [0,1]."""
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-12))
    return psnr.mean().item(), psnr


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0).item()


_LPIPS = None
def compute_lpips(a, b):
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net='alex').cuda()
    a_n = (a * 2 - 1).cuda()
    b_n = (b * 2 - 1).cuda()
    with torch.no_grad():
        return _LPIPS(a_n, b_n).mean().item()


def compress_tail_residual(residual_tail, bits=8):
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
    if isinstance(mv, (int, float)):
        mv = [mv]
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
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--ddim_tail", type=int, default=3)
    ap.add_argument("--g_scale", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--flow_score", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tail_residual", action="store_true")
    ap.add_argument("--tail_residual_bits", type=int, default=8)
    ap.add_argument("--tail_residual_frames", type=int, default=1,
                    help="how many latent frames (from the end) to apply residual to")
    ap.add_argument("--model_call_skip", choices=["none", "u_cache", "x0_cache"],
                    default="none")
    ap.add_argument("--model_call_period", type=int, default=2)
    ap.add_argument("--model_call_skip_until", type=int, default=-1)
    ap.add_argument("--out", default="/tmp/mobilei2v_smoke")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading UVG {args.seq} frames [{args.start_frame}, {args.start_frame+FPG})")
    yuv_path = find_uvg_sequence(UVG_DIR, args.seq)
    if yuv_path is None:
        print(f"Sequence {args.seq} not found in {UVG_DIR}")
        return
    raw = load_yuv420_frames(yuv_path, FPG, args.start_frame)
    frames = [f.resize((WIDTH, HEIGHT), Image.LANCZOS) for f in raw]
    print(f"[INFO] {len(frames)} frames @ {WIDTH}x{HEIGHT}")

    # Build wrapper + pipeline
    w = MobileI2VWrapper(flow_score=args.flow_score)
    pipe = MobileI2VGVCCPipeline(
        w, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
        guidance_scale=args.cfg, g_scale=args.g_scale, seed=args.seed,
        flow_score=args.flow_score,
        model_call_skip=args.model_call_skip,
        model_call_period=args.model_call_period,
        model_call_skip_until=args.model_call_skip_until,
    )

    # Encode
    print(f"\n[ENCODE]")
    t0 = time.time()
    step_data, x0_enc = pipe.encode(frames, prompt="", ref_image=frames[0])
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
        print(f"  Tail residual: {tail_bytes}B ({args.tail_residual_bits}bit × {n_tail}f)")

    # Decode
    print(f"\n[DECODE]")
    t0 = time.time()
    recon = pipe.decode(step_data, prompt="", ref_image=frames[0],
                        latent_correction=correction)
    t_dec = time.time() - t0

    # Crop reconstructed frames to original WIDTHxHEIGHT (VAE rounds up height)
    recon = [f.crop((0, 0, WIDTH, HEIGHT)) if f.size != (WIDTH, HEIGHT) else f
             for f in recon]
    n = min(len(frames), len(recon))
    print(f"  recon[0].size = {recon[0].size}, GT[0].size = {frames[0].size}")

    # Metrics
    t_gt = frames_to_tensor(frames[:n])
    t_rec = frames_to_tensor(recon[:n])
    psnr_mean, per_psnr = compute_psnr(t_gt, t_rec)
    msssim = compute_msssim(t_gt, t_rec)
    lpips_v = compute_lpips(t_gt, t_rec)
    last_psnr = per_psnr[-1].item()

    # Bitrate
    cb_bytes = pipe._total_codebook_bits // 8
    total_bytes = cb_bytes + tail_bytes  # GOP 0: no ref bytes (free)
    bpp = total_bytes * 8 / (FPG * HEIGHT * WIDTH)
    bitrate_kbps = total_bytes * 8 / (FPG / 16.0) / 1000.0

    print(f"\n{'='*60}")
    print(f"RESULT — {args.seq} GOP0 (frames {args.start_frame}–{args.start_frame+FPG-1})")
    print(f"{'='*60}")
    print(f"PSNR        = {psnr_mean:.2f} dB  (last frame {last_psnr:.2f})")
    print(f"MS-SSIM     = {msssim:.4f}")
    print(f"LPIPS       = {lpips_v:.4f}")
    print(f"Per-frame PSNR = {[round(p.item(),2) for p in per_psnr]}")
    print(f"Bytes       = {cb_bytes} (codebook) + {tail_bytes} (tail) = {total_bytes}")
    print(f"BPP         = {bpp:.6f}")
    print(f"Bitrate     = {bitrate_kbps:.2f} kbps")
    print(f"Time        = enc {t_enc:.1f}s + dec {t_dec:.1f}s")

    # Save outputs
    import json
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "seq": args.seq, "PSNR": psnr_mean, "MS_SSIM": msssim,
            "LPIPS": lpips_v, "last_PSNR": last_psnr,
            "BPP": bpp, "kbps": bitrate_kbps,
            "codebook_bytes": cb_bytes, "tail_bytes": tail_bytes,
            "enc_s": t_enc, "dec_s": t_dec,
            "args": vars(args),
        }, fh, indent=2)

    # Save side-by-side video + standalone mp4s
    try:
        import cv2
        side = []
        for gt, rc in zip(frames, recon):
            side.append(np.concatenate([np.asarray(gt), np.asarray(rc)], axis=1))
        side_arr = np.stack(side)  # (T, H, 2W, 3)
        gt_arr  = np.stack([np.asarray(f) for f in frames])
        rec_arr = np.stack([np.asarray(f) for f in recon])

        def write_mp4(arr, path, fps=16):
            T, H, W, _ = arr.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(path, fourcc, fps, (W, H))
            for frm in arr:
                vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
            vw.release()

        write_mp4(gt_arr, os.path.join(out_dir, "gt.mp4"))
        write_mp4(rec_arr, os.path.join(out_dir, "recon.mp4"))
        write_mp4(side_arr, os.path.join(out_dir, "side_by_side.mp4"))

        # Also save per-frame PNGs
        for i, frm in enumerate(side):
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:02d}.png"),
                        cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        print(f"Saved gt.mp4 / recon.mp4 / side_by_side.mp4 + per-frame PNGs to {out_dir}")
    except Exception as e:
        print(f"Frame save skipped: {e}")


if __name__ == "__main__":
    main()
