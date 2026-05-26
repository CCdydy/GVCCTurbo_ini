"""run_smoke.py — CogVideoX-Fun-2b GVCC smoke test on UVG."""
import sys, os, re, time, argparse, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_cogvideox_fun")
sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_wan22_fun")  # reuse run_smoke helpers
sys.path.insert(0, "/home/rog/Desktop/gvcc_turbo/turbogvcc")

from cogvideox_fun_wrapper import CogVideoXFunWrapper
from cogvideox_fun_pipeline import CogVideoXFunGVCCPipeline
from run_smoke import load_yuv420_frames, frames_to_tensor, psnr, lpips_v
from uvg_data import find_uvg_sequence
from pytorch_msssim import ms_ssim

UVG_DIR = "/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg"
FPG = 33

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Jockey")
    ap.add_argument("--height", type=int, default=480)   # CogVideoX default range
    ap.add_argument("--width",  type=int, default=720)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_cogvideox_fun/out_smoke")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    H, W = args.height, args.width

    yuv = find_uvg_sequence(UVG_DIR, args.seq)
    assert yuv, f"{args.seq} not found"
    print(f"[INFO] Loading {FPG} frames of {args.seq}")
    raw = load_yuv420_frames(yuv, FPG, args.start_frame)
    frames = [f.resize((W, H), Image.LANCZOS) for f in raw]
    print(f"[INFO] {len(frames)} frames @ {W}x{H}")

    w = CogVideoXFunWrapper()
    pipe = CogVideoXFunGVCCPipeline(
        w, K=args.K, M=args.M, num_steps=args.steps, eta=args.eta,
        guidance_scale=args.cfg, num_frames=FPG, height=H, width=W, seed=args.seed,
    )

    print("\n[ENCODE]")
    t0 = time.time()
    step_data, x0_enc = pipe.encode(frames, prompt="",
                                     first_image=frames[0], last_image=frames[-1])
    t_enc = time.time() - t0

    print("\n[DECODE]")
    t0 = time.time()
    recon = pipe.decode(step_data, prompt="",
                         first_image=frames[0], last_image=frames[-1])
    t_dec = time.time() - t0

    recon = [f.crop((0, 0, W, H)) if f.size != (W, H) else f for f in recon]
    n = min(len(frames), len(recon))
    t_gt = frames_to_tensor(frames[:n])
    t_rec = frames_to_tensor(recon[:n])
    p_mean, per_p = psnr(t_gt, t_rec)
    msssim = ms_ssim(t_gt, t_rec, data_range=1.0).item()
    lp = lpips_v(t_gt, t_rec)

    cb_bytes = pipe._total_codebook_bits // 8
    bpp = cb_bytes * 8 / (FPG * H * W)
    kbps = cb_bytes * 8 / (FPG / 16.0) / 1000.0

    print(f"\n{'='*60}")
    print(f"RESULT — {args.seq} GOP0 (CogVideoX-Fun-2b, FLF2V)")
    print(f"{'='*60}")
    print(f"PSNR        = {p_mean:.2f} dB  (last frame {per_p[-1].item():.2f})")
    print(f"MS-SSIM     = {msssim:.4f}")
    print(f"LPIPS       = {lp:.4f}")
    print(f"Per-frame PSNR = {[round(p.item(), 2) for p in per_p]}")
    print(f"Codebook    = {cb_bytes}B  BPP={bpp:.6f}  {kbps:.2f} kbps")
    print(f"Time        = enc {t_enc:.1f}s + dec {t_dec:.1f}s = {t_enc+t_dec:.1f}s")

    import cv2
    def write_mp4(arr_list, path, fps=16):
        arr = np.stack([np.asarray(f) for f in arr_list])
        T, h, w_, _ = arr.shape
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_, h))
        for frm in arr: vw.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        vw.release()
    write_mp4(frames, os.path.join(args.out, "gt.mp4"))
    write_mp4(recon, os.path.join(args.out, "recon.mp4"))
    side = [np.concatenate([np.asarray(g), np.asarray(r)], axis=1) for g, r in zip(frames, recon)]
    write_mp4([Image.fromarray(s) for s in side], os.path.join(args.out, "side.mp4"))
    print(f"Saved videos to {args.out}")
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"PSNR": p_mean, "MS_SSIM": msssim, "LPIPS": lp,
                   "BPP": bpp, "kbps": kbps, "enc_s": t_enc, "dec_s": t_dec,
                   "args": vars(args)}, fh, indent=2)


if __name__ == "__main__":
    main()
