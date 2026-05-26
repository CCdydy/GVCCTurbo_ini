"""run_smoke.py — Wan2.2-Fun-5B-InP GVCC smoke test on UVG."""
import sys, os, re, time, argparse, json
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

UVG_DIR = "/home/rog/Desktop/gvcc_turbo/turbogvcc/data/uvg"
FPG = 33
HEIGHT, WIDTH = 704, 1280


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
            if len(raw) < frame_size: break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:H*W].reshape(H, W)
            u = yuv[H*W:H*W+H*W//4].reshape(H//2, W//2)
            v = yuv[H*W+H*W//4:].reshape(H//2, W//2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(np.stack([y, u, v], -1), cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames


def frames_to_tensor(frames):
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    p = 10 * torch.log10(1.0 / mse.clamp(min=1e-12))
    return p.mean().item(), p


_LPIPS = None
def lpips_v(a, b):
    global _LPIPS
    if _LPIPS is None: _LPIPS = lpips.LPIPS(net='alex').cuda()
    with torch.no_grad():
        return _LPIPS((a*2-1).cuda(), (b*2-1).cuda()).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Jockey")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ddim_tail", type=int, default=3)
    ap.add_argument("--g_scale", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_call_skip", default="none", choices=["none", "u_cache", "x0_cache"])
    ap.add_argument("--model_call_period", type=int, default=2)
    ap.add_argument("--out", default="/home/rog/Desktop/gvcc_turbo/turbogvcc/exp_wan22_fun/out_smoke")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    yuv = find_uvg_sequence(UVG_DIR, args.seq)
    assert yuv, f"{args.seq} not found"
    print(f"[INFO] Loading {FPG} frames of {args.seq}")
    raw = load_yuv420_frames(yuv, FPG, args.start_frame)
    frames = [f.resize((WIDTH, HEIGHT), Image.LANCZOS) for f in raw]
    print(f"[INFO] {len(frames)} frames @ {WIDTH}x{HEIGHT}")

    w = Wan22FunWrapper()
    pipe = Wan22FunGVCCPipeline(
        w, K=args.K, M=args.M, num_steps=args.steps, num_ddim_tail=args.ddim_tail,
        guidance_scale=args.cfg, g_scale=args.g_scale,
        num_frames=FPG, height=HEIGHT, width=WIDTH, seed=args.seed,
        model_call_skip=args.model_call_skip,
        model_call_period=args.model_call_period,
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

    recon = [f.crop((0, 0, WIDTH, HEIGHT)) if f.size != (WIDTH, HEIGHT) else f for f in recon]
    n = min(len(frames), len(recon))
    t_gt = frames_to_tensor(frames[:n])
    t_rec = frames_to_tensor(recon[:n])
    p_mean, per_p = psnr(t_gt, t_rec)
    msssim = ms_ssim(t_gt, t_rec, data_range=1.0).item()
    lp = lpips_v(t_gt, t_rec)

    cb_bytes = pipe._total_codebook_bits // 8
    bpp = cb_bytes * 8 / (FPG * HEIGHT * WIDTH)
    kbps = cb_bytes * 8 / (FPG / 16.0) / 1000.0

    print(f"\n{'='*60}")
    print(f"RESULT — {args.seq} GOP0 (frames {args.start_frame}–{args.start_frame+FPG-1})")
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
        T, H, W, _ = arr.shape
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
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
