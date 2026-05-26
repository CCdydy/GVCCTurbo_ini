#!/usr/bin/env python3
"""Summarize V2-a Stage 0 result folders."""

import argparse
import json
from pathlib import Path


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _collect_times(summary):
    enc, dec = [], []
    for seq in summary.get("sequences", []):
        for gop in seq.get("gop_results", []):
            enc.append(gop.get("encode_s"))
            dec.append(gop.get("decode_s"))
    return _mean(enc), _mean(dec)


def _fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize V2-a summary.json files")
    parser.add_argument("--results_dir", default="exp_v2a/results")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*/summary.json")):
        with path.open() as f:
            summary = json.load(f)
        cfg = summary.get("config", {})
        overall = summary.get("overall", {})
        enc_s, dec_s = _collect_times(summary)
        rows.append({
            "name": path.parent.name,
            "mode": cfg.get("codebook_mode", "unknown"),
            "schedule": cfg.get("spectral_codebook_schedule") or "none",
            "psnr": overall.get("PSNR_dB"),
            "lpips": overall.get("LPIPS"),
            "bpp": overall.get("BPP"),
            "enc_s": enc_s,
            "dec_s": dec_s,
        })

    if not rows:
        print(f"No summary.json files found under {args.results_dir}")
        return

    headers = ["name", "mode", "PSNR", "LPIPS", "BPP", "enc_s/GOP", "dec_s/GOP", "schedule"]
    widths = [28, 18, 8, 8, 10, 10, 10, 38]
    print(" ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(" ".join("-" * w for w in widths))
    for row in rows:
        values = [
            row["name"],
            row["mode"],
            _fmt(row["psnr"], 2),
            _fmt(row["lpips"], 4),
            _fmt(row["bpp"], 6),
            _fmt(row["enc_s"], 1),
            _fmt(row["dec_s"], 1),
            row["schedule"],
        ]
        print(" ".join(str(v).ljust(w)[:w] for v, w in zip(values, widths)))


if __name__ == "__main__":
    main()
