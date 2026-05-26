# GVCCTurbo Paper Source

WACV 2027 submission draft. Target track: **algorithms**.

## Layout

```text
paper/
├── main.tex          # entry; loads sec/0..6
├── main.bib          # references
├── preamble.tex      # packages used by main.tex
├── sec/
│   ├── 0_abstract.tex
│   ├── 1_introduction.tex
│   ├── 2_related_works.tex
│   ├── 3_method.tex
│   ├── 4_experiments.tex
│   ├── 5_ablations.tex
│   └── 6_conclusion.tex
├── wacv.sty          # WACV class file (organizers' template)
├── ieeenat_fullname.bst   # bib style (template)
└── main_v1_draft.pdf # rendered v1 (not the camera-ready)
```

## Building

We use `tectonic` (modern self-contained LaTeX engine, no system TeX needed):

```bash
cd paper
tectonic main.tex   # writes main.pdf
```

If you have a normal TeX install (e.g.\ TeX Live), the classic incantation works too:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Build artifacts (`*.aux`, `*.bbl`, `*.log`, `main.pdf`) are `.gitignore`'d.
Only the tagged `main_v1_draft.pdf` snapshot is tracked, for offline review.

## Current draft state

10 pages: 9 body + 1 references. The page-budget headroom at the
algorithms track is 8 body + unlimited references, so a single page must
be cut before final submission. Most likely sources of trim:

- §3.2 (multi-resolution method) prose around the mismatch equation.
- §5.4 (vectorized RNG) can fold into a footnote.
- One ablation table can move to supplementary.

## Framing

The paper is written as a **video-codec paper**, not a diffusion-sampler
paper. The detailed framing constraints are recorded in the project
memory (see [README.md](../README.md) "Method Notes" and `RESEARCH_NOTES.md`).
Section structure:

1. **Intro** — open on video compression, position GVCCTurbo as
   GOP-structured generative codec.
2. **Related work** — traditional / learned / generative codecs, then
   diffusion-sampling acceleration.
3. **Method** — codec backbone (compact); multi-resolution coding with
   the target-domain fix; predictive skip with clean-endpoint caching;
   three GOP conditioning modes.
4. **Experiments** — UVG 1080p RD comparison + decoder wall-clock.
5. **Ablations** — lp_raw fix vs.\ naive MR; cache choice
   (velocity vs.\ endpoint); skip cap (one-step radius); stage count.
6. **Conclusion** — limitations and future work (no separate "limitations"
   section).

## What's still to do before submission

- [ ] Trim 1 page (see suggestions above).
- [ ] Fill in author + affiliation in `main.tex`.
- [ ] Add RD curve figure (Tier-1/2/3 LPIPS vs.\ BPP) alongside Table 1.
- [ ] Add Figure 1 / teaser: GOP block diagram + RD point comparison.
- [ ] Verify all reference details (some bib entries use `others` as a
      placeholder author; fill in real authors).
- [ ] Run full UVG (currently half) to close the BPP-amortization gap
      reported in §4.2.
- [ ] Add per-sequence breakdown table to supplementary.

## Data backing the tables and ablations

All numerical entries in Sections 4 and 5 come from the experiment
records under `../exp_*/results*/` (per-GOP `metrics.json`) and the
aggregate summaries (`summary.json`). The corresponding memory entries
provide context for why each number was generated and what was tried
and rejected:

- T2V FINAL 720p (Sec.\ 5.1, 5.2): see `exp_v1_mr/REPORT.md` and
  `../MEMORY.md → mr_x0cache_warmup_result`.
- MR-FLF2V 1080p (Sec.\ 4.2): `exp_mr_flf2v/results_1080p_half/`.
- T2V 1.3B 1080p (Sec.\ 4.2): `exp_v1_mr/results_1080p_half_T2V13B/`.
- DLC1 transition codebook (Sec.\ 5.1): `exp_v1_mr/REPORT.md`.

For new experiments that change a number cited in the paper, update
both the corresponding `metrics.json` / `summary.json` and the inline
number in the relevant `sec/*.tex` file.
