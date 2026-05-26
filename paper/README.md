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
The tagged `main_v2_draft.pdf` snapshot is tracked for offline review.

## Current draft state (v2.1)

11 pages: 10 body + 1 references at WACV algorithms track (limit 8 body
+ unlimited references). The body is over the budget, but final trimming
is deferred until figures and supplementary tables land, since those
will displace text and change the layout. Likely trim sources, when the
time comes:

- Move §5.1 Table 6 (joint-oracle diagnostic) to supplementary.
- Move §5.3 Table 9 (stage-count sweep) to supplementary.
- Fold §5.4 (vectorized RNG) into a footnote.
- Move §4.2 per-sequence detail to supplementary.

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

Ordered roughly by priority. Page length is **not** a current priority --
final trim happens after figures and supplementary land, since those
will reshuffle the layout.

- [ ] **Figures.** Teaser/Figure 1 (GOP pipeline block diagram + RD
      point comparison vs.\ HEVC / DCVC-RT / GLC-Video / paper GVCC).
      RD curve figure alongside Table 1 (LPIPS vs.\ BPP across all
      three tiers).
- [ ] **Full-UVG measurements at 1080p.** Currently the FLF2V and T2V
      1080p entries are preliminary half-UVG; rerunning the remaining
      9 GOPs per sequence will close the boundary-amortization gap
      reported in §4.2.
- [ ] **Author + affiliation.** Fill in `main.tex` lines 21--30 and
      the WACV paper ID (`\def\wacvPaperID{*****}`).
- [ ] **Bibliography hygiene.** Several entries in `main.bib` use
      `and others` as a placeholder author list (Free-GVC,
      GLC-Video, GNVC-VD, SPD, etc.); fill in the real authors and
      publication venue strings from the cited papers.
- [ ] **Supplementary file.** Per-sequence breakdown tables, all
      diagnostic-subset ablations (joint oracle, stage-count sweep),
      vectorized-RNG details, and the qualitative video frames.
- [ ] **Final layout pass.** Once figures + supplementary land, trim
      the body to 8 pages by moving overflow into supplementary.

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
