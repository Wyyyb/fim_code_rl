# Summary of changes (paper_v1 → paper_v2)

This document is the deliverable summary of revisions applied to your v1
NeurIPS submission. The detailed timeline of decisions is in
`REVISION_LOG.md`.

The output sits entirely in `paper_v2/` — `paper_v1/` was not modified.

## Length

- **Main text (intro → conclusion):** v1 ≈ 12.5 pages → v2 ≈ 10 pages.
  Target was 9.5–10; reduction of ~2.5 pages.
- **Appendix:** expanded with multi-function pseudocode + worked
  example, license inventory, hyperparameter stub, scaling-curve stub,
  and the SWE-Bench-Lite behavioral analysis folded in.
- Verified by compiling with TeX Gyre Termes (Times analog) locally.
  In the compiled PDF, `\section{Conclusion}` resolves to page 10;
  references span pages 10–12; appendix begins on page 12.

## Section-level changes

### `figs/pdg_score.pdf` (regenerated)
- Old layout was a 2-row figure with code listing in the top row and
  three score panels in the bottom row, occupying ~half a page.
- New layout: **single-row 3-panel figure**, 7.0 × 1.95 in, vector PDF.
  - Panel (a) keeps a condensed but real code listing on the left so
    readers can see where call edges come from, plus the PDG diagram.
  - Panel (b) reduces to two horizontal stacked bars (Ĥ, Î) with
    sub-segment labels; formulas/weights/caps removed (they live in the
    body equations).
  - Panel (c) keeps the (Ĥ, Î) scatter with a single iso-FIM contour
    at τ=0.20; the FIM=0.10 / FIM=0.30 dashed contours and the
    "hard-filtered" annotation arrow were removed for clarity.
- Generating script committed at `figs/plot_pdg_score.py`.

### `sec/3_method.tex`
- Dropped the `dataset_categories` figure + dataset-stats table from §3
  (moved to Appendix A); §3 now flows as: Motivation → Data Collection
  → Selection (with the new compact figure) → CoT.
- Removed inline worked-example numbers (LoC=10, CC=5, per-component
  Ĥ/Î decompositions); these now live solely in Appendix B.6.
- Kept the four selection equations and the four subsubsections.
- Multi-function group selection paragraph trimmed to one short
  paragraph with a forward pointer to the new
  `Algorithm~\ref{alg:selection_multi}` and the worked pair example.
- CoT subsection condensed.
- `\paragraph{}` macros mostly removed in favor of inline `\textbf{...}.`.

### `sec/4_experiments.tex`
- All tables unchanged numerically.
- Setup, Main Results, and Capability prose tightened by roughly 30%
  each; `\paragraph{}` macros replaced with `\textbf{...}.`.
- **Ablation block (B) red placeholders filled** from the table values:
  Random=13.95, Gemini-selected=15.05, PDG only=14.85, PDG+Ĥ=15.05,
  PDG+Î=15.35, Full=15.60. All deltas recomputed.
- **New prose for ablation block (C) Mask granularity** describing the
  pair/triple mixture vs.\ single-only, using the table values.
- Block (B) budget number harmonized to **200K** (table caption value)
  in body text (v1 said 100K in body / 200K in caption).
- Corrected $\tau$-bench drop in §4.3 prose from $5.30$ to $2.30$ to
  match the actual `\textbf{tab:cap}` values.

### `sec/5_analysis.tex`
- Moved `passrate_by_patchshape_verified.pdf` figure to the appendix
  (now `\ref{fig:patchshape}` in `app:behavior_extra:patchshape`).
- Trimmed the table caption; merged the two trailing paragraphs of the
  multi-function subsection into one.
- §5 now fits on a single page.

### `sec/6_related_works.tex`
- Reduced from 81 lines / 630 words to 55 lines / 376 words (~60%).
- `\paragraph{}` replaced with inline `\textbf{...}.`.
- Cut FIM three-difference enumeration verbosity, distillation
  meta-disclaimer, "concentrating signal" repetition.

### `sec/7_limitations.tex`
- Reduced from 72 lines / 554 words to 30 lines / 201 words.
- Six `\paragraph{Limitation N: …}` items collapsed into one paragraph
  with `(i)–(iv)` markers.
- Dropped the dynamic-PDG and "14B-only capability comparison"
  limitations per scope; kept Python-only, teacher dependency, partial
  cross-base, modularity assumption.
- Fixed `FullStackBench-ZH` → `FullStackBench-EN` (we evaluate on EN).

### `sec/8_conclusion.tex`
- Reduced from 29 lines / 197 words to 22 lines / 140 words (~71%).
- Cut redundant restatement of the structural-isomorphism framing and
  the long future-work enumeration.

### `sec/9_appendix.tex` (substantially expanded)
- New §A.1 **License Inventory** (table built from
  `code_repo_list_with_license.csv`; 60 "No License" entries grouped
  into Apache-2.0 per your request).
- New §B.7 **Multi-function group selection algorithm** as
  `Algorithm 2` (`alg:selection_multi`) and a worked
  caller-callee pair example (`normalize` + `word_freq`) showing how
  joint masking reduces Î.
- New §B.9 **Negative-observation patterns** (resolves a cross-ref from
  §5.1).
- New §C **Training Hyperparameters** (stub; resolves dangling `app:hp`
  refs from §4.1, §4.2).
- New §D **Full Qwen2.5-Coder Scaling Curves** (stub; resolves
  dangling `app:scaling` ref from §4.2).
- New §F **Behavioral Analysis on SWE-Bench-Lite** — content folded in
  from the previously-unincluded `sec/9_appendix_bk.tex`.
- The dataset-categories figure and the corpus statistics table that
  used to live in §3 now sit in §A.

### Style and configuration
- American spelling sweep (behavior, generalization, optimizing,
  synthesize, characterize, specialization). The only remaining `behaviour`
  was in the unused `sec/9_appendix_bk.tex` which was deleted.
- Citation style `\citep{...}` preserved throughout; no new bibkeys
  introduced.
- `\vspace{-0.5ex}` added after `\section{...}` and selected
  `\subsection{...}` calls; `\vspace{-1.5ex}` / `-2ex` used in the
  method figure caption to tighten the figure block.
- `neurips_2026.tex`: removed the duplicate `\usepackage{xcolor}` line
  that conflicted with `\usepackage[table]{xcolor}` (latent v1 bug
  causing an option clash on clean TeX Live installs).
- Dropped unused backup/template files from paper_v2 root
  (`*_backup.tex`, `template.tex`, `background_motivation.txt`,
  `9_appendix_bk.tex`); only the actually-compiled sources remain.

## Data and information that is still missing or placeholder

Marked with `\textcolor{red}{...}` in the source where applicable, and
listed here so you can fill them at camera-ready time:

1. **Standard-deviation bands in `tab:main`** — the `(\pm?)` red marker
   on a few cells (e.g., the official SWE-Smith row), per your instruction
   left as-is awaiting final replication runs.
2. **Appendix Table 1 (corpus statistics)**, Appendix A, red rows:
   - Mean LoC per file (`≈ 425`)
   - Single-function FIM targets (`≈ 412K`)
   - Multi-function targets at `k=2` (`≈ 96K`) and `k=3` (`≈ 31K`)
   - Mean target LoC (`≈ 38`)
   - Targets with Gemini-3 CoT (`100%`)
3. **Appendix C Training Hyperparameters** is a stub. Concrete values
   needed:
   - Mid-training: epochs, peak LR, warm-up steps, weight decay,
     gradient-clip norm, optimizer, batch size, per base model
     (Qwen2.5-Coder-7B/14B/32B, Qwen3-8B).
   - Sequence length per base model (matching native pretraining
     context), packing strategy, attention-mask boundaries.
   - Post-training pipeline configuration (R2E-Gym/SWE-Smith/SWE-Lego),
     epochs, LR schedule, seed.
   - Evaluation: agent harness configuration, max turns, decoding
     temperature, per-step timeout, seed values.
4. **Appendix D Full Qwen2.5-Coder Scaling Curves** is a stub awaiting
   the actual scaling figure (PDF) — numerical values are already in
   `tab:main`, but a curve figure has not been produced.
5. **License inventory**: the 60 "No License" entries are folded into
   Apache-2.0 per your direction. Manual adjustment may be needed if
   any of those repos turn out to be CC, GPL, or proprietary.
6. **References**: the bibliography is unchanged from v1
   (`references.bib`). All `\citep{...}` keys in the body resolve
   without warnings, but if the camera-ready version updates any
   citations (e.g., the SWE-Lego entry currently has placeholder
   author/title), they need re-checking.

## Open editorial nits the author may want to revisit

- The teaser figure (`figs/teaser_v33.pdf`) was not regenerated; if
  you also want the teaser tightened, that is a follow-up.
- The author block in `neurips_2026.tex` is empty (anonymous
  submission); fill in for camera-ready.
- The Qwen3-8B SWE-Lego row in `tab:main` shows "officially reported"
  was not given. Once the SWE-Lego paper publishes final numbers, that
  row can be added.

## File list in `paper_v2/`

- `neurips_2026.tex` — main entry, `xcolor` clash fixed.
- `neurips_2026.sty` — unchanged copy of NeurIPS 2026 style.
- `references.bib` — unchanged.
- `checklist.tex` — untouched per your instruction.
- `sec/1_abstract.tex`, `sec/2_introduction.tex` — unchanged.
- `sec/3_method.tex` — rewritten.
- `sec/4_experiments.tex` — rewritten.
- `sec/5_analysis.tex` — rewritten.
- `sec/6_related_works.tex` — tightened.
- `sec/7_limitations.tex` — collapsed to one paragraph.
- `sec/8_conclusion.tex` — tightened.
- `sec/9_appendix.tex` — substantially expanded.
- `figs/pdg_score.pdf` + `figs/plot_pdg_score.py` — new compact figure.
- `REVISION_LOG.md` — phase-by-phase log.
- `CHANGES.md` — this document.
