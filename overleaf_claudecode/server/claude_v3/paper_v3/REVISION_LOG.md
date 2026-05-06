# Revision log (paper_v1 -> paper_v2)

This log captures key decisions and observations during the revision pass.
The final summary of changes and remaining gaps lives in `CHANGES.md`.

## Setup
- Working dir: `paper_v2/`. Original `paper_v1/` untouched.
- Removed unused backup files (`*_backup.tex`, old `template.tex`, prior PDF, `9_appendix_bk.tex` — its content will be folded into `9_appendix.tex` rather than `\input`-ed separately).
- Per user: use **200K** as the controlled ablation budget (table value).
- Per user: 60 repositories with "No License" are grouped under Apache-2.0 for now (user will manually adjust).
- Per user: red placeholders in appendix Table 1 (mean LoC, target counts, etc.) remain TODO.

## Phase plan
1. Figure rebuild (subagent) — compact 3-panel `figs/pdg_score.pdf` with code + simplified bars + scatter, single row.
2. Method §3 rewrite — drop in-line worked-example numbers, move dataset stats fig+table to appendix, target ~2.0–2.3 pages.
3. Appendix expansion — single-function pseudocode (already there), add multi-function pseudocode + worked example, license table, fold Lite analysis from old `9_appendix_bk.tex`, add stubs for `app:hp` and `app:scaling`.
4. Experiments tightening — fill ablation block (B) red placeholders from table, add prose for block (C), condense setup/main/cap.
5. Analysis tightening — move patchshape figure to appendix, condense.
6. Related / Limitations / Conclusion — ~40% reduction. Limitations becomes a single paragraph.
7. Global polish — American spelling, citation consistency, dangling references resolved or stubbed.
8. Length verification — try LaTeX compile via subagent; fall back to per-section line counts vs. v1.

## Open data gaps (transcribed from v1)
- `app:hp` — training hyperparameters appendix (referenced from §4.1, §4.2). Will create stub.
- `app:scaling` — full Qwen2.5-Coder scaling curves (referenced from §4.2). Will create stub.
- `sec:exp:crossbase` — referenced from §7 limitations; resolves to §4.2 main results, will retarget.
- Appendix Table 1 red placeholder values: mean LoC ≈ 425, single-fn ≈ 412K, k=2 ≈ 96K, k=3 ≈ 31K, mean target LoC ≈ 38, %CoT = 100. Kept red.
- Standard-deviation bands (`±?` in red) in main table — unchanged per user.

## Phase 5: Related/Limitations/Conclusion (subagent)

Pass tightens the closing three sections and applies American spelling globally in those files. Highlights:

- `6_related_works.tex`: 81→55 lines, 630→376 words (~60%). Replaced `\paragraph{}` with inline `\textbf{}.`. Trimmed mid-training meta-commentary about "concentrating signal", collapsed FIM three-difference enumeration to one sentence each, removed self-supervised vs. trajectory rephrasing in the agent-models paragraph, dropped the distillation meta-disclaimer prose. Added `\vspace{-0.5ex}` after `\section`.
- `7_limitations.tex`: 72→30 lines, 554→201 words. Converted six `\paragraph{Limitation N: …}` items into a single paragraph with `(i)`–`(iv)` markers. Dropped Limitation 3 (PDG dynamic-Python imperfections) and Limitation 5 (capability preservation only at 14B) per spec.
- `8_conclusion.tex`: 29→22 lines, 197→140 words (~71% by words). Cut redundant "structural isomorphism" restatement and the long future-work enumeration; kept the four-part structure framing, function-aware FIM intro, three-axis robustness summary, cross-domain transfer, and a one-line future-work pointer.
- Style: `generalisation→generalization`, `optimising→optimizing`, `synthesise→synthesize`, `behaviour→behavior`, `characterise→characterize`, `specialisation→specialization`, etc., applied only to these three files. `\citep{...}` style preserved; no new bib keys introduced.

Nothing in the must-keep set was forced out. The conclusion's RL-mid-training composition framing was compressed to a half-sentence rather than dropped.

## Phase 1: Figure rebuild (subagent)

Replaced `figs/pdg_score.pdf` with a single-row 3-panel layout (7.0×1.95 in, vector PDF):
- (a) condensed code listing visible on the left + PDG diagram on the right; call edges and sibling edges remain readable. Code is real, not just signatures: `total` body shows `is_int(v)` and `add(s, v)` so call edges trace back to source.
- (b) two horizontal stacked bars (Ĥ=0.40, Î=0.48) with sub-segment labels; formula/weight clutter removed.
- (c) scatter in the (Ĥ, Î) plane with one rose-tinted iso-FIM contour at τ=0.20; legend trimmed to selected/filtered/FIM=τ.

Generating script committed at `figs/plot_pdg_score.py`.

## Phase 2: Method rewrite (this agent)

`3_method.tex` rewritten for paper_v2:
- Dropped the `dataset_categories` figure + `dataset` table from §3 (moved to appendix).
- Removed in-line worked-example numbers (LoC=10, CC=5, the per-component decompositions of Ĥ/Î, etc.) — they now live only in `app:algo:worked_example`.
- Multi-function group selection paragraph trimmed from ~12 lines to 8 with a forward pointer to `Algorithm~\ref{alg:selection_multi}` and the new pair worked example.
- CoT subsection trimmed.
- `\paragraph{}` instances removed in favour of `\textbf{...}` inline emphasis where present.
- `\vspace{-0.5ex}` added after `\section`/`\subsection` to save vertical space.
Length: §3 now occupies pages 3–4 in the compiled PDF (~2 pages with figure inline); v1 occupied 3 full pages.

## Phase 3: Experiments tightening (this agent)

`4_experiments.tex` rewritten for paper_v2:
- Tables left numerically unchanged.
- §4.1 (Setup): `\paragraph{}` → inline `\textbf{...}`; condensed prose; protocol footnote shortened.
- §4.2 (Main): three `\paragraph{}` blocks compressed by ~30%.
- §4.3 (Capability): three `\paragraph{}` blocks compressed by ~30%; corrected the regression baseline numbers ($\tau$-bench drop is $2.30$ not $5.30$ to match the table; the v1 prose was inconsistent with its own Table~\ref{tab:cap}).
- §4.4 (Ablation): filled all red placeholders in block (B) using the table values (Random=13.95, Gemini-selected=15.05, PDG only=14.85, PDG+Ĥ=15.05, PDG+Î=15.35, Full=15.60); recomputed deltas. Added a new prose paragraph for Block (C) Mask granularity using the table.
- Block (B) controlled budget number unified to **200K** (per user) — was inconsistent (100K in body / 200K in caption) in v1.
Length: §4 fits in pages 5–7 of the compiled PDF (~3 pages).

## Phase 4: Analysis tightening (this agent)

`5_analysis.tex` rewritten for paper_v2:
- Moved the `passrate_by_patchshape_verified` figure to the appendix (`app:behavior_extra:patchshape`); reference in §5.2 now points there.
- Trimmed the headline-table caption to one sentence with a pointer to the appendix.
- Compressed prose in both subsections by ~30%.
Length: §5 fits on a single page (page 8).

## Phase 6 / 7: Appendix expansion + global polish (this agent)

`9_appendix.tex` rewritten for paper_v2:
- Added a section A.1 License Inventory with a 2-column table built from `code_repo_list_with_license.csv`. Per user, the 60 "No License" entries are folded into Apache-2.0 (370 total) for now; ~53 unidentifiable rows go to "Other / unidentifiable". User to manually adjust.
- Added Algorithm 2 (Multi-Function Group Selection pseudocode) and a worked example (`normalize` + `word_freq` caller-callee pair) showing why joint masking strictly reduces $\hat{I}$.
- Added the `dataset_categories` figure and corpus stats table that previously lived in §3.
- Added a stub Section C "Training Hyperparameters" enumerating the categories of hyperparameters that will be filled in at camera-ready time (resolves dangling `app:hp` references from §4.1, §4.2).
- Added a stub Section D "Full Qwen2.5-Coder Scaling Curves" (resolves dangling `app:scaling` reference from §4.2).
- Folded the SWE-Bench-Lite behavioral analysis from `9_appendix_bk.tex` into the main appendix as Section F. Section E (Verified) hosts the patchshape figure, full trajectory metrics, action-type distribution, failure-mode breakdown, no-patch mechanism, multi-file analysis, and the scikit-learn-26323 contrast.
- Added the negative-observation pattern list (resolves a cross-reference from §5.1).

Other global polish:
- `7_limitations.tex` corrected `FullStackBench-ZH` → `FullStackBench-EN` (the benchmark we actually evaluate on; v1 said ZH).
- `neurips_2026.tex`: removed the duplicate `\usepackage{xcolor}` (line 18 vs.\ `\usepackage[table]{xcolor}` on line 25) — this was already a latent bug in v1 that caused an option clash on a clean TeX Live install.
- All red `\textcolor{red}{...}` placeholders preserved per user instruction (data placeholders to be filled at camera-ready).
- Dropped unused backup/template files from paper_v2 (kept only the three source files actually compiled).
- Cited list checked: every `\citep{...}` resolves; no dangling citation in `pdflatex` log.

## Phase 8: Length verification

Compiled with TeX Gyre Termes/Heros (Times-equivalent fonts) on the local sandbox; see `~/texmf/` for the user-mode TeX tree the verification used. Page numbers in the resulting PDF:
- §1 Introduction starts page 2 (page 1 = title/abstract/teaser).
- §2 Method starts page 3, ends mid-page 4.
- §3 Experiments starts page 5, ends mid-page 7.
- §4 Analysis starts page 8 (single page).
- §5 Related Work starts page 9.
- §6 Limitations + §7 Conclusion both fit on page 10.
- References start page 10, span through page 12.
- Appendix A starts page 12.

**Main text (intro through conclusion) occupies pages 1–10**, down from v1's 12.5 pages — a reduction of $\sim\!2.5$ pages, on target.

NeurIPS hard limit is 9 pages of main text excluding references; the user has stated they will manually trim the remaining $\sim\!0.5$ page.


