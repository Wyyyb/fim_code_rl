# Review Findings — paper_v3 final-pass proofread

This file collects items requiring author judgment. Items that I fixed
mechanically are listed at the bottom for record. Items that require a
decision or new data are at the top.

## Author-input needed

### A1. License-table TBD placeholders (sec/9_appendix.tex, Table tab:licenses)
Every per-license count in the appendix license-inventory table is
`\textcolor{red}{TBD}`. The accompanying prose in sec/9_appendix.tex
(L76-83) still asserts "$\sim$79\% permissive, $\sim$5\% copyleft,
$\sim$2\% Creative Commons" — these percentages depend on the same
counts that are TBD. Need final license counts (or temporarily soften
the "$\sim$79\%/5\%/2\%" claim until the counts are filled).

### A2. Standard-deviation placeholders (sec/4_experiments.tex)
Sec 4.1 setup states `\textcolor{red}{$\pm$?}` bands are placeholders
awaiting final replication runs. Most rows in tab:main do show std
bands but a couple of rows (e.g., R2E-Gym officially reported 7B/14B)
have ambiguous bands that look like guesses. Author should confirm
whether these are real or placeholder.

### A3. Hyperparameters appendix is a stub (sec/9_appendix.tex Section app:hp)
Section "Training Hyperparameters" is explicitly a stub
("Stub: hyperparameters and token budgets to be finalized at
camera-ready time"). NeurIPS reviewers will flag this. Fill in or
remove the stub note.

### A4. Cross-base-model claim now hinges on a single 8B point
Now that 32B was removed, the only non-Qwen2.5 evidence is Qwen3-8B
under SWE-Lego. The intro/limitations already concede this, but the
abstract phrasing "the improvement holds across post-training
pipelines and base-model families" overstates given $n=1$ alternative
family. Consider softening to "across two post-training pipelines and
on a non-Qwen2.5 base".

### A5. Abstract gain phrasing — coverage on Lite not stated
The abstract mentions Verified gains ($+2.8/+3.0/+3.2$) but does not
mention SWE-Lite (where the gains $+3.67/+4.00/+5.40$ are actually
larger on Qwen3-8B). The user's checklist suggests Lite gains are part
of the headline — author may want to add "and $+3.7/+4.0/+5.4$ on
SWE-Lite" to the abstract.

### A6. "doubles in parameters" — the 7B → 14B size step
Sec 4.2 "Consistent gains across model scales" now reads "as the base
model doubles in parameters". With only 7B and 14B, this is literally
correct but a bit thin. Verify that this is the framing you want,
versus dropping the "scales" framing entirely (since the cross-family
Qwen3-8B comparison sits in the next paragraph).

### A7. Multi-function gain claim ($4{\times}$) is intra-data, not parameter scaling
sec/5_analysis.tex L77 and sec/9_appendix.tex L633 say "more than
$4{\times}$ the gain on the $341$ single-function tasks ($+2.1$~pp)".
This is fine (it's a within-benchmark stratification). I flag it only
because the user's checklist asks specifically about "$4{\times}$"
language; this one is unrelated to the deleted 32B and should stay.

### A8. "+3.7/+4.0/+5.4" claim from the user's checklist
The user instructions list expected SWE-Lite gains as
"$+3.7$/$+4.0$/$+5.4$". The actual table values are $+3.67$, $+4.00$,
$+5.40$. The intro currently does not quote these explicitly; the only
SWE-Lite numbers that appear in prose are inside the experiments
section. If the abstract or contribution list should advertise them,
author needs to decide.

### A9. Unused references
The following bib entries are defined in references.bib but never
cited in the body:
- `starcoder2`
- `qwen3`
- `qwen3coder`
- `mmlu`
- `mathbench`
- `ifeval`
- `humanevalplus`

Per instructions, only flagged — not removed.

### A10. Compile blocked by missing LaTeX package
`pdflatex neurips_2026.tex` aborts with `File 'environ.sty' not
found.` This is a missing TeX Live package on this machine
(`tetex-extra`/`texlive-latex-extra` provides it), not a paper
problem. Author should confirm the paper compiles in their normal
environment. I could not produce a final PDF for sanity-check.

### A11. Ablation block (A) wording — "FIM, no CoT" tiny gain
Block A reports "+0.13 average / +0.60 Verified" for "no CoT". The
new prose in sec/4_experiments.tex calls this "small ... attribute to
most of the underlying code already being seen by the base model
during pretraining". This is plausible but is now the author's claim
rather than something the data forces — verify you are comfortable
with the framing.

### A12. `tab:dataset` and `fig:dataset_categories` are defined but never `\ref`'d
Both are in the appendix and the surrounding prose references the
table only by title, not via \ref. Not strictly broken, but a NeurIPS
reviewer will notice. Adding `(Table~\ref{tab:dataset})` or a
`Figure~\ref{fig:dataset_categories}` reference would be tidy.

## Mechanical fixes I applied

| File | Line(s) | Change |
| --- | --- | --- |
| sec/2_introduction.tex | 110-112 | "$+2.5$ on BFCL and $+4.0$ on tau-bench" → "$+2.4$ on BFCL and $+3.9$ on tau-bench" to match Table tab:cap ($+2.40$ / $+3.90$). |
| sec/1_abstract.tex | 21-23 | Reworked the awkward "---a consistent gain across scales---and" into "by $+2.8/+3.0$ points at 7B/14B and by a comparable $+3.2$ on Qwen3-8B" so the gains are stated once rather than restated as a parenthetical. |
| sec/4_experiments.tex | 121-129 | "stays in a narrow band as the base model doubles in parameters" → "is essentially unchanged as the base model doubles in parameters"; appended "we evaluate" to the deployment-range sentence so the two-point scaling claim is properly hedged. |

## Items I checked but did NOT change (consistent or intentional)

- `85% single + 15% pair` row in tab:ablation (C) — intentional: this is the single-pair-no-triple variant and is meant to differ from the main 80/15/5 mix. Per the user's instructions this row is meant to be 85/15.
- All other token / repo / file counts are now consistent: 968 repos, 78K files, 428 LoC/file, 34 LoC/target, 2.6B tokens, 320K/60K/20K split, 400K final samples.
- No `app:scaling`, `Full Qwen2.5-Coder Scaling Curves`, `teaser_v33`, `32B`, or `990` mentions remain anywhere in `sec/*.tex`, `neurips_2026.tex`, or `references.bib`.
- `\ref{...}` and `\eqref{...}` targets all resolve to `\label{...}`s that exist.
- The `$4{\times}$` mentions in sec/5 and the appendix are intra-benchmark patch-shape stratification, unrelated to the removed 32B scaling curve.
