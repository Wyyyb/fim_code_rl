"""Failure-mode breakdown (Fig 8) on SWE-Bench-Verified, 14B + r2e-gym.

Counts (per-run mean over three runs) parsed from sec/5_analysis.tex narrative:
  baseline:  no-patch ~11, loc-error ~131, patch-error ~227 -> failed total 369
  ours    :  no-patch ~1,  loc-error ~126, patch-error ~227 -> failed total 354
Solved per run = 500 - failed.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent

categories = ["Solved", "Patch error", "Loc. error", "No-patch"]
colors     = ["#2e8b57", "#e08e3a", "#9c6bbf", "#c0392b"]

# baseline (per-run mean), ours (per-run mean)
baseline = [131, 227, 131, 11]
ours     = [146, 227, 126,  1]
# Resolve "Solved" by 500-(other three)
baseline[0] = 500 - sum(baseline[1:])
ours[0]     = 500 - sum(ours[1:])

settings = ["+ r2e-gym", "+ FIM-Midtrain\n+ r2e-gym"]
data = [baseline, ours]

fig, ax = plt.subplots(figsize=(5.0, 2.8))
bottom = [0.0, 0.0]
for i, (cat, color) in enumerate(zip(categories, colors)):
    vals = [data[0][i], data[1][i]]
    bars = ax.bar(settings, vals, bottom=bottom,
                  color=color, edgecolor="white", linewidth=0.6,
                  width=0.55, label=cat)
    for j, (b, v) in enumerate(zip(bottom, vals)):
        if v >= 8:  # only label segments that are big enough
            ax.text(j, b + v / 2, f"{v}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottom = [b + v for b, v in zip(bottom, vals)]

ax.set_ylabel("Tasks per evaluation run (n=500)", fontsize=9)
ax.set_ylim(0, 540)
ax.tick_params(axis="both", labelsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.4, color="grey", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=False, fontsize=8.5,
          title="Outcome", title_fontsize=9)

# annotate solved gain on top of bar
delta = ours[0] - baseline[0]
ax.text(1, 510, f"+{delta} solved",
        ha="center", va="bottom",
        fontsize=8.5, color="#2e8b57", fontweight="bold")
# label No-patch values explicitly with arrows since segments are tiny
for j, v in enumerate([baseline[3], ours[3]]):
    ax.annotate(f"{v}", xy=(j, 500 - v / 2),
                xytext=(j - 0.45, 470 - 25 * j),
                fontsize=7.5, color="#c0392b",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#c0392b"))

plt.tight_layout(pad=0.4)
out = ROOT / "figs" / "failure_modes.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}  baseline={baseline}  ours={ours}")
