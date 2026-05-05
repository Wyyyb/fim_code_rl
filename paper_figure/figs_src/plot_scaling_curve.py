"""Scaling figure (Fig 6): SWE-Bench-V/L vs model size."""
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

sizes      = [7, 14, 32]
verified_b = [15.00, 26.20, 31.80]
verified_o = [17.80, 29.20, 35.10]
lite_b     = [11.33, 18.00, 24.70]
lite_o     = [15.00, 22.00, 26.80]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharex=True)

for ax, base, ours, ylabel in [
    (axes[0], verified_b, verified_o, "SWE-Bench-Verified (%)"),
    (axes[1], lite_b,     lite_o,     "SWE-Bench-Lite (%)"),
]:
    ax.plot(sizes, base, marker="o", linewidth=1.6, markersize=6,
            color="#999999", label="post-training only", zorder=2)
    ax.plot(sizes, ours, marker="s", linewidth=1.8, markersize=6,
            color="#c0392b", label="+ FIM mid-training", zorder=3)
    for x, b, o in zip(sizes, base, ours):
        gain = o - b
        ax.annotate(
            f"+{gain:.1f}",
            xy=(x, o), xytext=(0, 7), textcoords="offset points",
            ha="center", fontsize=8, color="#c0392b",
        )
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Model size (params)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.4, color="grey", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)
    ymin = min(base + ours) - 3
    ymax = max(base + ours) + 5
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(loc="lower right", frameon=False, fontsize=8)

plt.tight_layout(pad=0.4)
out = ROOT / "figs" / "scaling_curve.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
