"""Token budget scaling figure (Fig 9). PLACEHOLDER values pending experiment."""
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

# PLACEHOLDER -- replace with measured values when ablation is run.
budgets   = [0.5, 1.0, 2.0, 3.0, 5.0]      # billions of tokens
swe_v     = [25.4, 27.0, 28.4, 29.2, 29.4] # placeholder (saturating ~3B)

baseline_v = 26.20  # 14B + r2e-gym only

fig, ax = plt.subplots(figsize=(4.4, 2.5))
ax.axhline(baseline_v, color="#999999", linestyle="--", linewidth=1.0,
           label="post-training only (no mid-train)")
ax.plot(budgets, swe_v, marker="s", linewidth=1.8, markersize=6,
        color="#c0392b", label="+ FIM mid-training")

# default budget marker
ax.scatter([3.0], [29.2], s=120, facecolor="none", edgecolor="#c0392b",
           linewidth=1.5, zorder=4)
ax.annotate("default\n(3B tokens)",
            xy=(3.0, 29.2), xytext=(3.0, 27.4),
            ha="center", fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#c0392b"))

ax.set_xscale("log")
ax.set_xticks(budgets)
ax.set_xticklabels([f"{b:g}B" for b in budgets])
ax.minorticks_off()
ax.set_xlabel("Mid-training token budget", fontsize=9)
ax.set_ylabel("SWE-Bench-Verified (%)", fontsize=9)
ax.grid(True, linestyle=":", linewidth=0.4, color="grey", alpha=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="both", labelsize=8)
ax.set_ylim(24.0, 30.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=8)

# warn that values are placeholders
fig.text(0.99, 0.99, "PLACEHOLDER VALUES",
         color="#c0392b", fontsize=7, ha="right", va="top",
         alpha=0.55)

plt.tight_layout(pad=0.4)
out = ROOT / "figs" / "token_budget.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
