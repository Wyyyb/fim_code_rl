"""Teaser figure (Fig 1): three-panel matplotlib.
  (left)   function call site, four-color decomposition
  (middle) agent step, four-color decomposition
  (right)  SWE-Bench-V grouped bars across 7B/14B/32B
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent

# colour palette (consistent with isomorphism / panels)
C_CTX  = "#cfe2f3"   # blue   – context
C_ACT  = "#fce5cd"   # orange – action / call
C_RET  = "#d9ead3"   # green  – return / observation
C_CONT = "#e6d5f0"   # violet – continuation / downstream
EDGE   = {C_CTX: "#3b6fb8", C_ACT: "#cc7a00",
          C_RET: "#3a8c4f", C_CONT: "#7a4a8e"}


def add_block(ax, x, y, w, h, color, text, label,
              monospace=True, fs_text=8.0, fs_label=7.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        linewidth=0.7, facecolor=color, edgecolor=EDGE[color], zorder=2,
    )
    ax.add_patch(rect)
    fam = "monospace" if monospace else "serif"
    ax.text(x + 0.022, y + h / 2, text, fontsize=fs_text, va="center",
            ha="left", family=fam, zorder=3)
    ax.text(x + w + 0.012, y + h / 2, label, fontsize=fs_label, va="center",
            ha="left", color=EDGE[color], style="italic", zorder=3)


fig = plt.figure(figsize=(13.0, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.10, 1.05], wspace=0.45,
                      left=0.012, right=0.99, bottom=0.03, top=0.93)

# ----- left panel: function call site -----
axL = fig.add_subplot(gs[0, 0])
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.axis("off")
axL.text(0.5, 0.97, "Function call site", fontsize=10.5,
         fontweight="bold", ha="center", va="top")

ROW_H = 0.13
GAP   = 0.022
y0    = 0.78
def lrow(i): return y0 - i * (ROW_H + GAP)

add_block(axL, 0.02, lrow(0), 0.65, ROW_H, C_CTX,
          "cfg = load_config(path)\nx   = preprocess(cfg, raw)",
          "pre-call context")
add_block(axL, 0.02, lrow(1), 0.65, ROW_H, C_ACT,
          "y = transform(x, cfg.mode)",
          "call")
add_block(axL, 0.02, lrow(2), 0.65, ROW_H, C_RET,
          "# returns: tensor (B, D)",
          "return")
add_block(axL, 0.02, lrow(3), 0.65, ROW_H, C_CONT,
          "loss = criterion(y, target)\nloss.backward()",
          "downstream use")

# ----- middle panel: agent step -----
axM = fig.add_subplot(gs[0, 1])
axM.set_xlim(0, 1); axM.set_ylim(0, 1)
axM.axis("off")
axM.text(0.5, 0.97, "Coding-agent step", fontsize=10.5,
         fontweight="bold", ha="center", va="top")

add_block(axM, 0.02, lrow(0), 0.65, ROW_H, C_CTX,
          "history h_t : prior tool calls,\nfile contents, plan so far",
          "context", monospace=False)
add_block(axM, 0.02, lrow(1), 0.65, ROW_H, C_ACT,
          "action a_t : run_tests(...)",
          "action", monospace=False)
add_block(axM, 0.02, lrow(2), 0.65, ROW_H, C_RET,
          "obs. o_{t+1} : AssertionError ...",
          "external return", monospace=False)
add_block(axM, 0.02, lrow(3), 0.65, ROW_H, C_CONT,
          "next step: reason about error,\npropose next edit",
          "continuation", monospace=False)

# arrow between left and middle panels
# the gap between axes is at gs wspace=0.28, located near x≈0.35
fig.text(0.346, 0.52, r"$\Longleftrightarrow$", fontsize=24,
         color="#333333", ha="center", va="center", fontweight="bold")
fig.text(0.346, 0.40, "isomorphic", fontsize=8.5, color="#333333",
         ha="center", va="center", style="italic")

# ----- right panel: SWE-Bench-V grouped bars -----
axR = fig.add_subplot(gs[0, 2])
sizes = ["7B", "14B", "32B"]
post_v = [15.0, 26.2, 31.8]
ours_v = [17.8, 29.2, 35.1]
post_l = [11.3, 18.0, 24.7]
ours_l = [15.0, 22.0, 26.8]
gains  = [o - p for p, o in zip(post_v, ours_v)]

x = np.arange(len(sizes))
w = 0.18
axR.bar(x - 1.5*w, post_v, w, color="#bbbbbb", edgecolor="#666666",
        linewidth=0.4, label="post-only (V)")
axR.bar(x - 0.5*w, ours_v, w, color="#c0392b", edgecolor="#7a2418",
        linewidth=0.4, label="+ FIM (V)")
axR.bar(x + 0.5*w, post_l, w, color="#dddddd", edgecolor="#888888",
        linewidth=0.4, hatch="//", label="post-only (L)")
axR.bar(x + 1.5*w, ours_l, w, color="#e88675", edgecolor="#a34c3a",
        linewidth=0.4, hatch="//", label="+ FIM (L)")

# annotate gains for verified
for xi, p, o in zip(x, post_v, ours_v):
    axR.annotate(f"+{o - p:.1f}",
                 xy=(xi - 0.5*w, o), xytext=(0, 4),
                 textcoords="offset points",
                 ha="center", fontsize=7.5, color="#7a2418",
                 fontweight="bold")

axR.set_xticks(x)
axR.set_xticklabels(sizes, fontsize=9)
axR.set_xlabel("Model size", fontsize=9)
axR.set_ylabel("SWE-Bench (%)", fontsize=9)
axR.set_ylim(0, 45)
axR.tick_params(axis="y", labelsize=8)
axR.grid(axis="y", linestyle=":", linewidth=0.3, color="grey", alpha=0.6)
axR.set_axisbelow(True)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0),
           fontsize=6.8, frameon=False, ncol=2,
           handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
axR.set_title("Mid-training gain across scales",
              fontsize=10.5, fontweight="bold", pad=4)

out = ROOT / "figs" / "teaser.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
