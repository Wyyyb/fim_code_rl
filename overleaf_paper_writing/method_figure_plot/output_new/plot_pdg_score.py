"""Method figure (Fig 2) — function-aware FIM target selection.

Single-row, deliberately minimal layout:
  (a) Program dependency graph (PDG) on the running calculator example
  (b) Score schematic: H and I as stacked bars, with the harmonic-mean form
  (c) Selection in the (H, I) plane: threshold curve and candidates

All numerical detail and formulas live in the body / appendix; this figure
only carries the visual intuition.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

OUT = Path(__file__).resolve().parent

# ============================================================================
# Palette
# ============================================================================
TOPFN_FILL = "#dbeef0"
TOPFN_EDGE = "#5a8b94"
CLSFN_FILL = "#dbeadc"
CLSFN_EDGE = "#52866a"
CLSBG      = "#eef6f0"
SEL_FG     = "#c63452"
FILT_FG    = "#9aa0a6"

H_COLS = ["#3a6896", "#6c9bc4", "#b3cee5"]
I_COLS = ["#d99046", "#e2ad75", "#d6c198", "#cc8e96", "#a26988"]

THR_FILL = "#fdeaee"
THR_EDGE = "#c63452"
TEXT_DK  = "#202020"
TEXT_MD  = "#505050"

# ============================================================================
# Figure layout — single row, short
# ============================================================================
FIG_W, FIG_H = 9.0, 3.2

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.05, 0.95, 1.10],
    wspace=0.06,
    left=0.010, right=0.990, bottom=0.030, top=0.970,
)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

for ax in (axA, axB, axC):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_border(ax, pad=0.005):
    rect = FancyBboxPatch(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0.0,rounding_size=0.020",
        linewidth=0.6, edgecolor="#888", facecolor="white", zorder=0,
    )
    ax.add_patch(rect)


def panel_title(ax, text, x=0.025, y=0.945):
    ax.text(x, y, text, fontsize=11.0, weight="bold",
            ha="left", va="top", color=TEXT_DK, family="serif")


for ax in (axA, axB, axC):
    panel_border(ax)


# ============================================================================
# PANEL A — PDG only (no code)
# ============================================================================
panel_title(axA, "(a) Program dependency graph")

NODE_W, NODE_H = 0.260, 0.135

nodes = {
    "add":      dict(xy=(0.265, 0.74), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "is_int":   dict(xy=(0.265, 0.51), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "__init__": dict(xy=(0.755, 0.755), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "push":     dict(xy=(0.755, 0.575), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "total":    dict(xy=(0.755, 0.395), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "mean":     dict(xy=(0.755, 0.215), fill=CLSFN_FILL, edge=CLSFN_EDGE),
}


def draw_node(ax, label, xy, fill, edge):
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - NODE_W / 2, cy - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        linewidth=0.6, edgecolor=edge, facecolor=fill, zorder=4,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, fontsize=9.0, family="monospace",
            ha="center", va="center", color=TEXT_DK, zorder=6)


# Class container box
cls_pad_x, cls_pad_y_top, cls_pad_y_bot = 0.026, 0.045, 0.028
cls_x0 = nodes["__init__"]["xy"][0] - NODE_W / 2 - cls_pad_x
cls_x1 = nodes["__init__"]["xy"][0] + NODE_W / 2 + cls_pad_x
cls_y1 = nodes["__init__"]["xy"][1] + NODE_H / 2 + cls_pad_y_top
cls_y0 = nodes["mean"]["xy"][1] - NODE_H / 2 - cls_pad_y_bot
axA.add_patch(FancyBboxPatch(
    (cls_x0, cls_y0), cls_x1 - cls_x0, cls_y1 - cls_y0,
    boxstyle="round,pad=0.0,rounding_size=0.026",
    linewidth=0.5, edgecolor=CLSFN_EDGE + "aa",
    facecolor=CLSBG, zorder=2,
))
axA.text(cls_x0 + 0.018, cls_y1 - 0.020, "class Calculator",
         fontsize=8.5, family="monospace", color="#345c44",
         style="italic", ha="left", va="top", zorder=3)


def sibling_edge(ax, p, q, rad=0.55):
    arr = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-", linewidth=0.8, color="#5e7a68",
        linestyle=(0, (2.0, 1.6)), zorder=3,
    )
    ax.add_patch(arr)


sib_x = nodes["__init__"]["xy"][0] + NODE_W / 2
for a, b, rad in [
    ("__init__", "push",  0.50),
    ("push",     "total", 0.50),
    ("total",    "mean",  0.50),
    ("__init__", "mean",  0.95),
]:
    pa = (sib_x, nodes[a]["xy"][1])
    pb = (sib_x, nodes[b]["xy"][1])
    sibling_edge(axA, pa, pb, rad=rad)


def call_edge(ax, src, dst, rad=0.0, color="#404040"):
    p, q = nodes[src]["xy"], nodes[dst]["xy"]
    arr = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=11.0,
        linewidth=1.0, color=color,
        shrinkA=14, shrinkB=14, zorder=5,
    )
    ax.add_patch(arr)


call_edge(axA, "total", "is_int", rad=0.18)
call_edge(axA, "total", "add",   rad=-0.22)
call_edge(axA, "mean",  "total", rad=0.0)

for name, attr in nodes.items():
    draw_node(axA, name, attr["xy"], attr["fill"], attr["edge"])

# legend (kept tight inside the frame)
leg_y = 0.060
axA.annotate(
    "", xy=(0.115, leg_y), xytext=(0.045, leg_y),
    arrowprops=dict(arrowstyle="-|>", linewidth=0.95,
                    color="#404040", mutation_scale=9),
)
axA.text(0.130, leg_y, r"$\mathcal{E}_{\mathrm{call}}$",
         fontsize=8.8, va="center", ha="left", color=TEXT_DK)

axA.add_line(Line2D(
    [0.260, 0.330], [leg_y, leg_y],
    linewidth=1.0, linestyle=(0, (2.5, 1.8)), color="#5e7a68",
))
axA.text(0.345, leg_y, r"$\mathcal{E}_{\mathrm{sib}}$",
         fontsize=8.8, va="center", ha="left", color=TEXT_DK)

axA.add_patch(Rectangle((0.470, leg_y - 0.018), 0.038, 0.038,
                        facecolor=CLSFN_FILL, edgecolor=CLSFN_EDGE,
                        linewidth=0.5))
axA.text(0.515, leg_y, "method",
         fontsize=8.6, va="center", ha="left", color=TEXT_DK)

axA.add_patch(Rectangle((0.700, leg_y - 0.018), 0.038, 0.038,
                        facecolor=TOPFN_FILL, edgecolor=TOPFN_EDGE,
                        linewidth=0.5))
axA.text(0.745, leg_y, "top-level",
         fontsize=8.6, va="center", ha="left", color=TEXT_DK)


# ============================================================================
# PANEL B — minimal score schematic: two stacked bars
# ============================================================================
panel_title(axB, r"(b) Score $=\hat{H}\hat{I}/(\hat{H}+\hat{I})$")

bar_x0, bar_x1 = 0.180, 0.910
bar_w = bar_x1 - bar_x0
bar_h = 0.150

# Hhat bar — three components (LoC, CC, D), proportions illustrative
H_FRACS = [0.20, 0.50, 0.30]
H_BAR_Y = 0.595
axB.text(bar_x0 - 0.025, H_BAR_Y + bar_h / 2, r"$\hat{H}$",
         fontsize=17, ha="right", va="center",
         weight="bold", color=H_COLS[0], family="serif")
x = bar_x0
for f, c in zip(H_FRACS, H_COLS):
    w = f * bar_w
    axB.add_patch(Rectangle((x, H_BAR_Y), w, bar_h,
                            facecolor=c, edgecolor="none", zorder=3))
    x += w
axB.add_patch(Rectangle((bar_x0, H_BAR_Y), bar_w, bar_h,
                        facecolor="none", edgecolor="#888",
                        linewidth=0.5, zorder=5))
# small caption below the bar
axB.text(0.5, H_BAR_Y - 0.045,
         r"$\mathrm{LoC},\;\mathrm{CC},\;\mathrm{nesting\;depth}$",
         fontsize=8.6, ha="center", va="top",
         color=TEXT_MD, family="serif", style="italic")

# Ihat bar — five components
I_FRACS = [0.13, 0.27, 0.21, 0.10, 0.29]
I_BAR_Y = 0.270
axB.text(bar_x0 - 0.025, I_BAR_Y + bar_h / 2, r"$\hat{I}$",
         fontsize=17, ha="right", va="center",
         weight="bold", color=I_COLS[-1], family="serif")
x = bar_x0
for f, c in zip(I_FRACS, I_COLS):
    w = f * bar_w
    axB.add_patch(Rectangle((x, I_BAR_Y), w, bar_h,
                            facecolor=c, edgecolor="none", zorder=3))
    x += w
axB.add_patch(Rectangle((bar_x0, I_BAR_Y), bar_w, bar_h,
                        facecolor="none", edgecolor="#888",
                        linewidth=0.5, zorder=5))
axB.text(0.5, I_BAR_Y - 0.045,
         r"$\mathrm{caller},\,\mathrm{callee},\,\mathrm{sig},\,"
         r"\mathrm{doc},\,\mathrm{class}$",
         fontsize=8.6, ha="center", va="top",
         color=TEXT_MD, family="serif", style="italic")

# tagline at the bottom
axB.text(0.5, 0.085,
         r"select iff $\hat{H}\hat{I}/(\hat{H}+\hat{I})\geq\tau$",
         fontsize=10.0, ha="center", va="center",
         color=SEL_FG, family="serif", weight="bold")


# ============================================================================
# PANEL C — Hhat-Ihat plane with threshold and candidates
# ============================================================================
panel_title(axC, r"(c) Selection in $\hat{H}$–$\hat{I}$ plane")

inset = axC.inset_axes([0.165, 0.155, 0.795, 0.700],
                       transform=axC.transAxes)
inset.set_xlim(0, 1.05)
inset.set_ylim(0, 1.0)
inset.set_xlabel(r"complexity $\hat{H}$", fontsize=9.5,
                 color=TEXT_DK, labelpad=2)
inset.set_ylabel(r"inferability $\hat{I}$", fontsize=9.5,
                 color=TEXT_DK, labelpad=2)
inset.tick_params(axis="both", labelsize=8.0, length=2.5,
                  colors="#777", width=0.4)
for spine in inset.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#999")

tau = 0.20
H_grid = np.linspace(0, 1.05, 320)
I_grid = np.linspace(0, 1.0, 320)
HH, II = np.meshgrid(H_grid, I_grid)
FIM = (HH * II) / (HH + II + 1e-9)

inset.contourf(HH, II, FIM, levels=[tau, 10.0],
               colors=[THR_FILL], zorder=1)
inset.contour(HH, II, FIM, levels=[tau],
              colors=[THR_EDGE], linewidths=1.1,
              linestyles="solid", zorder=2)

# tau label inline along the curve, at H = 0.7 (clear of points)
H_lab = 0.70
I_lab = tau * H_lab / (H_lab - tau)
inset.text(H_lab + 0.025, I_lab + 0.01, fr"$\mathrm{{FIM}}=\tau$",
           fontsize=8.4, color=THR_EDGE, ha="left", va="bottom",
           family="serif", weight="bold")

# candidate points (4 filtered + 1 selected)
candidates = [
    ("add",    0.03, 0.18, "filt"),
    ("push",   0.05, 0.30, "filt"),
    ("is_int", 0.07, 0.42, "filt"),
    ("mean",   0.08, 0.55, "filt"),
    ("total",  0.40, 0.48, "sel"),
]

for name, H, I, status in candidates:
    if status == "filt":
        inset.plot(H, I, marker="x", markersize=8.0,
                   markeredgewidth=1.4, color=FILT_FG, zorder=6)
        inset.text(H + 0.030, I, name, fontsize=8.4,
                   color=FILT_FG, ha="left", va="center",
                   family="serif")
    elif status == "sel":
        inset.plot(H, I, marker="o", markersize=18,
                   markerfacecolor="none", markeredgecolor=SEL_FG,
                   markeredgewidth=0.9, zorder=7)
        inset.plot(H, I, marker="o", markersize=10,
                   markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
                   markeredgewidth=0.6, zorder=8)
        inset.text(H, I + 0.085, name + r" $\checkmark$",
                   fontsize=10, color=SEL_FG, ha="center", va="bottom",
                   family="serif", weight="bold")

# legend (top-right where there are no points)
leg_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
           markersize=8, label="selected"),
    Line2D([0], [0], marker="x", color=FILT_FG,
           markersize=8, markeredgewidth=1.4,
           linestyle="None", label="filtered"),
    mpatches.Patch(facecolor=THR_FILL, edgecolor=THR_EDGE,
                   linewidth=0.5, label=r"$\mathrm{FIM}\geq\tau$"),
]
leg = inset.legend(handles=leg_handles, loc="upper right",
                   fontsize=8.0, frameon=True,
                   framealpha=0.95, edgecolor="#bbb",
                   handlelength=1.2, handletextpad=0.5,
                   borderpad=0.4, labelspacing=0.32)
leg.get_frame().set_linewidth(0.4)

# ============================================================================
# Save
# ============================================================================
pdf_path = OUT / "pdg_score.pdf"
png_path = OUT / "pdg_score.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")
