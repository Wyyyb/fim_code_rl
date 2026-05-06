"""Method figure (Fig 2) -- function-aware FIM target selection.

Single-row 3-panel layout sized to NeurIPS \\linewidth (~7in wide, ~1.9in tall).
  (a) Program dependency graph (condensed code + PDG diagram)
  (b) Score breakdown for Calculator.total (two stacked bars)
  (c) Selection in the H-I plane (scatter + threshold contour)
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
    "font.size": 8,
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
SEL_BG     = "#fce0e6"
FILT_FG    = "#9aa0a6"

H_COLS = ["#3a6896", "#6c9bc4", "#b3cee5"]
I_COLS = ["#d99046", "#e2ad75", "#d6c198", "#cc8e96", "#a26988"]

THR_FILL = "#fdeaee"
THR_EDGE = "#c63452"
GRID_FG  = "#777777"
TEXT_DK  = "#202020"
TEXT_MD  = "#505050"

# ============================================================================
# Figure layout
# ============================================================================
FIG_W, FIG_H = 7.0, 1.95

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.4, 1.0, 1.1],
    wspace=0.10,
    left=0.008, right=0.992, bottom=0.020, top=0.980,
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
        boxstyle="round,pad=0.0,rounding_size=0.022",
        linewidth=0.6, edgecolor="#888", facecolor="white", zorder=0,
    )
    ax.add_patch(rect)


def panel_title(ax, text, x=0.025, y=0.975):
    ax.text(x, y, text, fontsize=7.8, weight="bold",
            ha="left", va="top", color=TEXT_DK, family="serif")


for ax in (axA, axB, axC):
    panel_border(ax)


# ============================================================================
# PANEL A -- code listing + PDG
# ============================================================================
panel_title(axA, "(a) Program dependency graph")

# Condensed code: keep signatures + a couple of key call lines.
CODE_LINES = [
    ("# calc.py", "comment"),
    ("def add(a, b): ...", "code"),
    ("def is_int(x) -> bool: ...", "code"),
    ("", "blank"),
    ("class Calculator:", "code"),
    ("    def __init__(self): ...", "code"),
    ("    def push(self, v): ...", "code"),
    ("    def total(self) -> int:", "code"),
    ('        """Sum positive int entries."""', "doc"),
    ("        for v in self.history:", "code"),
    ("            if is_int(v) and v > 0:", "code"),
    ("                s = add(s, v)", "code"),
    ("        # ...elided...", "doc"),
    ("    def mean(self) -> float:", "code"),
    ("        return self.total() / n", "code"),
]

# Code box (left ~48% of panel)
code_x0, code_x1 = 0.020, 0.475
code_y0, code_y1 = 0.150, 0.905
axA.add_patch(FancyBboxPatch(
    (code_x0, code_y0), code_x1 - code_x0, code_y1 - code_y0,
    boxstyle="round,pad=0.0,rounding_size=0.014",
    linewidth=0.45, edgecolor="#bbb", facecolor="#fafafa", zorder=1,
))

n_lines = len(CODE_LINES)
line_h = (code_y1 - code_y0 - 0.020) / n_lines
y_top = code_y1 - 0.012
for i, (line, kind) in enumerate(CODE_LINES):
    if kind == "blank":
        continue
    y = y_top - (i + 0.5) * line_h
    if kind in ("comment", "doc"):
        col, style = "#7a8896", "italic"
    else:
        col, style = TEXT_DK, "normal"
    axA.text(code_x0 + 0.010, y, line, fontsize=5.6,
             family="monospace", color=col, style=style,
             ha="left", va="center", zorder=2)

# ---- PDG nodes (right side of panel A) ----
NODE_W, NODE_H = 0.150, 0.090

nodes = {
    "add":      dict(xy=(0.610, 0.78), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "is_int":   dict(xy=(0.610, 0.62), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "__init__": dict(xy=(0.870, 0.78), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "push":     dict(xy=(0.870, 0.62), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "total":    dict(xy=(0.870, 0.46), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "mean":     dict(xy=(0.870, 0.30), fill=CLSFN_FILL, edge=CLSFN_EDGE),
}


def draw_node(ax, label, xy, fill, edge):
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - NODE_W / 2, cy - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        linewidth=0.55, edgecolor=edge, facecolor=fill, zorder=4,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, fontsize=6.0, family="monospace",
            ha="center", va="center", color=TEXT_DK, zorder=6)


# Class container box
cls_pad_x, cls_pad_y_top, cls_pad_y_bot = 0.022, 0.030, 0.018
cls_x0 = nodes["__init__"]["xy"][0] - NODE_W / 2 - cls_pad_x
cls_x1 = nodes["__init__"]["xy"][0] + NODE_W / 2 + cls_pad_x
cls_y1 = nodes["__init__"]["xy"][1] + NODE_H / 2 + cls_pad_y_top
cls_y0 = nodes["mean"]["xy"][1] - NODE_H / 2 - cls_pad_y_bot
axA.add_patch(FancyBboxPatch(
    (cls_x0, cls_y0), cls_x1 - cls_x0, cls_y1 - cls_y0,
    boxstyle="round,pad=0.0,rounding_size=0.024",
    linewidth=0.5, edgecolor=CLSFN_EDGE + "aa",
    facecolor=CLSBG, linestyle=(0, (3, 2)), zorder=2,
))
axA.text(cls_x0 + 0.010, cls_y1 - 0.004, "class Calculator",
         fontsize=5.4, family="monospace", color="#345c44",
         style="italic", ha="left", va="top", zorder=3)


def sibling_edge(ax, p, q, rad=0.55):
    arr = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-", linewidth=0.6, color="#5e7a68",
        linestyle=(0, (2.0, 1.6)), zorder=3,
    )
    ax.add_patch(arr)


sib_x = nodes["__init__"]["xy"][0] + NODE_W / 2 - 0.005
for a, b, rad in [
    ("__init__", "push",  0.55),
    ("push",     "total", 0.55),
    ("total",    "mean",  0.55),
]:
    pa = (sib_x, nodes[a]["xy"][1])
    pb = (sib_x, nodes[b]["xy"][1])
    sibling_edge(axA, pa, pb, rad=rad)


def call_edge(ax, src, dst, rad=0.0, color="#404040"):
    p, q = nodes[src]["xy"], nodes[dst]["xy"]
    arr = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=8.0,
        linewidth=0.75, color=color,
        shrinkA=11, shrinkB=11, zorder=5,
    )
    ax.add_patch(arr)


call_edge(axA, "total", "is_int", rad=0.18)
call_edge(axA, "total", "add",   rad=-0.22)
call_edge(axA, "mean",  "total", rad=0.0)

for name, attr in nodes.items():
    draw_node(axA, name, attr["xy"], attr["fill"], attr["edge"])

# ---- compact legend at bottom of panel A ----
leg_y = 0.075
axA.annotate(
    "", xy=(0.085, leg_y), xytext=(0.035, leg_y),
    arrowprops=dict(arrowstyle="-|>", linewidth=0.75,
                    color="#404040", mutation_scale=7),
)
axA.text(0.095, leg_y, r"$\mathcal{E}_{\mathrm{call}}$",
         fontsize=6.6, va="center", ha="left", color=TEXT_DK)

axA.add_line(Line2D(
    [0.180, 0.235], [leg_y, leg_y],
    linewidth=0.7, linestyle=(0, (2.5, 1.8)), color="#5e7a68",
))
axA.text(0.245, leg_y, r"$\mathcal{E}_{\mathrm{sib}}$",
         fontsize=6.6, va="center", ha="left", color=TEXT_DK)

axA.add_patch(Rectangle((0.330, leg_y - 0.018), 0.024, 0.034,
                        facecolor=TOPFN_FILL, edgecolor=TOPFN_EDGE,
                        linewidth=0.4))
axA.text(0.362, leg_y, "top-level",
         fontsize=6.2, va="center", ha="left", color=TEXT_DK)

axA.add_patch(Rectangle((0.475, leg_y - 0.018), 0.024, 0.034,
                        facecolor=CLSFN_FILL, edgecolor=CLSFN_EDGE,
                        linewidth=0.4))
axA.text(0.508, leg_y, "in-class",
         fontsize=6.2, va="center", ha="left", color=TEXT_DK)


# ============================================================================
# PANEL B -- score breakdown for Calculator.total (simplified)
# ============================================================================
panel_title(axB, r"(b) Score breakdown for $\mathtt{total}$")

bar_x0, bar_x1 = 0.060, 0.760
bar_w = bar_x1 - bar_x0
bar_h = 0.110


def draw_stacked_bar(ax, y, values, colors, total, total_label_text,
                     total_color):
    x = bar_x0
    for v, c in zip(values, colors):
        w = (v / total) * bar_w
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=c,
                               edgecolor="none", zorder=3))
        r, g, b = mpl.colors.to_rgb(c)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        tcol = "white" if lum < 0.62 else TEXT_DK
        if w > 0.040:
            ax.text(x + w / 2, y + bar_h / 2, f"{v:.2f}",
                    fontsize=5.8, ha="center", va="center",
                    color=tcol, family="serif", zorder=4)
        x += w
    ax.add_patch(Rectangle((bar_x0, y), bar_w, bar_h,
                           facecolor="none", edgecolor="#888",
                           linewidth=0.45, zorder=5))
    ax.text(bar_x1 + 0.014, y + bar_h / 2, total_label_text,
            fontsize=7.6, ha="left", va="center",
            weight="bold", color=total_color, family="serif")


def draw_segment_names(ax, y, values, total, names, colors):
    seg_left = bar_x0
    for v, name, c in zip(values, names, colors):
        seg_w = (v / total) * bar_w
        cx = seg_left + seg_w / 2
        ax.text(cx, y, name, fontsize=5.2,
                ha="center", va="center", color=c, family="serif",
                weight="bold")
        seg_left += seg_w


# ---- Hhat bar ----
H_TOTAL  = 0.40
H_VALS   = [0.08, 0.20, 0.12]
H_NAMES  = ["LoC", "CC", "depth"]

H_BAR_Y = 0.640
draw_stacked_bar(axB, H_BAR_Y, H_VALS, H_COLS, H_TOTAL,
                 r"$\hat{H}\!=\!0.40$", H_COLS[0])
draw_segment_names(axB, H_BAR_Y - 0.060, H_VALS, H_TOTAL,
                   H_NAMES, H_COLS)

# ---- Ihat bar ----
I_TOTAL  = 0.48
I_VALS   = [0.06, 0.13, 0.10, 0.05, 0.14]
I_NAMES  = ["caller", "callee", "sig", "doc", "class"]

I_BAR_Y = 0.380
draw_stacked_bar(axB, I_BAR_Y, I_VALS, I_COLS, I_TOTAL,
                 r"$\hat{I}\!=\!0.48$", I_COLS[-1])
draw_segment_names(axB, I_BAR_Y - 0.060, I_VALS, I_TOTAL,
                   I_NAMES, I_COLS)

# ---- combined FIM at the bottom ----
fim_box_y0, fim_box_y1 = 0.060, 0.220
axB.add_patch(FancyBboxPatch(
    (0.045, fim_box_y0), 0.910, fim_box_y1 - fim_box_y0,
    boxstyle="round,pad=0.005,rounding_size=0.020",
    linewidth=0.5, edgecolor=SEL_FG, facecolor=SEL_BG, zorder=2,
))
axB.text(0.500, 0.140,
         r"$\mathrm{FIM}=\hat{H}\hat{I}/(\hat{H}+\hat{I})\approx 0.22"
         r"\;\geq\;\tau\!=\!0.20\;\Rightarrow\;\mathrm{selected}$",
         fontsize=7.4, ha="center", va="center", color=SEL_FG,
         family="serif", weight="bold")


# ============================================================================
# PANEL C -- selection in H-I plane
# ============================================================================
panel_title(axC, r"(c) Selection in $\hat{H}$--$\hat{I}$ plane")

inset = axC.inset_axes([0.190, 0.160, 0.770, 0.700],
                       transform=axC.transAxes)
inset.set_xlim(0, 1.0)
inset.set_ylim(0, 1.0)
inset.set_xlabel(r"complexity $\hat{H}$", fontsize=7.6,
                 color=TEXT_DK, labelpad=1.5)
inset.set_ylabel(r"inferability $\hat{I}$", fontsize=7.6,
                 color=TEXT_DK, labelpad=1.5)
inset.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
inset.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
inset.tick_params(axis="both", labelsize=6.4, length=2.0,
                  colors=GRID_FG, width=0.4, pad=1.5)
for spine in inset.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#999")

tau = 0.20
H_grid = np.linspace(0, 1.0, 320)
I_grid = np.linspace(0, 1.0, 320)
HH, II = np.meshgrid(H_grid, I_grid)
FIM = (HH * II) / (HH + II + 1e-9)

inset.contourf(HH, II, FIM, levels=[tau, 10.0],
               colors=[THR_FILL], zorder=1)
inset.contour(HH, II, FIM, levels=[tau],
              colors=[THR_EDGE], linewidths=1.0,
              linestyles="solid", zorder=2)

candidates = [
    ("add",    0.04, 0.18, "filt"),
    ("is_int", 0.06, 0.42, "filt"),
    ("push",   0.08, 0.32, "filt"),
    ("mean",   0.08, 0.55, "filt"),
    ("total",  0.40, 0.48, "sel"),
]

for name, H, I, status in candidates:
    if status == "filt":
        inset.plot(H, I, marker="x", markersize=5.5,
                   markeredgewidth=1.1, color=FILT_FG, zorder=6)
        inset.text(H + 0.030, I, name, fontsize=6.4,
                   color=FILT_FG, ha="left", va="center",
                   family="serif")
    else:  # sel
        inset.plot(H, I, marker="o", markersize=7.5,
                   markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
                   markeredgewidth=0.6, zorder=8)
        inset.text(H, I + 0.075, name,
                   fontsize=7.0, color=SEL_FG, ha="center", va="bottom",
                   family="serif", weight="bold")

leg_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
           markersize=5.5, label="selected"),
    Line2D([0], [0], marker="x", color=FILT_FG,
           markersize=5.5, markeredgewidth=1.1,
           linestyle="None", label="filtered"),
    Line2D([0], [0], color=THR_EDGE, linewidth=1.0,
           linestyle="solid", label=r"$\mathrm{FIM}=\tau$"),
]
leg = inset.legend(handles=leg_handles, loc="lower right",
                   fontsize=6.2, frameon=True,
                   framealpha=0.95, edgecolor="#bbb",
                   handlelength=1.2, handletextpad=0.4,
                   borderpad=0.3, labelspacing=0.25)
leg.get_frame().set_linewidth(0.4)

# ============================================================================
# Save
# ============================================================================
pdf_path = OUT / "pdg_score.pdf"
png_path = OUT / "pdg_score.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")
