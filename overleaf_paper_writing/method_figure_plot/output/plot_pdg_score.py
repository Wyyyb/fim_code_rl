"""Method figure (Fig 2) — function-aware FIM target selection.

Layout (taller-than-wide so text stays legible at \\linewidth in NeurIPS):
  Row 1 (full width)  : (a) Program dependency graph (code + PDG)
  Row 2 (split 50/50) : (b) score breakdown        (c) FIM selection in H-I plane

The example file is intentionally simple (a calculator class) so the PDG and
the score components are easy to follow.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
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

CONTOUR  = "#bdbdbd"
THR_FILL = "#fdeaee"
THR_EDGE = "#c63452"
GRID_FG  = "#777777"
TEXT_DK  = "#202020"
TEXT_MD  = "#505050"
HYPER    = "#7a7a7a"

# ============================================================================
# Figure layout
# ============================================================================
FIG_W, FIG_H = 9.0, 7.6

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[0.95, 1.10],
    width_ratios=[1.0, 1.0],
    hspace=0.07, wspace=0.07,
    left=0.012, right=0.988, bottom=0.018, top=0.985,
)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

for ax in (axA, axB, axC):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_border(ax, pad=0.005):
    rect = FancyBboxPatch(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.6, edgecolor="#888", facecolor="white", zorder=0,
    )
    ax.add_patch(rect)


def panel_title(ax, text, x=0.020, y=0.965):
    ax.text(x, y, text, fontsize=10.5, weight="bold",
            ha="left", va="top", color=TEXT_DK, family="serif")


for ax in (axA, axB, axC):
    panel_border(ax)


# ============================================================================
# PANEL A — code listing + PDG
# ============================================================================
panel_title(axA, "(a) Program dependency graph")

CODE_LINES = [
    ("# calc.py", "comment"),
    ("def add(a, b):", "code"),
    ("    return a + b", "code"),
    ("", "blank"),
    ("def is_int(x) -> bool:", "code"),
    ('    """Check if x is an integer."""', "doc"),
    ("    return isinstance(x, int)", "code"),
    ("", "blank"),
    ("class Calculator:", "code"),
    ("    def __init__(self):", "code"),
    ("        self.history = []", "code"),
    ("    def push(self, v):", "code"),
    ("        self.history.append(v)", "code"),
    ("    def total(self) -> int:", "code"),
    ('        """Sum positive integer entries."""', "doc"),
    ("        s, n = 0, 0", "code"),
    ("        for v in self.history:", "code"),
    ("            if is_int(v):", "code"),
    ("                if v > 0:", "code"),
    ("                    s = add(s, v)", "code"),
    ("                    n += 1", "code"),
    ("        if n == 0: return 0", "code"),
    ("        return s", "code"),
    ("    def mean(self) -> float:", "code"),
    ("        return self.total() / max(len(self.history), 1)", "code"),
]

# Code box
code_x0, code_x1 = 0.018, 0.420
code_y0, code_y1 = 0.090, 0.910
axA.add_patch(FancyBboxPatch(
    (code_x0, code_y0), code_x1 - code_x0, code_y1 - code_y0,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    linewidth=0.45, edgecolor="#bbb", facecolor="#fafafa", zorder=1,
))

n_lines = len(CODE_LINES)
line_h = (code_y1 - code_y0 - 0.030) / n_lines
y_top = code_y1 - 0.018
for i, (line, kind) in enumerate(CODE_LINES):
    if kind == "blank":
        continue
    y = y_top - (i + 0.5) * line_h
    if kind in ("comment", "doc"):
        col, style = "#7a8896", "italic"
    else:
        col, style = TEXT_DK, "normal"
    axA.text(code_x0 + 0.008, y, line, fontsize=7.5,
             family="monospace", color=col, style=style,
             ha="left", va="center", zorder=2)

# Arrow "AST parse →" between code and PDG
axA.annotate(
    "", xy=(0.500, 0.50), xytext=(0.445, 0.50),
    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#666"),
)
axA.text(0.4725, 0.555, "AST\nparse", fontsize=7.5,
         color=TEXT_MD, ha="center", va="bottom", family="serif",
         linespacing=1.0)

# ---- PDG nodes ----
NODE_W, NODE_H = 0.130, 0.105

nodes = {
    "add":      dict(xy=(0.610, 0.78), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "is_int":   dict(xy=(0.610, 0.62), fill=TOPFN_FILL, edge=TOPFN_EDGE),
    "__init__": dict(xy=(0.870, 0.83), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "push":     dict(xy=(0.870, 0.66), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "total":    dict(xy=(0.870, 0.49), fill=CLSFN_FILL, edge=CLSFN_EDGE),
    "mean":     dict(xy=(0.870, 0.32), fill=CLSFN_FILL, edge=CLSFN_EDGE),
}


def draw_node(ax, label, xy, fill, edge):
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - NODE_W / 2, cy - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.6, edgecolor=edge, facecolor=fill, zorder=4,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, fontsize=8.0, family="monospace",
            ha="center", va="center", color=TEXT_DK, zorder=6)


# Class container box
cls_pad_x, cls_pad_y_top, cls_pad_y_bot = 0.020, 0.040, 0.020
cls_x0 = nodes["__init__"]["xy"][0] - NODE_W / 2 - cls_pad_x
cls_x1 = nodes["__init__"]["xy"][0] + NODE_W / 2 + cls_pad_x
cls_y1 = nodes["__init__"]["xy"][1] + NODE_H / 2 + cls_pad_y_top
cls_y0 = nodes["mean"]["xy"][1] - NODE_H / 2 - cls_pad_y_bot
axA.add_patch(FancyBboxPatch(
    (cls_x0, cls_y0), cls_x1 - cls_x0, cls_y1 - cls_y0,
    boxstyle="round,pad=0.0,rounding_size=0.022",
    linewidth=0.5, edgecolor=CLSFN_EDGE + "aa",
    facecolor=CLSBG, zorder=2,
))
axA.text(cls_x0 + 0.012, cls_y1 - 0.014 + 0.008, "class Calculator",
         fontsize=8.0, family="monospace", color="#345c44",
         style="italic", ha="left", va="top", zorder=3)


def sibling_edge(ax, p, q, rad=0.55):
    arr = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-", linewidth=0.7, color="#5e7a68",
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
        arrowstyle="-|>", mutation_scale=10.0,
        linewidth=0.85, color=color,
        shrinkA=12, shrinkB=12, zorder=5,
    )
    ax.add_patch(arr)


call_edge(axA, "total", "is_int", rad=0.18)
call_edge(axA, "total", "add",   rad=-0.22)
call_edge(axA, "mean",  "total", rad=0.0)

for name, attr in nodes.items():
    draw_node(axA, name, attr["xy"], attr["fill"], attr["edge"])

# ---- legend (compactly tucked inside the frame) ----
leg_y = 0.040
# call-edge entry
axA.annotate(
    "", xy=(0.355, leg_y + 0.005), xytext=(0.295, leg_y + 0.005),
    arrowprops=dict(arrowstyle="-|>", linewidth=0.85,
                    color="#404040", mutation_scale=8.5),
)
axA.text(0.370, leg_y + 0.005, r"$\mathcal{E}_{\mathrm{call}}$",
         fontsize=8.6, va="center", ha="left", color=TEXT_DK)

# sib-edge entry
axA.add_line(Line2D(
    [0.460, 0.530], [leg_y + 0.005, leg_y + 0.005],
    linewidth=0.9, linestyle=(0, (2.5, 1.8)), color="#5e7a68",
))
axA.text(0.545, leg_y + 0.005, r"$\mathcal{E}_{\mathrm{sib}}$",
         fontsize=8.6, va="center", ha="left", color=TEXT_DK)

# method swatch
axA.add_patch(Rectangle((0.640, leg_y - 0.013), 0.030, 0.034,
                        facecolor=CLSFN_FILL, edgecolor=CLSFN_EDGE,
                        linewidth=0.5))
axA.text(0.677, leg_y + 0.005, "in-class methods",
         fontsize=8.2, va="center", ha="left", color=TEXT_DK)

# top-level swatch
axA.add_patch(Rectangle((0.795, leg_y - 0.013), 0.030, 0.034,
                        facecolor=TOPFN_FILL, edgecolor=TOPFN_EDGE,
                        linewidth=0.5))
axA.text(0.832, leg_y + 0.005, "top-level functions",
         fontsize=8.2, va="center", ha="left", color=TEXT_DK)


# ============================================================================
# PANEL B — score breakdown for Calculator.total
# ============================================================================
panel_title(axB,
            r"(b) Score breakdown for $\mathtt{Calculator.total}$")

# bar geometry: leave 0.18 on the right for the "= Ĥ = 0.40" total label.
bar_x0, bar_x1 = 0.075, 0.780
bar_w = bar_x1 - bar_x0
bar_h = 0.046


def draw_stacked_bar(ax, y, values, colors, total, total_label_text,
                     total_color):
    """Draw a horizontal stacked bar with per-segment numeric labels."""
    x = bar_x0
    for v, c in zip(values, colors):
        w = (v / total) * bar_w
        ax.add_patch(Rectangle((x, y), w, bar_h, facecolor=c,
                               edgecolor="none", zorder=3))
        r, g, b = mpl.colors.to_rgb(c)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        tcol = "white" if lum < 0.62 else TEXT_DK
        if w > 0.030:
            ax.text(x + w / 2, y + bar_h / 2, f"{v:.2f}",
                    fontsize=7.4, ha="center", va="center",
                    color=tcol, family="serif", zorder=4)
        x += w
    ax.add_patch(Rectangle((bar_x0, y), bar_w, bar_h,
                           facecolor="none", edgecolor="#888",
                           linewidth=0.45, zorder=5))
    ax.text(bar_x1 + 0.014, y + bar_h / 2, total_label_text,
            fontsize=10.0, ha="left", va="center",
            weight="bold", color=total_color, family="serif")


def draw_input_arrows(ax, y_input, y_bar_top, values, total, inputs):
    """Place input strings above each segment, with a small arrow down."""
    seg_left = bar_x0
    for v, inp in zip(values, inputs):
        seg_w = (v / total) * bar_w
        cx = seg_left + seg_w / 2
        ax.text(cx, y_input, inp, fontsize=7.4,
                ha="center", va="center", color=TEXT_DK, family="serif")
        ax.annotate("",
                    xy=(cx, y_bar_top + 0.004),
                    xytext=(cx, y_input - 0.020),
                    arrowprops=dict(arrowstyle="->", linewidth=0.55,
                                    color="#888", mutation_scale=6))
        seg_left += seg_w


def draw_segment_names(ax, y, values, total, names, colors):
    seg_left = bar_x0
    for v, name, c in zip(values, names, colors):
        seg_w = (v / total) * bar_w
        cx = seg_left + seg_w / 2
        ax.text(cx, y, name, fontsize=7.2,
                ha="center", va="center", color=c, family="serif",
                weight="bold")
        seg_left += seg_w


# ---- Hhat block --------------------------------------------------------
H_TOTAL  = 0.40
H_VALS   = [0.08, 0.20, 0.12]
H_INPUTS = ["LoC = 10", "CC = 5", "D = 3"]
H_NAMES  = ["LoC", "CC", "D"]

# Formula + hyperparameters (two lines, second line in small grey).
axB.text(0.040, 0.910,
         r"$\hat{H}\;=\;w_{\ell}\,\phi(\mathrm{LoC},c_{\ell})"
         r"\,+\,w_{c}\,\phi(\mathrm{CC},c_{c})"
         r"\,+\,w_{d}\,\phi(\mathrm{D},c_{d})$, "
         r"$\;\phi(x,c){=}\min(x/c,\,2)$",
         fontsize=8.6, ha="left", va="top", color=TEXT_DK)
axB.text(0.040, 0.870-0.01,
         r"$(w_{\ell},w_{c},w_{d})\!=\!(0.4,0.4,0.2)$,"
         r"$\;(c_{\ell},c_{c},c_{d})\!=\!(50,10,5)$",
         fontsize=7.6, ha="left", va="top", color=HYPER, style="italic")

H_BAR_Y_BOT = 0.690
H_BAR_Y_TOP = H_BAR_Y_BOT + bar_h
H_INPUT_Y   = 0.795

draw_input_arrows(axB, H_INPUT_Y, H_BAR_Y_TOP, H_VALS, H_TOTAL, H_INPUTS)
draw_stacked_bar(axB, H_BAR_Y_BOT, H_VALS, H_COLS, H_TOTAL,
                 r"$\hat{H}\!=\!0.40$", H_COLS[0])
draw_segment_names(axB, H_BAR_Y_BOT - 0.040, H_VALS, H_TOTAL,
                   H_NAMES, H_COLS)

# ---- Ihat block --------------------------------------------------------
I_TOTAL  = 0.48
I_VALS   = [0.06, 0.13, 0.10, 0.05, 0.14]
I_INPUTS = ["1", "2", "type+name", r"$\checkmark$", "3 sibs"]
I_NAMES  = ["caller", "callee", "sig", "doc", "class"]

axB.text(0.040, 0.555,
         r"$\hat{I}\;=\;\alpha C_{\mathrm{caller}}"
         r"+\beta C_{\mathrm{callee}}"
         r"+\gamma C_{\mathrm{sig}}"
         r"+\delta C_{\mathrm{doc}}"
         r"+\varepsilon C_{\mathrm{class}}$",
         fontsize=8.6, ha="left", va="top", color=TEXT_DK)
axB.text(0.040, 0.515-0.01,
         r"$(\alpha,\beta,\gamma,\delta,\varepsilon)"
         r"\!=\!(0.30,0.25,0.20,0.10,0.15)$, "
         r"each $C_{\bullet}\!\in\![0,1]$",
         fontsize=7.6, ha="left", va="top", color=HYPER, style="italic")

I_BAR_Y_BOT = 0.335
I_BAR_Y_TOP = I_BAR_Y_BOT + bar_h
I_INPUT_Y   = 0.440

draw_input_arrows(axB, I_INPUT_Y, I_BAR_Y_TOP, I_VALS, I_TOTAL, I_INPUTS)
draw_stacked_bar(axB, I_BAR_Y_BOT, I_VALS, I_COLS, I_TOTAL,
                 r"$\hat{I}\!=\!0.48$", I_COLS[-1])
draw_segment_names(axB, I_BAR_Y_BOT - 0.040, I_VALS, I_TOTAL,
                   I_NAMES, I_COLS)

# ---- combined FIM at the bottom ----------------------------------------
fim_box_y0, fim_box_y1 = 0.030, 0.230
axB.add_patch(FancyBboxPatch(
    (0.040, fim_box_y0), 0.920, fim_box_y1 - fim_box_y0,
    boxstyle="round,pad=0.005,rounding_size=0.018",
    linewidth=0.5, edgecolor=SEL_FG, facecolor=SEL_BG, zorder=2,
))
# inline (not \dfrac) keeps height small so the two lines don't collide
axB.text(0.060, 0.175,
         r"$\mathrm{FIM}\;=\;\hat{H}\hat{I}\,/\,(\hat{H}+\hat{I}+\epsilon)"
         r"\;=\;0.40\times 0.48\,/\,(0.40+0.48)\;\approx\;0.22$",
         fontsize=9, ha="left", va="center", color=TEXT_DK)
axB.text(0.060, 0.085,
         r"$\geq\tau\!=\!0.20\;\;\Rightarrow\;\;"
         r"\mathtt{Calculator.total}$ is selected as a mask target.",
         fontsize=9, ha="left", va="center", color=SEL_FG,
         family="serif", weight="bold")


# ============================================================================
# PANEL C — final FIM score & selection in H-I plane
# ============================================================================
panel_title(axC,
            r"(c) Selection in $\hat{H}$–$\hat{I}$ plane")

inset = axC.inset_axes([0.155, 0.130, 0.795, 0.770],
                       transform=axC.transAxes)
inset.set_xlim(0, 1.05)
inset.set_ylim(0, 1.0)
inset.set_xlabel(r"complexity $\hat{H}$", fontsize=9.5,
                 color=TEXT_DK, labelpad=2)
inset.set_ylabel(r"inferability $\hat{I}$", fontsize=9.5,
                 color=TEXT_DK, labelpad=2)
inset.tick_params(axis="both", labelsize=8.0, length=2.5,
                  colors=GRID_FG, width=0.4)
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
              colors=[THR_EDGE], linewidths=1.0,
              linestyles="solid", zorder=2)
inset.contour(HH, II, FIM, levels=[0.10, 0.30],
              colors=[CONTOUR], linewidths=0.6,
              linestyles="dashed", zorder=2)

def label_iso(level, h_at, color="#666", weight="normal", text=None):
    if h_at <= level:
        return
    i_at = level * h_at / (h_at - level)
    if 0.04 <= i_at <= 0.95:
        lbl = text if text is not None else f"FIM={level:.2f}"
        t = inset.text(h_at, i_at, lbl, fontsize=7.2,
                       color=color, ha="center", va="center",
                       family="serif", weight=weight, zorder=4)
        t.set_path_effects([
            patheffects.Stroke(linewidth=2.5, foreground="white"),
            patheffects.Normal(),
        ])

label_iso(0.10, 0.55, color="#888")
label_iso(0.20, 0.62, color=THR_EDGE, weight="bold",
          text=fr"$\tau={tau:.2f}$")
label_iso(0.30, 0.85, color="#888")

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
        inset.text(H + 0.025, I, name, fontsize=8.0,
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

inset.annotate(
    "hard-filtered\n(LoC<10)",
    xy=(0.10, 0.35), xytext=(0.32, 0.20),
    fontsize=7.6, color=FILT_FG, ha="center", va="center",
    family="serif",
    arrowprops=dict(arrowstyle="->", linewidth=0.5,
                    color=FILT_FG, shrinkA=2, shrinkB=4),
    zorder=9,
)
inset.annotate(
    "selected\nregion",
    xy=(0.85, 0.32), xytext=(0.78, 0.10),
    fontsize=8.0, color=THR_EDGE, ha="center", va="center",
    family="serif", weight="bold",
    arrowprops=dict(arrowstyle="->", linewidth=0.55,
                    color=THR_EDGE, shrinkA=2, shrinkB=4),
    zorder=9,
)

leg_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
           markersize=8, label="selected"),
    Line2D([0], [0], marker="x", color=FILT_FG,
           markersize=8, markeredgewidth=1.4,
           linestyle="None", label="filtered"),
    Line2D([0], [0], color=THR_EDGE, linewidth=1.0,
           linestyle="solid", label=fr"$\mathrm{{FIM}}=\tau$"),
    mpatches.Patch(facecolor=THR_FILL, edgecolor=THR_EDGE,
                   linewidth=0.4, label=r"$\mathrm{FIM}\geq\tau$"),
]
leg = inset.legend(handles=leg_handles, loc="upper right",
                   fontsize=7.6, frameon=True,
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
