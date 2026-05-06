"""Method figure (Fig 2): function-aware FIM target selection.

Three panels:
  (a) Program dependency graph (code + PDG) on a small "agent-y" example
  (b) Score breakdown for the candidate function CodingAgent.act
  (c) Final FIM score & selection in the H-I plane

Style is intentionally distinct from the teaser figure (cool teal/slate/rose
palette vs. teaser's blue/orange/green/violet).
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

mpl.patheffects = patheffects

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

OUT = Path(__file__).resolve().parent

# ============================================================================
# Palette (distinct from teaser)
# ============================================================================
TOPFN_FILL = "#dbeef0"      # light cyan
TOPFN_EDGE = "#5a8b94"
CLSFN_FILL = "#dbeadc"      # light teal-green
CLSFN_EDGE = "#52866a"
CLSBG      = "#eef6f0"
SEL_FG     = "#c63452"
SEL_BG     = "#f9d6dd"
FILT_FG    = "#9aa0a6"
FILT_BG    = "#ebebeb"

# H bar segments  (slate-blue)
H_COLS = ["#2c5d8a", "#5e94be", "#a9c4e1"]
# I bar segments  (gold -> rose)
I_COLS = ["#d99046", "#e0aa6f", "#d3bf94", "#cb8c95", "#9b6585"]

CONTOUR  = "#bdbdbd"
THR_FILL = "#fdeaee"
GRID_FG  = "#777777"
TEXT_DK  = "#202020"
TEXT_MD  = "#505050"

# ============================================================================
# Figure layout
# ============================================================================
FIG_W, FIG_H = 13.5, 5.1

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.55, 1.10, 1.20],
    wspace=0.10,
    left=0.005, right=0.995, bottom=0.02, top=0.98,
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
for ax in (axA, axB, axC):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")


def panel_border(ax, pad=0.005):
    rect = FancyBboxPatch(
        (0 + pad, 0 + pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.55, edgecolor="#888", facecolor="white", zorder=0,
    )
    ax.add_patch(rect)


def panel_title(ax, text, x=0.02, y=0.965):
    ax.text(x, y, text, fontsize=10.5, weight="bold",
            ha="left", va="top", color=TEXT_DK, family="serif")


for ax in (axA, axB, axC):
    panel_border(ax)

# ============================================================================
# PANEL A — code + PDG
# ============================================================================
panel_title(axA, "(a) Program dependency graph")

# ---- code listing (left half) ----
CODE_LINES = [
    ("# example.py", "comment"),
    ("def parse_action(t: str) -> Action:", "code"),
    ('    """Extract tool call."""', "doc"),
    ("    m = ACTION_RE.search(t)", "code"),
    ("    if not m: return Action.noop()", "code"),
    ("    return Action(m['t'], m['a'])", "code"),
    ("", "blank"),
    ("def format_prompt(hist):", "code"),
    ("    return '\\n'.join(map(str, hist))", "code"),
    ("", "blank"),
    ("class CodingAgent:", "code"),
    ("    def __init__(self, m): ...", "code"),
    ("    def observe(self, o):  ...", "code"),
    ("    def act(self, obs: str) -> Action:", "code"),
    ('        """Plan next action."""', "doc"),
    ("        p = format_prompt(", "code"),
    ("            self.history + [obs])", "code"),
    ("        for _ in range(self.retries):", "code"),
    ("            t = self.model.generate(p)", "code"),
    ("            a = parse_action(t)", "code"),
    ("            if a.is_valid(): return a", "code"),
    ("        return Action.noop()", "code"),
    ("    def run(self, task) -> Traj:", "code"),
    ("        ...  # orchestrate observe/act", "code"),
]

# code box background
code_x0, code_x1 = 0.030, 0.380
code_y0, code_y1 = 0.085, 0.905
axA.add_patch(FancyBboxPatch(
    (code_x0, code_y0), code_x1 - code_x0, code_y1 - code_y0,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    linewidth=0.45, edgecolor="#bbb", facecolor="#fafafa", zorder=1,
))

n_lines = len(CODE_LINES)
line_h = (code_y1 - code_y0 - 0.025) / n_lines
y_top = code_y1 - 0.012
for i, (line, kind) in enumerate(CODE_LINES):
    y = y_top - (i + 0.5) * line_h
    if kind == "comment":
        col = "#7a8896"
        style = "italic"
        weight = "normal"
    elif kind == "doc":
        col = "#7a8896"
        style = "italic"
        weight = "normal"
    elif kind == "blank":
        continue
    else:
        col = TEXT_DK
        style = "normal"
        weight = "normal"
    # bold the def/class keyword line names for visibility
    txt = line
    axA.text(code_x0 + 0.012, y, txt, fontsize=6.6,
             family="monospace", color=col, style=style, weight=weight,
             ha="left", va="center", zorder=2)

# ---- arrow "AST parse →" between code and PDG ----
axA.annotate(
    "", xy=(0.452, 0.50), xytext=(0.395, 0.50),
    arrowprops=dict(arrowstyle="->", linewidth=0.9, color="#666"),
)
axA.text(0.4235, 0.555, "AST parse", fontsize=7.5,
         color=TEXT_MD, ha="center", va="bottom", family="serif")

# ---- PDG nodes & edges ----
# coordinates (in axes fraction)
nodes = {
    # top-level functions (left column inside the PDG area)
    "parse_action":  dict(xy=(0.560, 0.78), fill=TOPFN_FILL, edge=TOPFN_EDGE,
                          kind="top"),
    "format_prompt": dict(xy=(0.560, 0.62), fill=TOPFN_FILL, edge=TOPFN_EDGE,
                          kind="top"),
    # CodingAgent methods (right column, will be wrapped in a class box)
    "__init__":      dict(xy=(0.860, 0.83), fill=CLSFN_FILL, edge=CLSFN_EDGE,
                          kind="cls"),
    "observe":       dict(xy=(0.860, 0.66), fill=CLSFN_FILL, edge=CLSFN_EDGE,
                          kind="cls"),
    "act":           dict(xy=(0.860, 0.49), fill=CLSFN_FILL, edge=CLSFN_EDGE,
                          kind="cls"),
    "run":           dict(xy=(0.860, 0.32), fill=CLSFN_FILL, edge=CLSFN_EDGE,
                          kind="cls"),
}

NODE_W, NODE_H = 0.165, 0.085


def draw_node(ax, label, xy, fill, edge, kind, fontweight="normal",
              extra_outline=False, text_label=None):
    cx, cy = xy
    x0 = cx - NODE_W / 2
    y0 = cy - NODE_H / 2
    box = FancyBboxPatch(
        (x0, y0), NODE_W, NODE_H,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0.55, edgecolor=edge, facecolor=fill, zorder=4,
    )
    ax.add_patch(box)
    if extra_outline:
        outline = FancyBboxPatch(
            (x0 - 0.008, y0 - 0.008), NODE_W + 0.016, NODE_H + 0.016,
            boxstyle="round,pad=0.0,rounding_size=0.022",
            linewidth=0.9, edgecolor=SEL_FG, facecolor="none", zorder=5,
        )
        ax.add_patch(outline)
    txt = text_label if text_label is not None else label
    ax.text(cx, cy, txt, fontsize=7.0, family="monospace",
            ha="center", va="center", color=TEXT_DK,
            weight=fontweight, zorder=6)


# class container box (drawn first, behind nodes)
cls_pad_x, cls_pad_y = 0.018, 0.025
cls_x0 = nodes["__init__"]["xy"][0] - NODE_W / 2 - cls_pad_x
cls_x1 = nodes["__init__"]["xy"][0] + NODE_W / 2 + cls_pad_x
cls_y1 = nodes["__init__"]["xy"][1] + NODE_H / 2 + cls_pad_y + 0.020
cls_y0 = nodes["run"]["xy"][1] - NODE_H / 2 - cls_pad_y
axA.add_patch(FancyBboxPatch(
    (cls_x0, cls_y0), cls_x1 - cls_x0, cls_y1 - cls_y0,
    boxstyle="round,pad=0.0,rounding_size=0.022",
    linewidth=0.45, edgecolor=CLSFN_EDGE + "aa", facecolor=CLSBG, zorder=2,
))
axA.text(cls_x0 + 0.010, cls_y1 - 0.020, "class CodingAgent",
         fontsize=6.6, family="monospace", color="#345c44",
         style="italic", ha="left", va="top", zorder=3)


# Sibling edges (undirected, dashed) — drawn first, behind call edges
def sibling_edge(ax, p, q, rad=0.30):
    arr = FancyArrowPatch(
        p, q,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=0.55, color="#5e7a68",
        linestyle=(0, (2.0, 1.6)),
        zorder=3,
    )
    ax.add_patch(arr)


# pick four representative sibling edges (right-side curves)
sib_x = nodes["__init__"]["xy"][0] + NODE_W / 2
for (a, b, rad) in [
    ("__init__", "observe", 0.55),
    ("observe",  "act",     0.55),
    ("act",      "run",     0.55),
    ("__init__", "run",     0.95),
]:
    pa = (sib_x, nodes[a]["xy"][1])
    pb = (sib_x, nodes[b]["xy"][1])
    sibling_edge(axA, pa, pb, rad=rad)


# Call edges (directed)
def call_edge(ax, src, dst, rad=0.0, color="#404040"):
    p = nodes[src]["xy"]
    q = nodes[dst]["xy"]
    arr = FancyArrowPatch(
        p, q,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=8.5,
        linewidth=0.65, color=color,
        shrinkA=10, shrinkB=10,
        zorder=5,
    )
    ax.add_patch(arr)


# act -> parse_action, act -> format_prompt
call_edge(axA, "act", "parse_action", rad=-0.18)
call_edge(axA, "act", "format_prompt", rad=0.10)
# run -> act, run -> observe (within class — short edges)
call_edge(axA, "run", "act", rad=0.0)
call_edge(axA, "run", "observe", rad=-0.30)


# Now draw the nodes
for name, attr in nodes.items():
    draw_node(axA, name, attr["xy"], attr["fill"], attr["edge"], attr["kind"])

# Legend (bottom of panel A)
leg_y = 0.045
# call edge swatch
axA.annotate(
    "", xy=(0.330, leg_y + 0.005), xytext=(0.260, leg_y + 0.005),
    arrowprops=dict(arrowstyle="-|>", linewidth=0.65,
                    color="#404040", mutation_scale=7),
)
axA.text(0.345, leg_y + 0.005, r"$\mathcal{E}_{\mathrm{call}}$",
         fontsize=8.3, va="center", ha="left", color=TEXT_DK)
axA.add_line(Line2D(
    [0.480, 0.555], [leg_y + 0.005, leg_y + 0.005],
    linewidth=0.7, linestyle=(0, (2.5, 1.8)), color="#5e7a68",
))
axA.text(0.570, leg_y + 0.005, r"$\mathcal{E}_{\mathrm{sib}}$",
         fontsize=8.3, va="center", ha="left", color=TEXT_DK)
# class swatch
axA.add_patch(Rectangle((0.700, leg_y - 0.010), 0.030, 0.030,
                        facecolor=CLSFN_FILL, edgecolor=CLSFN_EDGE,
                        linewidth=0.5))
axA.text(0.738, leg_y + 0.005, "class methods",
         fontsize=7.6, va="center", ha="left", color=TEXT_DK)
axA.add_patch(Rectangle((0.880, leg_y - 0.010), 0.030, 0.030,
                        facecolor=TOPFN_FILL, edgecolor=TOPFN_EDGE,
                        linewidth=0.5))
axA.text(0.916, leg_y + 0.005, "top-level fns",
         fontsize=7.6, va="center", ha="left", color=TEXT_DK)


# ============================================================================
# PANEL B — H / I score breakdown for CodingAgent.act
# ============================================================================
panel_title(axB, r"(b) $\hat{H}$ & $\hat{I}$ for $\mathtt{CodingAgent.act}$")

bar_x0, bar_x1 = 0.085, 0.935
bar_w = bar_x1 - bar_x0
bar_h = 0.058


def stacked_bar(ax, y, segs, colors, total, total_color,
                inside_text_color=("#fff", TEXT_DK)):
    """Draw a horizontal stacked bar normalized to bar_w.

    segs: list of (value, label) — `label` may be None to skip in-bar label.
    """
    x = bar_x0
    for (v, lbl), c in zip(segs, colors):
        w = (v / total) * bar_w
        ax.add_patch(Rectangle((x, y), w, bar_h,
                               facecolor=c, edgecolor="none", zorder=3))
        if lbl is not None and w > 0.025:
            # choose text color by background luminance approx
            r, g, b = mpl.colors.to_rgb(c)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tcol = inside_text_color[0] if lum < 0.62 else inside_text_color[1]
            ax.text(x + w / 2, y + bar_h / 2, lbl,
                    fontsize=6.7, ha="center", va="center",
                    color=tcol, family="serif", zorder=4)
        x += w
    # outline
    ax.add_patch(Rectangle((bar_x0, y), bar_w, bar_h,
                           facecolor="none", edgecolor="#888",
                           linewidth=0.45, zorder=5))
    # total annotation to the right
    ax.text(bar_x1 + 0.012, y + bar_h / 2, total_color, fontsize=8.5,
            ha="left", va="center", weight="bold", color=H_COLS[0],
            family="serif")


# --- Hhat block ---
axB.text(0.030, 0.875,
         r"$\hat{H} = w_{\ell}\,\phi(\mathrm{LoC})"
         r" + w_{c}\,\phi(\mathrm{CC})"
         r" + w_{d}\,\phi(\mathrm{D})$",
         fontsize=8.4, ha="left", va="top", color=TEXT_DK)

# Hhat = 0.82, segments 0.32 / 0.30 / 0.20
H_TOTAL = 0.82
h_segs = [(0.32, "0.32"), (0.30, "0.30"), (0.20, "0.20")]
h_y = 0.770
x_cur = bar_x0
for (v, lbl), c in zip(h_segs, H_COLS):
    w = (v / H_TOTAL) * bar_w
    axB.add_patch(Rectangle((x_cur, h_y), w, bar_h,
                            facecolor=c, edgecolor="none", zorder=3))
    r, g, b = mpl.colors.to_rgb(c)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    tcol = "white" if lum < 0.62 else TEXT_DK
    axB.text(x_cur + w / 2, h_y + bar_h / 2, lbl,
             fontsize=6.9, ha="center", va="center", color=tcol,
             family="serif", zorder=4)
    x_cur += w
axB.add_patch(Rectangle((bar_x0, h_y), bar_w, bar_h,
                        facecolor="none", edgecolor="#888",
                        linewidth=0.45, zorder=5))
axB.text(bar_x1 + 0.012, h_y + bar_h / 2, r"$=0.82$",
         fontsize=9.0, ha="left", va="center",
         weight="bold", color=H_COLS[0], family="serif")

# Hhat segment legend
seg_y = h_y - 0.080
seg_x = bar_x0
swatch_w, swatch_h = 0.026, 0.024
for col, lbl in zip(H_COLS, ["LoC=18", "CC=4", "D=3"]):
    axB.add_patch(Rectangle((seg_x, seg_y), swatch_w, swatch_h,
                            facecolor=col, edgecolor="none", zorder=3))
    axB.text(seg_x + swatch_w + 0.008, seg_y + swatch_h / 2, lbl,
             fontsize=7.4, ha="left", va="center", color=TEXT_DK,
             family="serif")
    seg_x += 0.270

# --- Ihat block ---
axB.text(0.030, 0.555,
         r"$\hat{I} = \alpha C_{\mathrm{caller}}"
         r" + \beta C_{\mathrm{callee}}"
         r" + \gamma C_{\mathrm{sig}}"
         r" + \delta C_{\mathrm{doc}}"
         r" + \varepsilon C_{\mathrm{class}}$",
         fontsize=8.0, ha="left", va="top", color=TEXT_DK)

I_TOTAL = 0.78
i_segs = [(0.18, ".18"), (0.20, ".20"), (0.16, ".16"), (0.05, ".05"),
          (0.19, ".19")]
i_y = 0.450
x_cur = bar_x0
for (v, lbl), c in zip(i_segs, I_COLS):
    w = (v / I_TOTAL) * bar_w
    axB.add_patch(Rectangle((x_cur, i_y), w, bar_h,
                            facecolor=c, edgecolor="none", zorder=3))
    r, g, b = mpl.colors.to_rgb(c)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    tcol = "white" if lum < 0.62 else TEXT_DK
    if w > 0.020:
        axB.text(x_cur + w / 2, i_y + bar_h / 2, lbl,
                 fontsize=6.7, ha="center", va="center", color=tcol,
                 family="serif", zorder=4)
    x_cur += w
axB.add_patch(Rectangle((bar_x0, i_y), bar_w, bar_h,
                        facecolor="none", edgecolor="#888",
                        linewidth=0.45, zorder=5))
axB.text(bar_x1 + 0.012, i_y + bar_h / 2, r"$=0.78$",
         fontsize=9.0, ha="left", va="center",
         weight="bold", color=I_COLS[-1], family="serif")

# Ihat segment legend (2 rows)
seg_y2 = i_y - 0.090
labels_I = ["caller", "callee", "sig", "doc", "class"]
positions = [(bar_x0,           seg_y2 + 0.030),
             (bar_x0 + 0.270,   seg_y2 + 0.030),
             (bar_x0 + 0.540,   seg_y2 + 0.030),
             (bar_x0,           seg_y2 - 0.005),
             (bar_x0 + 0.270,   seg_y2 - 0.005)]
for col, lbl, (sx, sy) in zip(I_COLS, labels_I, positions):
    axB.add_patch(Rectangle((sx, sy), swatch_w, swatch_h,
                            facecolor=col, edgecolor="none", zorder=3))
    axB.text(sx + swatch_w + 0.008, sy + swatch_h / 2, lbl,
             fontsize=7.4, ha="left", va="center", color=TEXT_DK,
             family="serif")

# Evidence box
ev_text = (
    r"$\mathbf{evidence:}$ "
    r"$\mathtt{run}$ calls $\mathtt{act}$ (caller); "
    r"$\mathtt{act}$ calls $\mathtt{format\_prompt}$, "
    r"$\mathtt{parse\_action}$ (callees);"
    "\n"
    r"typed signature $+$ docstring; "
    r"3 in-class siblings via $\mathcal{E}_{\mathrm{sib}}$."
)
axB.add_patch(FancyBboxPatch(
    (0.040, 0.045), 0.920, 0.225,
    boxstyle="round,pad=0.005,rounding_size=0.020",
    linewidth=0.4, edgecolor="#bbb", facecolor="#f6f6f6", zorder=2,
))
axB.text(0.060, 0.158, ev_text, fontsize=7.3, ha="left", va="center",
         color=TEXT_DK, family="serif")

# ============================================================================
# PANEL C — final FIM score & selection in H-I plane
# ============================================================================
panel_title(axC,
            r"(c) FIM$(v) = \dfrac{\hat{H}\,\hat{I}}{\hat{H}+\hat{I}+\epsilon}\,\rho(\Delta)$")

# inset axis to host the actual plot (with axes/ticks)
inset = axC.inset_axes([0.135, 0.115, 0.825, 0.75], transform=axC.transAxes)
inset.set_xlim(0, 1.4)
inset.set_ylim(0, 1.05)
inset.set_xlabel(r"complexity $\hat{H}$", fontsize=9, color=TEXT_DK,
                 labelpad=2)
inset.set_ylabel(r"inferability $\hat{I}$", fontsize=9, color=TEXT_DK,
                 labelpad=2)
inset.tick_params(axis="both", labelsize=7.5, length=2.5,
                  colors=GRID_FG, width=0.4)
for spine in inset.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#999")

# threshold-shaded region: FIM(H,I) >= tau, where FIM≈H*I/(H+I)
tau = 0.20
H_grid = np.linspace(0, 1.4, 400)
I_grid = np.linspace(0, 1.05, 400)
HH, II = np.meshgrid(H_grid, I_grid)
FIM = (HH * II) / (HH + II + 1e-9)
inset.contourf(HH, II, FIM, levels=[tau, 10.0],
               colors=[THR_FILL], alpha=1.0, zorder=1)

# iso-FIM contours
levels = [0.10, 0.20, 0.30, 0.40]
cs = inset.contour(HH, II, FIM, levels=levels,
                   colors=CONTOUR, linewidths=0.55,
                   linestyles="dashed",
                   zorder=2)
# label contours along the curves (interior, not at the right edge)
contour_label_pos = {
    0.10: 0.30,   # H value to anchor the label on this iso-curve
    0.20: 0.45,
    0.30: 0.60,
    0.40: 0.95,
}
for lev, H_star in contour_label_pos.items():
    if H_star > lev + 0.005:
        I_star = lev * H_star / (H_star - lev)
        if 0.05 <= I_star <= 1.0:
            # white halo behind the label so it sits on top of the curve cleanly
            txt = inset.text(H_star, I_star, f"{lev:.2f}",
                             fontsize=6.5, color="#666",
                             ha="center", va="center", family="serif",
                             zorder=3)
            txt.set_path_effects([
                mpl.patheffects.Stroke(linewidth=2.2, foreground="white"),
                mpl.patheffects.Normal(),
            ])

# candidate points
candidates = [
    # name,           H,    I,    status
    ("observe",      0.18, 0.62, "filt"),
    ("format_prompt",0.22, 0.42, "filt"),
    ("parse_action", 0.45, 0.55, "cand"),
    ("run",          1.10, 0.55, "cand"),
    ("act",          0.82, 0.78, "sel"),
]

for name, H, I, status in candidates:
    if status == "filt":
        inset.plot(H, I, marker="x", markersize=7.0,
                   markeredgewidth=1.2, color=FILT_FG, zorder=6)
        inset.text(H + 0.04, I + 0.022, name, fontsize=7.2,
                   color=FILT_FG, ha="left", va="bottom",
                   family="serif")
    elif status == "cand":
        inset.plot(H, I, marker="o", markersize=5.0,
                   markerfacecolor="#404040", markeredgecolor="#202020",
                   markeredgewidth=0.4, zorder=6)
        # label position tuned per point
        if name == "run":
            inset.text(H - 0.04, I + 0.028, name, fontsize=7.4,
                       color=TEXT_DK, ha="right", va="bottom",
                       family="serif")
        else:
            inset.text(H + 0.04, I + 0.028, name, fontsize=7.4,
                       color=TEXT_DK, ha="left", va="bottom",
                       family="serif")
    elif status == "sel":
        inset.plot(H, I, marker="o", markersize=8.0,
                   markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
                   markeredgewidth=0.6, zorder=8)
        # halo
        inset.plot(H, I, marker="o", markersize=14.0,
                   markerfacecolor="none", markeredgecolor=SEL_FG,
                   markeredgewidth=0.8, zorder=7)
        inset.text(H + 0.06, I + 0.04, name + r" $\checkmark$",
                   fontsize=8.2, color=SEL_FG, ha="left", va="bottom",
                   family="serif", weight="bold")

# legend (top-left of inset)
leg_handles = [
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor=SEL_FG, markeredgecolor=SEL_FG,
           markersize=7, label="selected"),
    Line2D([0], [0], marker="o", color="none",
           markerfacecolor="#404040", markeredgecolor="#202020",
           markersize=5, label="candidate"),
    Line2D([0], [0], marker="x", color=FILT_FG,
           markersize=7, markeredgewidth=1.2,
           linestyle="None", label="hard-filtered"),
    mpatches.Patch(facecolor=THR_FILL, edgecolor="#d8b8c0",
                   linewidth=0.4, label=r"FIM $\geq \tau$"),
]
leg = inset.legend(handles=leg_handles, loc="upper right",
                   fontsize=7.0, frameon=True,
                   framealpha=0.95, edgecolor="#bbb",
                   handlelength=1.2, handletextpad=0.5,
                   borderpad=0.4, labelspacing=0.32)
leg.get_frame().set_linewidth(0.4)

# tau annotation pointing to threshold curve
# locate point on curve at H=0.50: I = tau*H/(H-tau) = 0.2*0.50/0.30 = 0.333
arrow_kwargs = dict(arrowstyle="->", color="#a85060", linewidth=0.55)
inset.annotate(r"threshold $\tau$",
               xy=(0.50, 0.333), xytext=(0.16, 0.18),
               fontsize=7.6, color="#a85060", family="serif",
               arrowprops=arrow_kwargs, zorder=9)

# ============================================================================
# Save
# ============================================================================
pdf_path = OUT / "pdg_score.pdf"
png_path = OUT / "pdg_score.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")
