"""Teaser v12: clean graph-based left panel, color-coded 4-stage analogy.

Design:
- Left mini-graph: caller (blue context) → callee (green return), with
  orange call-arrow and green return-arrow. Caller is split visually into
  'pre-call' and 'post-return' regions to convey continuation (purple).
- Right mini-graph: agent (blue) ↔ tool/env (green), same arrow scheme.
- 4 colors map to the 4 stages: blue=context, orange=call, green=return, purple=continuation.
- Bottom legend explains the color-to-stage mapping ONCE for both graphs.
- No numbered tags.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent

C_CTX  = "#cfe2f3"
C_ACT  = "#fce5cd"
C_RET  = "#d9ead3"
C_CONT = "#e6d5f0"
EDGE   = {C_CTX: "#3b6fb8", C_ACT: "#cc7a00",
          C_RET: "#3a8c4f", C_CONT: "#7a4a8e"}
C_BG_FIM = "#f7f5fb"
C_RATIONALE = "#fff5d6"
C_INPUT_TINT = "#eef2f7"
C_TARGET_TINT = "#fdf3e9"


fig = plt.figure(figsize=(13.0, 3.5))

gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.40, 1.65, 1.00],
    wspace=0.18,
    left=0.012, right=0.985, bottom=0.08, top=0.80,
)

# Header
fig.text(0.5, 0.955,
         r"Function call site  $\sim$  agent step  "
         r"$\mathit{(structurally\ similar)}$  "
         r"$\Rightarrow$  use as self-supervised FIM mid-training prior",
         fontsize=11.0, ha="center", va="center", fontweight="bold",
         color="#1a1a1a")

# =============================================================
# LEFT PANEL — graph-based analogy
# =============================================================
axL = fig.add_subplot(gs[0, 0])
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.axis("off")

axL.text(0.255, 1.01, "Function call site", fontsize=9.5,
         fontweight="bold", ha="center", va="bottom")
axL.text(0.745, 1.01, "Coding-agent step", fontsize=9.5,
         fontweight="bold", ha="center", va="bottom")

# Two background panels
LBG = mpatches.FancyBboxPatch(
    (0.005, 0.30), 0.485, 0.62,
    boxstyle="round,pad=0.002,rounding_size=0.010",
    linewidth=0.5, facecolor="#f7faf7", edgecolor="#bbb",
    zorder=1,
)
axL.add_patch(LBG)
RBG = mpatches.FancyBboxPatch(
    (0.510, 0.30), 0.485, 0.62,
    boxstyle="round,pad=0.002,rounding_size=0.010",
    linewidth=0.5, facecolor="#f7faf7", edgecolor="#bbb",
    zorder=1,
)
axL.add_patch(RBG)


def split_caller_node(ax, cx, cy, w, h, text, fs=7.2, monospace=True,
                       top_label="pre-call", bot_label="post-return"):
    """2-band node: top half = context (blue), bottom half = continuation (purple).

    The function name (`text`) is rendered INSIDE each half on the LEFT side,
    so the seam between halves does not bisect the label. Each half also has
    a small role label (top_label / bot_label) on the right side.
    """
    # Top half: context (pre-call / history)
    top = mpatches.FancyBboxPatch(
        (cx - w/2, cy), w, h/2,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        linewidth=0.8, facecolor=C_CTX, edgecolor=EDGE[C_CTX], zorder=3,
    )
    ax.add_patch(top)
    # Bottom half: continuation (post-return / next step)
    bot = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h/2,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        linewidth=0.8, facecolor=C_CONT, edgecolor=EDGE[C_CONT], zorder=3,
    )
    ax.add_patch(bot)
    fam = "monospace" if monospace else "serif"
    # Render the name label inside the TOP half, left-aligned, vertically
    # centered within the upper sub-rectangle (so it does not span the seam).
    ax.text(cx - w/2 + 0.012, cy + h/4, text,
            fontsize=fs, ha="left", va="center",
            family=fam, zorder=4)
    # Top-half role tag on the right
    ax.text(cx + w/2 - 0.008, cy + h/4, top_label,
            fontsize=5.5, ha="right", va="center",
            color=EDGE[C_CTX], style="italic", zorder=4)
    # Bottom-half role tag on the right
    ax.text(cx + w/2 - 0.008, cy - h/4, bot_label,
            fontsize=5.5, ha="right", va="center",
            color=EDGE[C_CONT], style="italic", zorder=4)


def simple_node(ax, cx, cy, w, h, color, text, fs=7.0, monospace=True):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        linewidth=0.8, facecolor=color, edgecolor=EDGE[color], zorder=3,
    )
    ax.add_patch(rect)
    fam = "monospace" if monospace else "serif"
    ax.text(cx, cy, text, fontsize=fs, ha="center", va="center",
            family=fam, zorder=4)


def labeled_arrow(ax, x1, y1, x2, y2, label, color, fs=6.5,
                  label_side="right"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color=color, zorder=2,
    )
    ax.add_patch(arr)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if label_side == "right":
        ax.text(mx + 0.014, my, label, fontsize=fs, ha="left", va="center",
                color=color, style="italic", fontweight="bold", zorder=2)
    else:
        ax.text(mx - 0.014, my, label, fontsize=fs, ha="right", va="center",
                color=color, style="italic", fontweight="bold", zorder=2)


# ---- LEFT GRAPH: function call site ----
caller_cx, caller_cy = 0.250, 0.83
caller_w, caller_h = 0.34, 0.16
split_caller_node(axL, caller_cx, caller_cy, caller_w, caller_h,
                  "caller A( )", fs=7.0)

callee_cx, callee_cy = 0.250, 0.39
callee_w, callee_h = 0.34, 0.13
simple_node(axL, callee_cx, callee_cy, callee_w, callee_h, C_RET,
            "callee B( )", fs=7.0)

# Down arrow from caller bottom-left -> callee top-left: the call
labeled_arrow(axL,
              caller_cx - caller_w/4, caller_cy - caller_h/2,
              callee_cx - callee_w/4, callee_cy + callee_h/2,
              "call(args)", EDGE[C_ACT], label_side="left")
# Up arrow from callee top-right -> caller bottom-right: return
labeled_arrow(axL,
              callee_cx + callee_w/4, callee_cy + callee_h/2,
              caller_cx + caller_w/4, caller_cy - caller_h/2,
              "return val", EDGE[C_RET], label_side="right")


# ---- RIGHT GRAPH: agent step ----
agent_cx, agent_cy = 0.745, 0.83
agent_w, agent_h = 0.34, 0.16
split_caller_node(axL, agent_cx, agent_cy, agent_w, agent_h,
                  "agent ($h_t$)", fs=7.0, monospace=False,
                  top_label="history", bot_label="next step")

tool_cx, tool_cy = 0.745, 0.39
tool_w, tool_h = 0.34, 0.13
simple_node(axL, tool_cx, tool_cy, tool_w, tool_h, C_RET,
            "tool / environment", fs=6.8, monospace=False)

labeled_arrow(axL,
              agent_cx - agent_w/4, agent_cy - agent_h/2,
              tool_cx - tool_w/4, tool_cy + tool_h/2,
              "action $a_t$", EDGE[C_ACT], label_side="left")
labeled_arrow(axL,
              tool_cx + tool_w/4, tool_cy + tool_h/2,
              agent_cx + agent_w/4, agent_cy - agent_h/2,
              "obs. $o_{t+1}$", EDGE[C_RET], label_side="right")


# ---- Bottom legend: the 4-color → 4-stage mapping (single legend for both graphs) ----
LEGEND_Y = 0.13
legend_items = [
    (C_CTX, "context"),
    (C_ACT, "call / action"),
    (C_RET, "return / observation"),
    (C_CONT, "continuation"),
]
xs = np.linspace(0.085, 0.915, len(legend_items))
for cx, (col, lab) in zip(xs, legend_items):
    sw, sh = 0.020, 0.030
    rect = mpatches.FancyBboxPatch(
        (cx - 0.080, LEGEND_Y - sh/2), sw, sh,
        boxstyle="round,pad=0.0,rounding_size=0.004",
        linewidth=0.5, facecolor=col, edgecolor=EDGE[col],
    )
    axL.add_patch(rect)
    axL.text(cx - 0.080 + sw + 0.006, LEGEND_Y, lab,
             fontsize=6.4, ha="left", va="center",
             color=EDGE[col], style="italic")

# Bottom italic caption
axL.text(0.50, 0.035,
         "same 4-stage decomposition: context $\\to$ call $\\to$ return $\\to$ continuation",
         fontsize=7.2, ha="center", va="center",
         color="#444444", style="italic")


# =============================================================
# MIDDLE PANEL — function-level FIM as SFT (input → target)
# =============================================================
axM = fig.add_subplot(gs[0, 1])
# Shift middle panel left by ~5 pixels (0.005 of figure width at 1300px wide PNG)
# Increase MIDDLE_X_SHIFT to push further left; decrease to shift right.
MIDDLE_X_SHIFT = 0.015
_pos = axM.get_position()
axM.set_position([_pos.x0 - MIDDLE_X_SHIFT, _pos.y0,
                  _pos.width, _pos.height])
axM.set_xlim(0, 1); axM.set_ylim(0, 1)
axM.axis("off")

axM.text(0.5, 1.01, "Function-level FIM mid-training (SFT)",
         fontsize=9.5, fontweight="bold", ha="center", va="bottom")

PDG_LEFT, PDG_RIGHT = 0.005, 0.27
FIM_LEFT, FIM_RIGHT = 0.30, 0.99

pdg_bg = mpatches.FancyBboxPatch(
    (PDG_LEFT, 0.04), PDG_RIGHT - PDG_LEFT, 0.88,
    boxstyle="round,pad=0.002,rounding_size=0.010",
    linewidth=0.5, facecolor="#f4f4f4", edgecolor="#bbb",
    zorder=1,
)
axM.add_patch(pdg_bg)
axM.text((PDG_LEFT + PDG_RIGHT) / 2, 0.86,
         "dependency graph", fontsize=7.2,
         ha="center", va="top", color="#666",
         style="italic")

NODE_R = 0.038
node_A = (0.135, 0.69)
node_B = (0.135, 0.51)
node_C = (0.135, 0.33)

for (x1, y1), (x2, y2) in [(node_A, node_B), (node_B, node_C)]:
    arr = FancyArrowPatch(
        (x1, y1 - NODE_R), (x2, y2 + NODE_R),
        arrowstyle="-|>", mutation_scale=8,
        linewidth=0.8, color="#888", zorder=2,
    )
    axM.add_patch(arr)
    axM.text(x1 + 0.028, (y1 + y2) / 2, "calls",
             fontsize=6.0, ha="left", va="center",
             color="#888", style="italic", zorder=2)

circ_A = mpatches.Circle(node_A, NODE_R,
                          facecolor=C_CTX, edgecolor=EDGE[C_CTX],
                          linewidth=0.9, zorder=3)
axM.add_patch(circ_A)
axM.text(node_A[0], node_A[1], "A", fontsize=9.5, fontweight="bold",
         ha="center", va="center", family="monospace", zorder=4)

circ_B_halo = mpatches.Circle(node_B, NODE_R + 0.012,
                               facecolor="none", edgecolor=EDGE[C_ACT],
                               linewidth=1.4, linestyle=(0, (3, 2)),
                               zorder=3)
axM.add_patch(circ_B_halo)
circ_B = mpatches.Circle(node_B, NODE_R,
                          facecolor=C_ACT, edgecolor=EDGE[C_ACT],
                          linewidth=0.9, zorder=4)
axM.add_patch(circ_B)
axM.text(node_B[0], node_B[1], "B", fontsize=9.5, fontweight="bold",
         ha="center", va="center", family="monospace", zorder=5)
axM.text(node_B[0] + NODE_R + 0.030, node_B[1], "mask",
         fontsize=7.0, ha="left", va="center",
         color=EDGE[C_ACT], style="italic", fontweight="bold", zorder=5)

circ_C = mpatches.Circle(node_C, NODE_R,
                          facecolor=C_CONT, edgecolor=EDGE[C_CONT],
                          linewidth=0.9, zorder=3)
axM.add_patch(circ_C)
axM.text(node_C[0], node_C[1], "C", fontsize=9.5, fontweight="bold",
         ha="center", va="center", family="monospace", zorder=4)

axM.text((PDG_LEFT + PDG_RIGHT) / 2, 0.115,
         "B selected via\nPDG + $\\hat{H}$ + $\\hat{I}$",
         fontsize=6.6, ha="center", va="center",
         color="#555", style="italic")

arr_bridge = FancyArrowPatch(
    (PDG_RIGHT + 0.005, 0.51), (FIM_LEFT - 0.005, 0.51),
    arrowstyle="-|>", mutation_scale=10,
    linewidth=0.9, color="#666", zorder=2,
)
axM.add_patch(arr_bridge)

bg = mpatches.FancyBboxPatch(
    (FIM_LEFT, 0.04), FIM_RIGHT - FIM_LEFT, 0.88,
    boxstyle="round,pad=0.002,rounding_size=0.010",
    linewidth=0.5, facecolor=C_BG_FIM, edgecolor="#bbb5cc",
    zorder=1,
)
axM.add_patch(bg)

INPUT_TOP = 0.88
INPUT_BOTTOM = 0.50
input_group = mpatches.FancyBboxPatch(
    (FIM_LEFT + 0.006, INPUT_BOTTOM),
    (FIM_RIGHT - FIM_LEFT) - 0.012,
    INPUT_TOP - INPUT_BOTTOM,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    linewidth=0.0, facecolor=C_INPUT_TINT, zorder=1.5,
)
axM.add_patch(input_group)
axM.text(FIM_LEFT + 0.014, INPUT_TOP + 0.025, "input prompt",
         fontsize=6.8, ha="left", va="top",
         color="#3b6fb8", fontweight="bold", style="italic", zorder=10)

TARGET_TOP = 0.485
TARGET_BOTTOM = 0.145
target_group = mpatches.FancyBboxPatch(
    (FIM_LEFT + 0.006, TARGET_BOTTOM),
    (FIM_RIGHT - FIM_LEFT) - 0.012,
    TARGET_TOP - TARGET_BOTTOM,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    linewidth=0.0, facecolor=C_TARGET_TINT, zorder=1.5,
)
axM.add_patch(target_group)
axM.text(FIM_LEFT + 0.014, TARGET_TOP - 0.015, "SFT target",
         fontsize=6.8, ha="left", va="top",
         color="#cc7a00", fontweight="bold", style="italic", zorder=10)


def add_fim_row(y, h, sentinel, body_text, body_color, sentinel_color,
                body_fs=6.2, kind="code", body_right_inset=0.055):
    PILL_W = 0.130
    PILL_X = FIM_LEFT + 0.020
    pill = mpatches.FancyBboxPatch(
        (PILL_X, y), PILL_W, h,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        linewidth=0.0, facecolor=sentinel_color, zorder=3,
    )
    axM.add_patch(pill)
    axM.text(PILL_X + PILL_W / 2, y + h / 2, sentinel,
             fontsize=5.5, ha="center", va="center",
             family="monospace", color="white", fontweight="bold", zorder=4)
    body_x = PILL_X + PILL_W + 0.010
    body_w = FIM_RIGHT - body_right_inset - body_x
    rect = mpatches.FancyBboxPatch(
        (body_x, y), body_w, h,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        linewidth=0.6, facecolor=body_color,
        edgecolor=EDGE.get(body_color, "#888"), zorder=3,
    )
    axM.add_patch(rect)
    fam = "monospace" if kind == "code" else "serif"
    style = "italic" if kind == "cot" else "normal"
    axM.text(body_x + 0.010, y + h / 2, body_text,
             fontsize=body_fs, va="center", ha="left",
             family=fam, style=style, zorder=4)


add_fim_row(0.74, 0.10, "<fim_prefix>",
            "def A(...):\n    ...; result = B(x); ...   # caller\ndef B(x):           # target",
            C_CTX, "#3b6fb8", 5.9, "code", body_right_inset=0.020)

add_fim_row(0.62, 0.10, "<fim_suffix>",
            "def C(z):\n    return z * 2           # callee",
            C_CONT, "#7a4a8e", 5.9, "code", body_right_inset=0.020)

sentinel_only_y, sentinel_only_h = 0.51, 0.07
PILL_W_S = 0.130
PILL_X_S = FIM_LEFT + 0.020
pill_mid = mpatches.FancyBboxPatch(
    (PILL_X_S, sentinel_only_y), PILL_W_S, sentinel_only_h,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    linewidth=0.0, facecolor="#cc9a00", zorder=3,
)
axM.add_patch(pill_mid)
axM.text(PILL_X_S + PILL_W_S / 2, sentinel_only_y + sentinel_only_h / 2,
         "<fim_middle>",
         fontsize=5.5, ha="center", va="center",
         family="monospace", color="white", fontweight="bold", zorder=4)
axM.text(PILL_X_S + PILL_W_S + 0.014,
         sentinel_only_y + sentinel_only_h / 2,
         "$\\leftarrow$ generation starts here",
         fontsize=6.0, ha="left", va="center",
         color="#cc9a00", style="italic", zorder=4)

rationale_y, rationale_h = 0.31, 0.13
body_x_r = PILL_X_S + PILL_W_S + 0.010
body_w_r = FIM_RIGHT - 0.110 - body_x_r
rect_r = mpatches.FancyBboxPatch(
    (body_x_r, rationale_y), body_w_r, rationale_h,
    boxstyle="round,pad=0.003,rounding_size=0.008",
    linewidth=0.6, facecolor=C_RATIONALE, edgecolor="#cc9a00", zorder=3,
)
axM.add_patch(rect_r)
axM.text(body_x_r + 0.010, rationale_y + rationale_h / 2,
         "Iterate over x, delegate per-element work\n"
         "to C, then aggregate results.",
         fontsize=6.2, va="center", ha="left",
         family="serif", style="italic", zorder=4)

body_y, body_h = 0.16, 0.14
rect_b = mpatches.FancyBboxPatch(
    (body_x_r, body_y), body_w_r, body_h,
    boxstyle="round,pad=0.003,rounding_size=0.008",
    linewidth=0.6, facecolor=C_ACT, edgecolor=EDGE[C_ACT], zorder=3,
)
axM.add_patch(rect_b)
axM.text(body_x_r + 0.010, body_y + body_h / 2,
         "out = []\nfor v in x: out.append(C(v))\nreturn out",
         fontsize=5.9, va="center", ha="left",
         family="monospace", zorder=4)

axM.text(FIM_RIGHT - 0.095, rationale_y + rationale_h / 2,
         "rationale", fontsize=6.3, ha="left", va="center",
         color="#a07000", style="italic")
axM.text(FIM_RIGHT - 0.095, body_y + body_h / 2,
         "B's body", fontsize=6.3, ha="left", va="center",
         color="#cc7a00", style="italic")

axM.text((FIM_LEFT + FIM_RIGHT) / 2, 0.110,
         "SFT loss applied only to the target  "
         "(rationale $\\to$ function body)",
         fontsize=7.0, ha="center", va="center",
         color="#333333")


# =============================================================
# RIGHT PANEL — Result bars
# =============================================================
axR = fig.add_subplot(gs[0, 2])
sizes = ["7B", "14B", "32B"]
post_v = [15.0, 26.2, 31.8]
ours_v = [17.8, 29.2, 35.1]
post_l = [11.3, 18.0, 24.7]
ours_l = [15.0, 22.0, 26.8]

x = np.arange(len(sizes))
w = 0.20

axR.bar(x - 1.5*w, post_v, w, color="#cccccc", edgecolor="#666666",
        linewidth=0.4)
axR.bar(x - 0.5*w, ours_v, w, color="#c0392b", edgecolor="#7a2418",
        linewidth=0.4)
axR.bar(x + 0.5*w, post_l, w, color="#e8e8e8", edgecolor="#888888",
        linewidth=0.4, hatch="///")
axR.bar(x + 1.5*w, ours_l, w, color="#e88675", edgecolor="#a34c3a",
        linewidth=0.4, hatch="///")

for xi, p, o in zip(x, post_v, ours_v):
    axR.annotate(f"+{o - p:.1f}",
                 xy=(xi - 0.5*w, o), xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", fontsize=7.5, color="#7a2418",
                 fontweight="bold")
for xi, p, o in zip(x, post_l, ours_l):
    axR.annotate(f"+{o - p:.1f}",
                 xy=(xi + 1.5*w, o), xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", fontsize=7.0, color="#a34c3a",
                 fontweight="bold")

axR.set_xticks(x)
axR.set_xticklabels(sizes, fontsize=9)
axR.set_xlabel("Qwen2.5-Coder model size", fontsize=8.5)
axR.set_ylabel("SWE-Bench resolved (%)", fontsize=8.5)
axR.set_ylim(0, 42)
axR.tick_params(axis="y", labelsize=7.5)
axR.grid(axis="y", linestyle=":", linewidth=0.3, color="grey", alpha=0.6)
axR.set_axisbelow(True)
axR.spines["top"].set_visible(False)
axR.spines["right"].set_visible(False)

post_handle = mpatches.Patch(facecolor="#cccccc", edgecolor="#666",
                              linewidth=0.4, label="post-train only")
ours_handle = mpatches.Patch(facecolor="#c0392b", edgecolor="#7a2418",
                              linewidth=0.4, label="+ FIM (ours)")
v_handle = mpatches.Patch(facecolor="white", edgecolor="#666",
                           linewidth=0.4, label="Verified")
l_handle = mpatches.Patch(facecolor="white", edgecolor="#666",
                           linewidth=0.4, hatch="///", label="Lite")

leg1 = axR.legend(handles=[post_handle, ours_handle],
                  loc="upper left", bbox_to_anchor=(0.02, 1.00),
                  fontsize=6.5, frameon=False,
                  handlelength=1.3, handletextpad=0.4)
axR.add_artist(leg1)
axR.legend(handles=[v_handle, l_handle],
           loc="upper left", bbox_to_anchor=(0.02, 0.84),
           fontsize=6.5, frameon=False,
           handlelength=1.3, handletextpad=0.4)

axR.set_title("SWE-Bench gain across scales",
              fontsize=9.5, fontweight="bold", pad=4)

out = ROOT / "teaser_v20.pdf"
fig.savefig(out, bbox_inches="tight")
out_png = ROOT / "teaser_v20.png"
fig.savefig(out_png, dpi=160, bbox_inches="tight")
print(f"wrote {out}")