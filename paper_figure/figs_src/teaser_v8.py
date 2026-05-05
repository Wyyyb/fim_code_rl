"""Teaser v7: fix sub-header overlap, clarify SFT framing, add 'structurally similar' to title."""
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
C_INPUT_TINT = "#eef2f7"   # subtle blue-grey tint for "input" group
C_TARGET_TINT = "#fdf3e9"  # subtle orange tint for "target" group


def add_block(ax, x, y, w, h, color, text, monospace=True, fs=7.0):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=0.7, facecolor=color, edgecolor=EDGE[color], zorder=2,
    )
    ax.add_patch(rect)
    fam = "monospace" if monospace else "serif"
    ax.text(x + 0.020, y + h / 2, text, fontsize=fs, va="center",
            ha="left", family=fam, zorder=3)


fig = plt.figure(figsize=(13.0, 3.5))

# Reduce top margin so we have a real gap below the header
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.40, 1.65, 1.00],
    wspace=0.18,
    left=0.012, right=0.985, bottom=0.08, top=0.80,
)

# Header — now explicit "structurally similar" wording
fig.text(0.5, 0.955,
         r"Function call site  $\sim$  agent step  "
         r"$\mathit{(structurally\ similar)}$  "
         r"$\Rightarrow$  use as self-supervised FIM mid-training prior",
         fontsize=11.0, ha="center", va="center", fontweight="bold",
         color="#1a1a1a")

# =============================================================
# LEFT PANEL — analogy
# =============================================================
axL = fig.add_subplot(gs[0, 0])
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.axis("off")

# Sub-headers — moved to y=1.04 (above the axes box) so they don't overlap blocks
axL.text(0.235, 1.01, "Function call site", fontsize=9.5,
         fontweight="bold", ha="center", va="bottom")
axL.text(0.745, 1.01, "Coding-agent step", fontsize=9.5,
         fontweight="bold", ha="center", va="bottom")

# Pull blocks down so they start clearly below subheaders
ROW_H = 0.180
GAP   = 0.030
y0    = 0.74        # was 0.78 in v6; lowered to give breathing room
def lrow(i): return y0 - i * (ROW_H + GAP)

LW = 0.45
RW = 0.45
LX = 0.020
RX = 0.535

add_block(axL, LX, lrow(0), LW, ROW_H, C_CTX,
          "cfg = load_config(path)\nx = preprocess(cfg, raw)", fs=7.0)
add_block(axL, LX, lrow(1), LW, ROW_H, C_ACT,
          "y = transform(x, cfg.mode)", fs=7.0)
add_block(axL, LX, lrow(2), LW, ROW_H, C_RET,
          "# returns: tensor (B, D)", fs=7.0)
add_block(axL, LX, lrow(3), LW, ROW_H, C_CONT,
          "loss = criterion(y, target)\nloss.backward()", fs=7.0)

add_block(axL, RX, lrow(0), RW, ROW_H, C_CTX,
          "history $h_t$: prior tool calls,\nfile contents, plan",
          monospace=False, fs=7.2)
add_block(axL, RX, lrow(1), RW, ROW_H, C_ACT,
          "action $a_t$: run_tests(...)",
          monospace=False, fs=7.2)
add_block(axL, RX, lrow(2), RW, ROW_H, C_RET,
          "obs. $o_{t+1}$: AssertionError",
          monospace=False, fs=7.2)
add_block(axL, RX, lrow(3), RW, ROW_H, C_CONT,
          "next: reason about error,\npropose next edit",
          monospace=False, fs=7.2)

# Bidirectional analogy arrows
for i in range(4):
    yc = lrow(i) + ROW_H / 2
    arr = FancyArrowPatch(
        (0.475, yc), (0.530, yc),
        arrowstyle="<->", mutation_scale=10,
        linewidth=0.9, color="#666666", zorder=4,
    )
    axL.add_patch(arr)

# Bottom caption — keep at lower position, no longer competing with sub-headers
axL.text(0.50, 0.005,
         "analogous structure: context $\\to$ call/action $\\to$ "
         "external return $\\to$ continuation",
         fontsize=7.4, ha="center", va="bottom",
         color="#444444", style="italic")


# =============================================================
# MIDDLE PANEL — function-level FIM as SFT (input → target)
# =============================================================
axM = fig.add_subplot(gs[0, 1])
axM.set_xlim(0, 1); axM.set_ylim(0, 1)
axM.axis("off")

# Title moved up just like left panel
axM.text(0.5, 1.01, "Function-level FIM mid-training (SFT)",
         fontsize=9.5, fontweight="bold", ha="center", va="bottom")

PDG_LEFT, PDG_RIGHT = 0.005, 0.27
FIM_LEFT, FIM_RIGHT = 0.30, 0.99

# ---------- PDG diagram ----------
pdg_bg = mpatches.FancyBboxPatch(
    (PDG_LEFT, 0.13), PDG_RIGHT - PDG_LEFT, 0.79,
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

axM.text((PDG_LEFT + PDG_RIGHT) / 2, 0.18,
         "B selected via\nPDG + $\\hat{H}$ + $\\hat{I}$",
         fontsize=6.6, ha="center", va="center",
         color="#555", style="italic")

arr_bridge = FancyArrowPatch(
    (PDG_RIGHT + 0.005, 0.51), (FIM_LEFT - 0.005, 0.51),
    arrowstyle="-|>", mutation_scale=10,
    linewidth=0.9, color="#666", zorder=2,
)
axM.add_patch(arr_bridge)

# ---------- FIM sample with explicit input / target grouping ----------
bg = mpatches.FancyBboxPatch(
    (FIM_LEFT, 0.13), FIM_RIGHT - FIM_LEFT, 0.79,
    boxstyle="round,pad=0.002,rounding_size=0.010",
    linewidth=0.5, facecolor=C_BG_FIM, edgecolor="#bbb5cc",
    zorder=1,
)
axM.add_patch(bg)

# --- INPUT group background (covers fim_prefix + fim_suffix + fim_middle sentinel) ---
INPUT_TOP = 0.88
INPUT_BOTTOM = 0.50    # just above the rationale-and-body block
input_group = mpatches.FancyBboxPatch(
    (FIM_LEFT + 0.006, INPUT_BOTTOM),
    (FIM_RIGHT - FIM_LEFT) - 0.012,
    INPUT_TOP - INPUT_BOTTOM,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    linewidth=0.0, facecolor=C_INPUT_TINT, zorder=1.5,
)
axM.add_patch(input_group)
# 'input (prompt)' label on the left edge of the input group
axM.text(FIM_LEFT + 0.014, INPUT_TOP - 0.012, "input prompt",
         fontsize=6.4, ha="left", va="top",
         color="#3b6fb8", fontweight="bold", style="italic", zorder=10)

# --- TARGET group background ---
TARGET_TOP = 0.485
TARGET_BOTTOM = 0.18
target_group = mpatches.FancyBboxPatch(
    (FIM_LEFT + 0.006, TARGET_BOTTOM),
    (FIM_RIGHT - FIM_LEFT) - 0.012,
    TARGET_TOP - TARGET_BOTTOM,
    boxstyle="round,pad=0.0,rounding_size=0.008",
    linewidth=0.0, facecolor=C_TARGET_TINT, zorder=1.5,
)
axM.add_patch(target_group)
axM.text(FIM_LEFT + 0.014, TARGET_TOP - 0.012, "SFT target",
         fontsize=6.4, ha="left", va="top",
         color="#cc7a00", fontweight="bold", style="italic", zorder=10)


def add_fim_row(y, h, sentinel, body_text, body_color, sentinel_color,
                body_fs=6.2, kind="code", body_right_inset=0.075):
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


# Within input group:
#   <fim_prefix> ... (file context up to B's signature)
#   <fim_suffix> ... (file context after B)
#   <fim_middle>     (just the sentinel — signals start of generation)
add_fim_row(0.74, 0.10, "<fim_prefix>",
            "def A(...):\n    ...; result = B(x); ...   # caller\ndef B(x):                              # target",
            C_CTX, "#3b6fb8", 5.9, "code", body_right_inset=0.012)

add_fim_row(0.62, 0.10, "<fim_suffix>",
            "def C(z):\n    return z * 2                       # callee",
            C_CONT, "#7a4a8e", 5.9, "code", body_right_inset=0.012)

# fim_middle sentinel only — no body in input; the target follows
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
# Hint that input ends here
axM.text(PILL_X_S + PILL_W_S + 0.014,
         sentinel_only_y + sentinel_only_h / 2,
         "$\\leftarrow$ generation starts here",
         fontsize=6.0, ha="left", va="center",
         color="#cc9a00", style="italic", zorder=4)

# --- Target group: rationale + body (the SFT label) ---
rationale_y, rationale_h = 0.34, 0.13
body_x_r = PILL_X_S + PILL_W_S + 0.010
body_w_r = FIM_RIGHT - 0.075 - body_x_r
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

body_y, body_h = 0.19, 0.14
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

# Right-side annotation labels for the two target sub-blocks
axM.text(FIM_RIGHT - 0.060, rationale_y + rationale_h / 2,
         "rationale", fontsize=6.3, ha="left", va="center",
         color="#a07000", style="italic")
axM.text(FIM_RIGHT - 0.060, body_y + body_h / 2,
         "B's body", fontsize=6.3, ha="left", va="center",
         color="#cc7a00", style="italic")

# Bottom caption
axM.text((FIM_LEFT + FIM_RIGHT) / 2, 0.145,
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

out = ROOT / "teaser_v8.pdf"
fig.savefig(out, bbox_inches="tight")
out_png = ROOT / "teaser_v8.png"
fig.savefig(out_png, dpi=160, bbox_inches="tight")
print(f"wrote {out}")