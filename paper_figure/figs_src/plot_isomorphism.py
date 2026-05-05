"""Isomorphism figure (Fig 2): top row = function call site, bottom row =
agent step, with vertical dashed lines aligning matching positions.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent

C_CTX  = "#cfe2f3"
C_ACT  = "#fce5cd"
C_RET  = "#d9ead3"
C_CONT = "#e6d5f0"
EDGE   = {C_CTX: "#3b6fb8", C_ACT: "#cc7a00",
          C_RET: "#3a8c4f", C_CONT: "#7a4a8e"}


def block(ax, x, y, w, h, color, head, body, head_color=None):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        linewidth=0.7, facecolor=color, edgecolor=EDGE[color], zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.13, head, ha="center", va="top",
            fontsize=8.4, fontweight="bold",
            color=head_color or EDGE[color], zorder=3)
    ax.text(x + w / 2, y + h - 0.32, body, ha="center", va="top",
            fontsize=7.6, family="serif", color="#222222", zorder=3,
            wrap=True)


def arrow(ax, x1, x2, y, color="#666666"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=color),
                zorder=1)


fig, ax = plt.subplots(figsize=(10.0, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
ax.axis("off")

W = 2.0
H = 1.05
GAP_X = 0.30  # gap between blocks within a row
LEFT_PAD = 0.40
xs = [LEFT_PAD + i * (W + GAP_X) for i in range(4)]
y_top = 1.95
y_bot = 0.30

# row label
ax.text(0.05, y_top + H / 2, "Function\ncall site",
        ha="left", va="center", fontsize=9.0, fontweight="bold",
        color="#444444", style="italic")
ax.text(0.05, y_bot + H / 2, "Agent\nstep",
        ha="left", va="center", fontsize=9.0, fontweight="bold",
        color="#444444", style="italic")

# top row: function call site
block(ax, xs[0], y_top, W, H, C_CTX, "pre-call context",
      "args bound,\nstate established")
block(ax, xs[1], y_top, W, H, C_ACT, "call",
      "callee invocation\n(fn name + args)")
block(ax, xs[2], y_top, W, H, C_RET, "return",
      "value computed\noutside scope")
block(ax, xs[3], y_top, W, H, C_CONT, "downstream use",
      "code consuming\nthe returned value")

# arrows top
for i in range(3):
    arrow(ax, xs[i] + W, xs[i + 1], y_top + H / 2)

# bottom row: agent step
block(ax, xs[0], y_bot, W, H, C_CTX, r"history $h_t$",
      "prior tool calls,\nfile contents, plan")
block(ax, xs[1], y_bot, W, H, C_ACT, r"action $a_t$",
      "tool call /\nshell command")
block(ax, xs[2], y_bot, W, H, C_RET, r"observation $o_{t+1}$",
      "external output\n(stdout, errors)")
block(ax, xs[3], y_bot, W, H, C_CONT, "continuation",
      "next reasoning step\nconditioned on trace")

# arrows bottom
for i in range(3):
    arrow(ax, xs[i] + W, xs[i + 1], y_bot + H / 2)

# vertical dashed alignment lines between matching pairs
for i, color in enumerate([EDGE[C_CTX], EDGE[C_ACT], EDGE[C_RET], EDGE[C_CONT]]):
    cx = xs[i] + W / 2
    ax.plot([cx, cx], [y_top, y_bot + H], linestyle=":", color=color,
            linewidth=0.9, alpha=0.7, zorder=1)

# central isomorphism caption
ax.text(LEFT_PAD + 2 * W + 1.5 * GAP_X, (y_top + y_bot + H) / 2 + 0.08,
        "structural isomorphism",
        ha="center", va="center", fontsize=9.0,
        color="#444444", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#bbbbbb", linewidth=0.5))

# what FIM masks: bracket spanning call+return on the top row
brack_y = y_top + H + 0.18
brack_x1 = xs[1] + 0.05
brack_x2 = xs[2] + W - 0.05
ax.plot([brack_x1, brack_x1, brack_x2, brack_x2],
        [brack_y - 0.07, brack_y, brack_y, brack_y - 0.07],
        color="#c0392b", linewidth=0.9)
ax.text((brack_x1 + brack_x2) / 2, brack_y + 0.05,
        "function-aware FIM masks the callee body\n"
        "(call site $\\to$ return value)",
        ha="center", va="bottom", fontsize=8.0, color="#c0392b",
        style="italic")

plt.tight_layout(pad=0.2)
out = ROOT / "figs" / "isomorphism.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
