"""Bar chart of repository counts per category for the mid-training corpus.

Reads data/code_repo_final_display.csv, aggregates by category, renders a
horizontal bar chart sorted by count, and writes figs/dataset_categories.pdf.
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "code_repo_final_display.csv"
OUT_PATH = ROOT / "figs" / "dataset_categories.pdf"

DISPLAY_NAME = {
    "Category 1: From Scratch":              "Reference Implementations",
    "Category 2: Domain Specific":           "Domain-Specific Apps",
    "Category 3: Algorithms":                "Algorithms",
    "Category 4: Scientific Computing":      "Scientific Computing",
    "Category 5: Small Frameworks":          "Small Frameworks",
    "Category 6: Visualization and Games":   "Visualization & Games",
    "Category 7: Educational":               "Educational",
    "Category 8: Compilers":                 "Compilers",
    "Category 9: Data Processing":           "Data Processing",
    "Category 10: Networking and Security":  "Networking & Security",
}

counts = Counter()
with CSV_PATH.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        counts[row["category"]] += 1

items = sorted(counts.items(), key=lambda kv: -kv[1])
labels = [DISPLAY_NAME[k] for k, _ in items]
values = [v for _, v in items]
total = sum(values)

fig, ax = plt.subplots(figsize=(5.6, 2.6))
y_pos = list(range(len(labels)))[::-1]
bars = ax.barh(
    y_pos, values,
    color="#3b6fb8", edgecolor="#1f3f6d", linewidth=0.5, height=0.72,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel(f"# repositories  (total {total})", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8.5)
ax.set_xlim(0, max(values) * 1.18)
ax.grid(axis="x", linestyle=":", linewidth=0.4, color="grey", alpha=0.6)
ax.set_axisbelow(True)

for rect, v in zip(bars, values):
    ax.text(
        rect.get_width() + max(values) * 0.012,
        rect.get_y() + rect.get_height() / 2.0,
        f"{v}",
        va="center", ha="left", fontsize=8,
    )

plt.tight_layout(pad=0.4)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"wrote {OUT_PATH}  total={total}")
