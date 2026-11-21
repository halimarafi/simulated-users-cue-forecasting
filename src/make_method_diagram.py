"""Draws figures/00_method.png: the study design in one picture. Run from the project folder: python src/make_method_diagram.py"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
ACCENT, TINT = "#2a78d6", "#e8f0fb"
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 120); ax.set_ylim(0, 46); ax.axis("off")

def box(x, y, w, h, title, lines, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=TINT if accent else "white", ec=ACCENT if accent else EDGE, lw=2 if accent else 1.2))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.6 - 3.6 * i, line, ha="center", va="center", fontsize=9, color=MUTED)

def arrow(x0, y0, x1, y1, dashed=False, label=None, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=EDGE,
                                 linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")

box(1, 15, 17, 19, "Real data", ["42 people", "2 days each", "Brands et al. 2024"])
box(23, 27, 17, 13, "Basic trials", ["no cue", "~8,000 choices"])
box(23, 5, 17, 13, "Cued trials", ["future or past cue", "~8,000 choices"])
box(45, 25, 24, 17, "Three models", ["0 · simple statistics", "1 · patience", "2 · patience + decision time"])
box(73.5, 25, 16, 17, "Copies", ["one per person", "per model", "200 repeats each"])
box(93.5, 25, 25.5, 17, "Forecasts for cued trials", ["A · no effect", "B · earlier studies", "C · learned from others"])
box(82, 1, 37, 15, "Compare with reality", ["each decision · whole session · run to run", "person by person"], accent=True)

arrow(18, 28, 23, 33); arrow(18, 21, 23, 12)
arrow(40, 33.5, 45, 33.5, label="fit", lx=42.5, ly=36)
arrow(69, 33.5, 73.5, 33.5, label="build", lx=71.25, ly=36)
arrow(89.5, 33.5, 93.5, 33.5)
arrow(106.25, 25, 106.25, 16)
arrow(40, 9, 82, 9, label="never seen by the models · used only to score them", lx=61, ly=6.6)
arrow(40, 15, 98, 25, dashed=True, label="forecast C only: other people's cued trials", lx=61, ly=13.4)
ax.text(1, 44.5, "Simulated users learn from trials without the cue, then forecast the trials with it", fontsize=12, fontweight="bold", color=INK, va="center")

out = Path(__file__).resolve().parents[1] / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "00_method.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out / "00_method.png")
