"""The two people the README follows: Person A barely responds to the cue, Person B responds strongly.

Every per-person figure draws them on top of the other dots, in their own colour and with a letter next
to the dot (the letter means the figures still work in black and white or for colour-blind readers).
"""
from matplotlib.lines import Line2D

PEOPLE = {"A": 12, "B": 30}                       # participant numbers in the Brands et al. data
COLOR = {"A": "#c2378f", "B": "#1f8a3b"}          # pink, green
INK = "#0b0b0b"
OFFSET = {"A": (-4, 8), "B": (-4, 8)}              # the letter sits just above the dot, so it cannot be read as labelling a neighbour


def mark(ax, xy, size=80, text=True):
    """xy: {"A": (x, y), "B": (x, y)}. People missing from xy are skipped."""
    for lab, (x, y) in xy.items():
        ax.scatter([x], [y], s=size, color=COLOR[lab], edgecolor="white", linewidth=1.5, zorder=6)
        if text:
            ax.annotate(lab, (x, y), xytext=OFFSET[lab], textcoords="offset points",
                        fontsize=10, fontweight="bold", color=INK, zorder=7)


def handles(size=8):
    """Legend entries for the two people."""
    return [Line2D([], [], marker="o", ls="", markersize=size, markerfacecolor=COLOR[l], markeredgecolor="white",
                   label=f"Person {l} (participant {p})") for l, p in PEOPLE.items()]


def present(index):
    """The labels of the two people who appear in `index` (lets quick runs with fewer people still work)."""
    return {l: p for l, p in PEOPLE.items() if p in set(index)}
