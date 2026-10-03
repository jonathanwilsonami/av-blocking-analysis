"""5x5 likelihood x consequence risk matrix.

Matrix structure, band labels, per-cell risk levels, and colors follow the
team's reference template. Only the generic matrix logic is reused here; the
hazard-specific probability and impact estimates come from
:mod:`av_blocking.frequency` and :mod:`av_blocking.severity`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .severity import IMPACT_LABELS

PROB_LABELS = ["Rare", "Unlikely", "Moderate", "Likely", "Almost certain"]

# The band label is NOT a simple threshold on the prob x impact score -- e.g.
# score 4 reads "Low" at (2, 2) but "Medium" at (1, 4) and (4, 1) -- so each
# cell's level is looked up directly; the score is shown alongside it.
LEVEL_OF = {
    (1, 1): "Very low",
    (1, 2): "Very low",
    (1, 3): "Low",
    (1, 4): "Medium",
    (1, 5): "Medium",
    (2, 1): "Very low",
    (2, 2): "Low",
    (2, 3): "Medium",
    (2, 4): "Medium",
    (2, 5): "High",
    (3, 1): "Low",
    (3, 2): "Medium",
    (3, 3): "Medium",
    (3, 4): "High",
    (3, 5): "Very high",
    (4, 1): "Medium",
    (4, 2): "Medium",
    (4, 3): "High",
    (4, 4): "Very high",
    (4, 5): "Extreme",
    (5, 1): "Medium",
    (5, 2): "High",
    (5, 3): "Very high",
    (5, 4): "Extreme",
    (5, 5): "Extreme",
}
LEVELS = ["Very low", "Low", "Medium", "High", "Very high", "Extreme"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
LEVEL_COLORS = {
    "Very low": "#3AB34A",
    "Low": "#2F903B",
    "Medium": "#F8EB10",
    "High": "#F79122",
    "Very high": "#E91720",
    "Extreme": "#BB121A",
}
LEVEL_TEXT_COLORS = {"Medium": "#4d3b00"}  # dark text on the bright-yellow band

HEADER_BG = "#F3F6FB"
ARROW_COLOR = "#5B5FC7"


def level_of(prob_1to5: int, impact_1to5: int) -> str:
    return LEVEL_OF[(prob_1to5, impact_1to5)]


def probability_band(p: float, edges=config.PROB_BAND_EDGES) -> int:
    """1-based probability band for P(>=1 event in the reference horizon)."""
    return min(int(np.searchsorted(edges, p)), 4) + 1


@dataclass
class Placement:
    label: str
    prob: int  # 1..5
    impact: int  # 1..5

    @property
    def level(self) -> str:
        return level_of(self.prob, self.impact)

    @property
    def score(self) -> int:
        return self.prob * self.impact


def plot_risk_matrix(placements: list[Placement], title: str = "", footnote: str = ""):
    """Draw the reference-style matrix and badge each placement in its cell."""
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.axis("off")

    def draw_cell(x, y, fc, lw=1.3):
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=fc, edgecolor="black", lw=lw, zorder=1))

    draw_cell(0, 6, HEADER_BG)
    for c in range(5):
        draw_cell(1 + c, 6, HEADER_BG)
        ax.text(
            1.5 + c,
            6.5,
            f"{IMPACT_LABELS[c]}\n{c + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=2,
        )

    by_cell: dict[tuple[int, int], list[str]] = defaultdict(list)
    for p in placements:
        by_cell[(p.prob, p.impact)].append(p.label)

    for r in range(5):
        prob_level = 5 - r
        y = 5 - r
        draw_cell(0, y, HEADER_BG)
        ax.text(
            0.5,
            y + 0.5,
            f"{PROB_LABELS[prob_level - 1]}\n{prob_level}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=2,
        )
        for c in range(5):
            impact_level = c + 1
            level = level_of(prob_level, impact_level)
            draw_cell(1 + c, y, LEVEL_COLORS[level])
            txt_color = LEVEL_TEXT_COLORS.get(level, "white")
            labels = by_cell.get((prob_level, impact_level), [])
            ty = y + 0.72 if labels else y + 0.5
            ax.text(
                1.5 + c,
                ty,
                f"{level} {prob_level * impact_level}",
                ha="center",
                va="center",
                fontsize=10.5 if labels else 11,
                fontweight="bold",
                color=txt_color,
                zorder=2,
            )
            if labels:
                ax.add_patch(
                    plt.Rectangle(
                        (1 + c, y), 1, 1, facecolor="none", edgecolor="black", lw=4, zorder=3
                    )
                )
                n = len(labels)
                for i, lab in enumerate(labels):
                    bx = 1.5 + c + (i - (n - 1) / 2) * min(0.3, 0.9 / n)
                    ax.text(
                        bx,
                        y + 0.32,
                        lab,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=1.0),
                    )

    ax.text(3.5, 7.55, "Impact", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.text(
        3.5,
        7.2,
        "How severe would the outcomes be if the risk occurred?",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontstyle="italic",
    )
    ax.annotate(
        "",
        xy=(5.9, 7.02),
        xytext=(1.1, 7.02),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.2),
    )
    ax.text(
        -1.0,
        3.5,
        "Probability",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        rotation=90,
    )
    ax.text(
        -0.65,
        3.5,
        "What is the probability the risk will happen?",
        ha="center",
        va="center",
        fontsize=10.5,
        fontstyle="italic",
        rotation=90,
    )
    ax.annotate(
        "",
        xy=(-0.2, 5.9),
        xytext=(-0.2, 1.1),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.2),
    )

    ax.set_xlim(-1.9, 6.3)
    ax.set_ylim(0.75, 7.9)
    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    if footnote:
        fig.text(0.5, 0.0, footnote, ha="center", va="bottom", fontsize=9.5, wrap=True)
    fig.tight_layout()
    return fig, ax
