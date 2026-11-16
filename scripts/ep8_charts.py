#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Episode 8 charts, from the episode's bench/summary.json (written by ep8_summary.py).

  uv run --no-project scripts/ep8_charts.py

  ep8_verdict_grid.png   Einstein's claims × library date, as the checker labelled them. Each cell
                         holds the two phrasings; a dot marks a label my reading disagrees with.
  ep8_groups.png         Share of phrasings supported, per claim group and library date, after reading.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

EPISODE = Path(__file__).resolve().parents[2] / "ContentEngine" / "episodes" / "ep08-beyondrag"
OUT = EPISODE / "screenshots"
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
# Diverging pair: blue (supported) <-> red (contradicted), gray midpoint (not stated). Validated.
FILL = {"supported": "#2a78d6", "contradicted": "#e34948", "not stated": "#f0efec"}
GLYPH = {"supported": "✓", "contradicted": "✗", "not stated": "–"}
GLYPH_INK = {"supported": "#ffffff", "contradicted": "#ffffff", "not stated": MUTED}
# Categorical, slots 1-4 in fixed order. Validated; slots 3-4 sit under 3:1, so every line is labelled.
SERIES = {"E": "#2a78d6", "K": "#eb6834", "W invented": "#1baf7a", "W of the day": "#eda100"}
SERIES_NAME = {"E": "Einstein's claims", "K": "Already in print by 1905",
               "W invented": "Made-up errors", "W of the day": "Errors of the day"}

LABELS = {
    "E01": "Magnet & conductor: one effect, two theories", "E02": "Nothing shows absolute rest",
    "E03": "Same laws in every moving system", "E04": "Light: same speed for every observer",
    "E05": "The ether is superfluous", "E06": "Set clocks by light signals",
    "E07": "Simultaneity depends on the observer", "E08": "Shortening is mutual",
    "E09": "Shortening comes from measuring, not forces", "E10": "Transformation from two principles",
    "E11": "A light sphere is a sphere for both", "E12": "Nothing faster than light",
    "E13": "A moving clock runs slow", "E14": "Travelling clock comes back behind",
    "E15": "Clock at the equator runs slow", "E16": "Velocities don't simply add",
    "E17": "Light's speed can't be increased", "E18": "E and B depend on the observer",
    "E19": "One explanation for magnet & conductor", "E20": "Light's energy & frequency change alike",
    "E21": "Charge is the same for every observer", "E22": "Kinetic energy infinite at light speed",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def year(d: str) -> str:
    return "June 1905" if d.startswith("1905") else d[:4]


def grid(summary: dict) -> None:
    claims = [c for c in summary["claims"] if c.startswith("E")]
    dates = sorted(next(iter(summary["claims"].values()))["by_date"])
    fig, ax = plt.subplots(figsize=(10, 9.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cw, ch, gap, step = 1.0, 0.84, 0.04, 1.22  # step > cw leaves a gap between years
    for row, cid in enumerate(claims):
        y = len(claims) - 1 - row
        for col, d in enumerate(dates):
            cell = summary["claims"][cid]["by_date"][d]
            for half, ph in enumerate(("1", "2")):
                v = cell[ph]
                x = col * step + half * (cw / 2) + gap / 2
                w = cw / 2 - gap
                ax.add_patch(Rectangle((x, y + (1 - ch) / 2), w, ch, facecolor=FILL[v["checker"]], edgecolor=SURFACE, linewidth=0, joinstyle="round"))
                ax.text(x + w / 2, y + 0.5, GLYPH[v["checker"]], ha="center", va="center", fontsize=9, color=GLYPH_INK[v["checker"]], fontweight="bold")
                if v["checker"] != v["after_reading"]:
                    ax.plot(x + w - 0.07, y + (1 + ch) / 2 - 0.12, marker="o", markersize=3.2, color=INK, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.text(-0.12, y + 0.5, LABELS[cid], ha="right", va="center", fontsize=9, color=INK)
    for col, d in enumerate(dates):
        ax.text(col * step + cw / 2, len(claims) + 0.25, year(d), ha="center", va="bottom", fontsize=10, color=INK2, fontweight="bold")
    ax.set_xlim(-0.1, (len(dates) - 1) * step + cw + 0.05)
    ax.set_ylim(-0.1, len(claims) + 0.8)
    ax.axis("off")
    fig.text(0.02, 0.985, "What my hallucination checker said about Einstein's 1905 claims", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.957, "Library of physics published by each date · two phrasings per claim · Episode 4 checker, unchanged",
             fontsize=10, color=INK2, va="top")
    legend = [("supported", "Supported"), ("contradicted", "Contradicted"), ("not stated", "Not stated")]
    lx = 0.02
    for key, name in legend:
        fig.patches.append(Rectangle((lx, 0.018), 0.016, 0.016, transform=fig.transFigure, facecolor=FILL[key], edgecolor=RULE if key == "not stated" else FILL[key], linewidth=0.6))
        fig.text(lx + 0.022, 0.026, name, fontsize=9.5, color=INK2, va="center")
        lx += 0.15
    fig.lines.append(plt.Line2D([lx + 0.008], [0.026], marker="o", markersize=3.5, color=INK, transform=fig.transFigure))
    fig.text(lx + 0.02, 0.026, "My reading of the sources disagrees", fontsize=9.5, color=INK2, va="center")
    fig.subplots_adjust(left=0.36, right=0.99, top=0.92, bottom=0.06)
    fig.savefig(OUT / "ep8_verdict_grid.png", facecolor=SURFACE)
    plt.close(fig)


def groups(summary: dict) -> None:
    dates = sorted(summary["table"]["E"])
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(dates)))
    series = {}
    for g, color in SERIES.items():
        cells = [summary["table"][g][d]["after_reading"] for d in dates]
        series[g] = [100 * c["supported"] / sum(c.values()) for c in cells]
        ax.plot(xs, series[g], color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round", zorder=3)

    def spread(points, gap=6.5):
        """Label positions at least `gap` points apart, kept in the same order as the data."""
        order = sorted(points, key=lambda g: points[g])
        placed = {}
        for g in order:
            placed[g] = max(points[g], placed[order[order.index(g) - 1]] + gap) if order.index(g) else points[g]
        return placed

    for end, x, ha, text in ((-1, xs[-1] + 0.12, "left", True), (0, xs[0] - 0.12, "right", False)):
        pos = spread({g: ys[end] for g, ys in series.items()})
        for g, ys in series.items():
            label = f"{SERIES_NAME[g]}  {ys[end]:.0f}%" if text else f"{ys[end]:.0f}%"
            ax.text(x, pos[g], label, va="center", ha=ha, fontsize=10, color=INK if text else INK2)
    ax.set_xticks(xs, [year(d) for d in dates])
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, len(dates) - 1 + 2.1)
    ax.set_xlabel("Library: everything published up to", labelpad=8)
    fig.text(0.02, 0.97, "Share of claims the checker should call \"supported\", by library date", fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.02, 0.92, "After reading every verdict against its four sources · two phrasings per claim", fontsize=10, color=INK2, va="top")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.14)
    fig.savefig(OUT / "ep8_groups.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = json.loads((EPISODE / "bench" / "summary.json").read_text())
    grid(data)
    groups(data)
    print("wrote", OUT / "ep8_verdict_grid.png", OUT / "ep8_groups.png")
