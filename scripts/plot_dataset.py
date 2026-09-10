#!/usr/bin/env python3
"""Create dataset distribution figures directly from the released CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def plot_distribution(counts: Counter, title: str, color: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels, values = zip(*ordered)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.32)))
        bars = ax.barh(labels, values, color=color, height=0.72)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=5, fontsize=10)
        ax.set_xlim(0, max(values) * 1.14)
        ax.set_xlabel("Persuader turns")
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=16)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets" / "dialogues.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "docs" / "assets")
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["speaker"] == "persuader"]
    if not rows:
        raise ValueError("Dataset contains no persuader turns")
    for field, title, color in (
        ("emotion", "Emotion distribution", "#4f63d8"),
        ("strategy", "Persuasion strategy distribution", "#168a85"),
    ):
        counts = Counter(row[field] for row in rows)
        if "" in counts:
            raise ValueError(f"Dataset contains missing {field} labels")
        plot_distribution(counts, title, color, args.output_dir / f"{field}-distribution.png")


if __name__ == "__main__":
    main()
