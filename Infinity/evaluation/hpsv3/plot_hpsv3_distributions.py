#!/usr/bin/env python3
"""
Plot score distributions for HPSv3 comparisons.

Usage:
  python plot_hpsv3_distributions.py --csv ../../hpsv3_comparison_results.csv --out scores.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HPSv3 score distributions")
    parser.add_argument("--csv", default="hpsv3_comparison_results.csv", help="CSV produced by eval_hpsv3.py")
    parser.add_argument("--out", default="hpsv3_score_distributions.png", help="Where to save the plot")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    score_cols = ["Score_Original", "Score_Fastvar", "Score_RL"]

    plt.style.use("ggplot")
    fig, (ax_hist, ax_violin) = plt.subplots(1, 2, figsize=(12, 5))
    stats = {}

    # Overlayed histograms
    colors = {
        "Score_Original": "#1f77b4",
        "Score_Fastvar": "#2ca02c",
        "Score_RL": "#d62728",
    }
    for col in score_cols:
        series = df[col].dropna()
        stats[col] = {"mean": series.mean()}
        ax_hist.hist(
            series,
            bins=args.bins,
            alpha=0.5,
            label=col.replace("Score_", ""),
            color=colors[col],
            density=True,
        )
        ax_hist.axvline(
            stats[col]["mean"],
            color=colors[col],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
    ax_hist.set_title("Score Distributions (Histogram)")
    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    # Annotate means near the top of the histogram for quick reference
    ymax = ax_hist.get_ylim()[1]
    for idx, col in enumerate(score_cols):
        label = col.replace("Score_", "")
        y_pos = ymax * (0.92 - 0.06 * idx)
        ax_hist.text(
            stats[col]["mean"],
            y_pos,
            f"{label} μ={stats[col]['mean']:.3f}",
            color=colors[col],
            fontsize=9,
            ha="right",
            va="center",
            rotation=90,
            backgroundcolor="white",
        )

    # Violin plot for quick comparison
    data = [df[col].dropna() for col in score_cols]
    ax_violin.violinplot(data, showmeans=True, widths=0.9)
    ax_violin.set_xticks(range(1, len(score_cols) + 1))
    ax_violin.set_xticklabels([c.replace("Score_", "") for c in score_cols])
    ax_violin.set_title("Score Distributions (Violin)")
    ax_violin.set_ylabel("Score")
    # Label means on top of the violin markers
    for idx, col in enumerate(score_cols, start=1):
        ax_violin.scatter(idx, stats[col]["mean"], color=colors[col], s=25, zorder=3)
        ax_violin.text(
            idx,
            stats[col]["mean"],
            f"{stats[col]['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors[col],
            backgroundcolor="white",
        )

    fig.suptitle("HPSv3 Score Comparison", fontsize=14)
    fig.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
