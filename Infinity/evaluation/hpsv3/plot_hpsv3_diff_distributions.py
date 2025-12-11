#!/usr/bin/env python3
"""
Plot score deltas (FastVAR/ RL minus Original) for HPSv3 comparisons.

Usage:
  python plot_hpsv3_diff_distributions.py --csv ../../hpsv3_comparison_results.csv --out diff_scores.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HPSv3 score differences vs Original")
    parser.add_argument("--csv", default="hpsv3_comparison_results.csv", help="CSV produced by eval_hpsv3.py")
    parser.add_argument("--out", default="hpsv3_score_diff_distributions.png", help="Where to save the plot")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Score_Original", "Score_Fastvar", "Score_RL"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(sorted(missing))}")

    df["Diff_Fastvar"] = df["Score_Fastvar"] - df["Score_Original"]
    df["Diff_RL"] = df["Score_RL"] - df["Score_Original"]

    plt.style.use("ggplot")
    fig, (ax_hist, ax_violin) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "Diff_Fastvar": "#2ca02c",
        "Diff_RL": "#d62728",
    }

    for col in ["Diff_Fastvar", "Diff_RL"]:
        ax_hist.hist(
            df[col].dropna(),
            bins=args.bins,
            alpha=0.6,
            label=col.replace("Diff_", ""),
            color=colors[col],
            density=True,
        )
    ax_hist.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax_hist.set_title("Score Difference Distributions (Histogram)")
    ax_hist.set_xlabel("Score difference vs Original")
    ax_hist.set_ylabel("Density")
    ax_hist.legend()

    data = [df["Diff_Fastvar"].dropna(), df["Diff_RL"].dropna()]
    ax_violin.violinplot(data, showmeans=True, widths=0.9)
    ax_violin.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax_violin.set_xticks([1, 2])
    ax_violin.set_xticklabels(["Fastvar - Original", "RL - Original"])
    ax_violin.set_title("Score Difference Distributions (Violin)")
    ax_violin.set_ylabel("Score difference")

    fig.suptitle("HPSv3 Score Differences vs Original", fontsize=14)
    fig.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
