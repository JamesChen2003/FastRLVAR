#!/usr/bin/env python
"""
FastRLVar vs FastVar visualization

Usage:
    python plot_fastrlvar.py                  # use simulated Gaussian data
    python plot_fastrlvar.py --json data.json # use real metrics if available

Expected JSON format (all arrays are lists of floats):

{
  "FastVar": {
    "similarity": [...],   # similarity(pruned, unpruned), e.g. LPIPS/SSIM/PSNR
    "quality": [...],      # absolute quality of pruned output, e.g. CLIP / DINO / VIEScore
    "speed": [...]         # e.g. images per second or 1 / latency
  },
  "FastRLVar": {
    "similarity": [...],
    "quality": [...],
    "speed": [...]
  }
}

If any key is missing, the script will fall back to simulated data
for that metric / method.
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


METHODS = ["FastVar", "FastRLVar"]


def simulate_1d_metrics(n: int = 400) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Simulate similarity, quality and speed for FastVar and FastRLVar
    using Gaussian distributions roughly matching the sketch.

    You can tweak the means/stds later if you want different shapes.
    """
    rng = np.random.default_rng(43)

    data = {}

    # --- 1) Similarity (pruned vs unpruned) ---
    # Higher is better; RLVar shifted to the right and narrower.
    sim_mean = {"FastVar": 0.70, "FastRLVar": 0.80}
    sim_std = {"FastVar": 0.10, "FastRLVar": 0.045}

    # --- 2) Absolute quality (pruned image) ---
    # Again, higher is better; RLVar > Var with smaller variance.
    qual_mean = {"FastVar": 0.70, "FastRLVar": 0.80}
    qual_std = {"FastVar": 0.10, "FastRLVar": 0.045}

    # --- 3) Speed (e.g. images per second) ---
    # RLVar is faster and more concentrated, FastVar slower and noisier.
    speed_mean = {"FastVar": 2.5, "FastRLVar": 2.3}
    speed_std = {"FastVar": 0.015, "FastRLVar": 0.15}

    for m in METHODS:
        similarity = rng.normal(sim_mean[m], sim_std[m], size=n)
        quality = rng.normal(qual_mean[m], qual_std[m], size=n)

        # Make speed weakly correlated with quality to get a nicer cloud:
        speed = rng.normal(speed_mean[m], speed_std[m], size=n)
        # speed = base_speed

        data[m] = {
            "similarity": similarity,
            "quality": quality,
            "speed": speed,
        }

    return data


def load_json_metrics(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load metrics from JSON. Missing keys will be filled later by simulation.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for m in METHODS:
        if m not in raw:
            data[m] = {}
            continue
        data[m] = {}
        for key in ("similarity", "quality", "speed"):
            if key in raw[m] and raw[m][key] is not None:
                data[m][key] = np.asarray(raw[m][key], dtype=float)
    return data


def merge_with_simulation(
    maybe_data: Dict[str, Dict[str, np.ndarray]],
    n_sim: int = 400,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For any missing metric in maybe_data, fill it with a simulated Gaussian.
    """
    sim = simulate_1d_metrics(n=n_sim)

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for m in METHODS:
        result[m] = {}
        for key in ("similarity", "quality", "speed"):
            if m in maybe_data and key in maybe_data[m]:
                result[m][key] = maybe_data[m][key]
            else:
                result[m][key] = sim[m][key]
    return result


def plot_distributions(data: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Create 3 subplots:

    (1) Similarity distribution (hist + density-like)
    (2) Quality distribution
    (3) Speed vs Quality scatter
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # ---------------- 1. Similarity distribution ----------------
    ax = axes[0]
    bins = 30

    for m in METHODS:
        ax.hist(
            data[m]["similarity"],
            bins=bins,
            density=True,
            alpha=0.4,
            label=m,
            histtype="stepfilled",
        )
    ax.set_title("Similarity (pruned vs unpruned)")
    ax.set_xlabel("Similarity (LPIPS, SSIM, etc.)")
    ax.set_ylabel("Density")
    ax.legend()

    # ---------------- 2. Absolute quality distribution ----------------
    ax = axes[1]
    for m in METHODS:
        ax.hist(
            data[m]["quality"],
            bins=bins,
            density=True,
            alpha=0.4,
            label=m,
            histtype="stepfilled",
        )
    ax.set_title("Absolute Quality (pruned output)")
    ax.set_xlabel("Quality (CLIP, VIEScore, etc.)")
    ax.set_ylabel("Density")
    ax.legend()

    # ---------------- 3. Speed vs Quality (scatter) ----------------
    ax = axes[2]
    for m in METHODS:
        ax.scatter(
            data[m]["quality"],
            data[m]["speed"],
            alpha=0.35,
            label=m,
            s=15,
        )
    ax.set_title("Speedup vs Quality")
    ax.set_xlabel("Quality (CLIP, VIEScore, etc.)")
    ax.set_ylabel("Relative Speedup")
    ax.legend()

    fig.tight_layout()
    plt.savefig("../results/fastrlvar_sketch.png", dpi=300, bbox_inches="tight")

    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON file with metrics. "
             "If omitted or partially specified, simulated Gaussians are used.",
    )
    parser.add_argument(
        "--n_sim",
        type=int,
        default=400,
        help="Number of simulated samples per method (if needed).",
    )
    args = parser.parse_args()

    if args.json is not None and os.path.isfile(args.json):
        loaded = load_json_metrics(args.json)
    else:
        loaded = {}

    data = merge_with_simulation(loaded, n_sim=args.n_sim)
    plot_distributions(data)


if __name__ == "__main__":
    main()
