import os
from pathlib import Path

# Limit threading to avoid shared memory issues in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_runtime_map(path: Path) -> dict:
    """Load runtimes from a simple hash: value text file."""
    mapping: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            mapping[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return mapping


def main() -> None:
    project_root = Path(__file__).resolve().parent
    runtime_path = project_root / "Infinity_v2" / "consume_time_v2_ppo.txt"
    scores_path = project_root / "Infinity" / "hpsv3_comparison_results.csv"

    runtime_map = load_runtime_map(runtime_path)
    scores_df = pd.read_csv(scores_path)
    scores_df["hash"] = scores_df["Filename"].apply(lambda x: Path(x).stem)
    scores_df["runtime"] = scores_df["hash"].map(runtime_map)

    matched = scores_df.dropna(subset=["runtime"])
    if matched.empty:
        raise SystemExit("No matching runtime entries found between the two files.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(matched["runtime"], matched["Score_RL"], alpha=0.7, edgecolor="none")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("RL Score")
    ax.set_title("Runtime vs RL Score")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = project_root / "Infinity" / "runtime_vs_rl_scatter.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved scatter plot to {output_path}")


if __name__ == "__main__":
    main()
