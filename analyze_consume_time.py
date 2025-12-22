from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


def load_values(path: Path) -> list[float]:
    """Parse `hash: value` lines and collect numeric values."""
    values: list[float] = []
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        _, raw_value = line.split(":", 1)
        try:
            values.append(float(raw_value.strip()))
        except ValueError:
            continue
    return values


def summarize(values: Iterable[float]) -> tuple[int, float, float]:
    data = list(values)
    if not data:
        raise ValueError("No numeric values found.")
    return len(data), mean(data), pstdev(data)


def find_targets(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.glob("c*.txt") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze consume time files (hash: time) and report count, mean, and"
            " population standard deviation."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "Infinity_v2",
        help="Directory containing consume*.txt files, or a single consume*.txt file.",
    )
    args = parser.parse_args()

    targets = find_targets(args.path)
    if not targets:
        raise SystemExit(f"No consume*.txt files found in {args.path}")

    for file_path in targets:
        values = load_values(file_path)
        if not values:
            print(f"{file_path}: no numeric values found, skipping.")
            continue
        count, avg, std = summarize(values)
        print(f"{file_path}")
        print(f"  count: {count}")
        print(f"  mean:  {avg:.6f}")
        print(f"  std:   {std:.6f}")


if __name__ == "__main__":
    main()
