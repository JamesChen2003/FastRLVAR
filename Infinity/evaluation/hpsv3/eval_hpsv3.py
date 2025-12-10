#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Allow importing the local hpsv3 package when running from this repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
HPSV3_ROOT = REPO_ROOT / "HPSv3"
if HPSV3_ROOT.is_dir() and str(HPSV3_ROOT) not in sys.path:
    sys.path.append(str(HPSV3_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPSv3 side-by-side evaluation (Original vs Fastvar vs RL)")
    parser.add_argument("--original", required=True, help="Folder with original images and meta_data.json")
    parser.add_argument("--fastvar", required=True, help="Folder with fastvar images")
    parser.add_argument("--RL", required=True, help="Folder with RL images")
    return parser.parse_args()


def list_images(folder: str) -> List[str]:
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])


def strip_ext(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_metadata_map(original_dir: str) -> Dict[str, str]:
    meta_path = os.path.join(original_dir, "meta_data.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta_data.json not found in {original_dir}")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        # Fallback: JSONL
        raw = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    return build_prompt_index(raw)


def add_mapping(mapping: Dict[str, str], keys: Iterable[str], prompt: str) -> None:
    for k in keys:
        if k:
            mapping[strip_ext(str(k))] = prompt


def build_prompt_index(raw) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                mapping[strip_ext(k)] = v
            elif isinstance(v, dict) and "prompt" in v:
                mapping[strip_ext(k)] = v["prompt"]
        return mapping

    if not isinstance(raw, list):
        return mapping

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        prompt = entry.get("prompt")
        if not prompt:
            continue
        candidates: List[str] = []
        for key in ("id", "file", "filename", "file_name", "name"):
            if key in entry:
                candidates.append(entry[key])
        for path_key in ("gen_image_paths", "image_paths", "paths"):
            for p in entry.get(path_key, []) or []:
                candidates.append(p)
        add_mapping(mapping, candidates or [None], prompt)
    return mapping


def load_hpsv3(device: str):
    from hpsv3 import HPSv3RewardInferencer

    return HPSv3RewardInferencer(device=device)


@torch.inference_mode()
def score_image(model, image_path: str, prompt: str) -> float:
    rewards = model.reward(prompts=[prompt], image_paths=[image_path])
    if isinstance(rewards, torch.Tensor):
        return float(rewards.view(-1)[0].item())
    if not rewards:
        raise RuntimeError("HPSv3 returned no rewards")
    first = rewards[0]
    if isinstance(first, torch.Tensor):
        return float(first.view(-1)[0].item())
    return float(first[0] if isinstance(first, (list, tuple)) else first)


def main() -> None:
    args = parse_args()

    for folder in (args.original, args.fastvar, args.RL):
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Input directory not found: {folder}")

    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        raise RuntimeError("CUDA device not available; HPSv3 requires GPU for this script.")
    model = load_hpsv3(device)

    prompt_map = load_metadata_map(args.original)
    orig_files = set(list_images(args.original))
    fast_files = set(list_images(args.fastvar))
    rl_files = set(list_images(args.RL))
    common_files = sorted(orig_files & fast_files & rl_files)
    if not common_files:
        raise RuntimeError("No matching filenames across the three folders.")

    rows = []
    for fname in tqdm(common_files, desc="Scoring with HPSv3"):
        prompt = prompt_map.get(strip_ext(fname))
        if not prompt:
            print(f"[WARN] Prompt missing for {fname}, skipping.", file=sys.stderr)
            continue

        paths = {
            "Score_Original": os.path.join(args.original, fname),
            "Score_Fastvar": os.path.join(args.fastvar, fname),
            "Score_RL": os.path.join(args.RL, fname),
        }
        if any(not os.path.isfile(path) for path in paths.values()):
            print(f"[WARN] Skipping {fname} due to missing/corrupted image.", file=sys.stderr)
            continue

        try:
            scores = {key: score_image(model, path, prompt) for key, path in paths.items()}
        except Exception as exc:
            print(f"[WARN] Scoring failed for {fname}: {exc}", file=sys.stderr)
            continue

        rows.append({
            "Filename": fname,
            "Text_Prompt": prompt,
            **scores,
        })

    if not rows:
        raise RuntimeError("No rows to write; all items were skipped.")

    output_csv = "hpsv3_comparison_results.csv"
    pd.DataFrame(rows, columns=["Filename", "Text_Prompt", "Score_Original", "Score_Fastvar", "Score_RL"]).to_csv(
        output_csv, index=False
    )
    print(f"Saved results to {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    main()
