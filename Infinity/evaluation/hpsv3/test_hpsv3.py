"""
Score every id_policy.jpg (or id-variant.jpg) in a folder using HPSv3 and print
`filename : score`. Prompts are read from meta_data.json using the id portion of
the filename. Uses the same HPSv3 wrapper approach as eval_hpsv3.py to avoid
transformers/qwen2_5_vl version issues.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

MODEL_ID = "MizzenAI/HPSv3"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Allow importing the local hpsv3 package when running from this repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
HPSV3_ROOT = REPO_ROOT / "HPSv3"
if HPSV3_ROOT.is_dir() and str(HPSV3_ROOT) not in sys.path:
    sys.path.append(str(HPSV3_ROOT))


def load_model(device: str):
    from hpsv3 import HPSv3RewardInferencer

    # HPSv3RewardInferencer handles the underlying HF model; GPU is expected.
    return HPSv3RewardInferencer(device=device)


def build_prompt_index(meta_path: Path) -> Dict[str, str]:
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta_data.json not found: {meta_path}")
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        raw = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    mapping: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                mapping[Path(k).stem] = v
            elif isinstance(v, dict) and "prompt" in v:
                mapping[Path(k).stem] = v["prompt"]
        return mapping

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            prompt = entry.get("prompt")
            if not prompt:
                continue
            for key in ("id", "file", "filename", "file_name", "name"):
                val = entry.get(key)
                if val:
                    mapping[Path(str(val)).stem] = prompt
            for path_key in ("gen_image_paths", "image_paths", "paths"):
                for p in entry.get(path_key, []) or []:
                    mapping[Path(str(p)).stem] = prompt
    return mapping


def load_consume_times(path: Path) -> Dict[str, float]:
    times: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            img_id, value = line.split(":", 1)
            img_id = img_id.strip()
            value = value.strip()
            if not img_id or not value:
                continue
            try:
                times[img_id] = float(value)
            except ValueError:
                continue
    return times


def collect_images(image_dir: Path, prompt_map: Dict[str, str]) -> Tuple[List[Path], List[str], List[str], List[str]]:
    img_paths: List[Path] = []
    prompts: List[str] = []
    names: List[str] = []
    ids: List[str] = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        if "_" in stem:
            img_id = stem.split("_", 1)[0]
        elif "-" in stem:
            img_id = stem.split("-", 1)[0]
        else:
            print(f"[WARN] Skipping {img_path.name}: expected id_policy.jpg or id-variant.jpg format.", file=sys.stderr)
            continue
        prompt = prompt_map.get(img_id)
        if not prompt:
            print(f"[WARN] Prompt missing for id {img_id} ({img_path.name}), skipping.", file=sys.stderr)
            continue
        img_paths.append(img_path)
        prompts.append(prompt)
        names.append(img_path.name)
        ids.append(img_id)

    if not img_paths:
        raise RuntimeError(f"No valid *_policy images found in {image_dir}")
    return img_paths, prompts, names, ids


@torch.inference_mode()
def score_image(model, image_path: Path, prompt: str) -> float:
    rewards = model.reward(prompts=[prompt], image_paths=[str(image_path)])
    if isinstance(rewards, torch.Tensor):
        return float(rewards.view(-1)[0].item())
    if not rewards:
        raise RuntimeError("HPSv3 returned no rewards")
    first = rewards[0]
    if isinstance(first, torch.Tensor):
        return float(first.view(-1)[0].item())
    return float(first[0] if isinstance(first, (list, tuple)) else first)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all id_policy.jpg images in a folder with HPSv3")
    default_dir = Path(__file__).parent
    parser.add_argument("--image-dir", type=Path, default=default_dir, help="Folder containing *_policy.jpg and meta_data.json")
    parser.add_argument("--meta", type=Path, default=None, help="Path to meta_data.json (defaults to image-dir/meta_data.json)")
    parser.add_argument("--consume-time", type=Path, default=None, help="Optional id -> time mapping (defaults to consume_time_ppo5.txt in image-dir when present)")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; HPSv3 wrapper expects GPU. Run with --device cpu only if your build supports it.", file=sys.stderr)
        sys.exit(1)

    meta_path = args.meta or (args.image_dir / "meta_data.json")
    prompt_map = build_prompt_index(meta_path)
    image_paths, prompts, names, img_ids = collect_images(args.image_dir, prompt_map)

    consume_time_path: Optional[Path] = args.consume_time
    if consume_time_path is None:
        default_consume = args.image_dir / "consume_time_ppo5.txt"
        if default_consume.is_file():
            consume_time_path = default_consume
    consume_times: Dict[str, float] = {}
    if consume_time_path:
        if not consume_time_path.is_file():
            raise FileNotFoundError(f"consume_time file not found: {consume_time_path}")
        consume_times = load_consume_times(consume_time_path)

    model = load_model(args.device)
    scores = [score_image(model, img_path, prompt) for img_path, prompt in zip(image_paths, prompts)]

    for name, img_id, score in zip(names, img_ids, scores):
        if consume_times:
            consume_time = consume_times.get(img_id)
            time_str = str(consume_time) if consume_time is not None else "N/A"
            print(f"{name} : {time_str} : {score:.6f}")
        else:
            print(f"{name} : {score:.6f}")


if __name__ == "__main__":
    main()
