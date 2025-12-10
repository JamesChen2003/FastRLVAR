"""
Minimal HPSv3 scorer without the hpsv3 wrapper package.

The original sample failed with:
  ModuleNotFoundError: No module named 'transformers.modeling_layers'
because the bundled hpsv3/peft build expects a newer transformers version.
This script uses the HF model directly, avoiding that dependency issue.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

MODEL_ID = "MizzenAI/HPSv3"


def load_model(device: str):
    # HPSv3 is a vision-language reward model; bfloat16/float16 is sufficient.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # qwen2_5_vl config needs trust_remote_code to register custom arch.
    # Some transformers versions drop trust_remote_code when only passed to AutoModel,
    # so build the config explicitly first.
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, config=config, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, processor


def load_images(paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            images.append(Image.open(path).convert("RGB"))
        except OSError as exc:
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc
    return images


@torch.inference_mode()
def score_images(model, processor, images: List[Image.Image], prompts: List[str], device: str) -> List[float]:
    if len(images) != len(prompts):
        raise ValueError(f"Number of images ({len(images)}) != number of prompts ({len(prompts)})")

    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model outputs do not contain 'logits'; cannot compute reward.")
    scores = logits.squeeze()
    if scores.ndim == 0:
        scores = scores.unsqueeze(0)
    return [float(x) for x in scores.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick HPSv3 scoring demo")
    parser.add_argument("--images", nargs="+", default=["0a.jpg", "0a2.jpg"], help="Image paths to score")
    parser.add_argument("--prompts", nargs="+", default=["German soliders capturing Moscow in 1941"] * 2, help="Prompts, one per image")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; run with --device cpu", file=sys.stderr)
        sys.exit(1)

    images = load_images(args.images)
    model, processor = load_model(args.device)
    scores = score_images(model, processor, images, args.prompts, args.device)
    print(f"Image scores: {scores}")


if __name__ == "__main__":
    main()
