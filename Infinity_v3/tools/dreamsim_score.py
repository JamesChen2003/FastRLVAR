from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from PIL import Image
from dreamsim import dreamsim

# Supported DreamSim backbones (informational)
dreamsim_type_list = [
    "ensemble",
    "dino_vitb16",
    "clip_vitb32",
    "open_clip_vitb32",
    "dinov2_vitb14",
    "synclr_vitb16",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL: Optional[torch.nn.Module] = None
_PREPROCESS = None
_MODEL_CFG: Optional[Tuple[str, bool, str]] = None  # (dreamsim_type, use_patch_model, cache_dir)


def _fmt_mb(x_bytes: int) -> str:
    return f"{x_bytes / (1024 ** 2):.2f} MB"


def report_vram(tag: str) -> None:
    """Print lightweight CUDA VRAM stats for the current process."""
    if not torch.cuda.is_available():
        print(f"[VRAM] {tag}: CUDA not available")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(
        f"[VRAM] {tag}: allocated={_fmt_mb(alloc)} reserved={_fmt_mb(reserved)} peak_allocated={_fmt_mb(peak)}"
    )


def _get_model(
    dreamsim_type: str = "ensemble",
    use_patch_model: bool = False,
    cache_dir: str = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/checkpoint",
    report_memory: bool = False,
):
    """
    Lazy-load DreamSim model so this module can be imported cheaply (important for VAREnv).
    Returns (model, preprocess).
    """
    global _MODEL, _PREPROCESS, _MODEL_CFG
    cfg = (dreamsim_type, use_patch_model, cache_dir)
    if _MODEL is not None and _PREPROCESS is not None and _MODEL_CFG == cfg:
        return _MODEL, _PREPROCESS

    if report_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        report_vram("dreamsim startup (before model load)")

    model, preprocess = dreamsim(
        pretrained=True,
        device=device,
        cache_dir=cache_dir,
        dreamsim_type=dreamsim_type,
        use_patch_model=use_patch_model,
    )
    _MODEL, _PREPROCESS, _MODEL_CFG = model, preprocess, cfg

    if report_memory:
        report_vram("dreamsim after model load")
    return _MODEL, _PREPROCESS


def get_dreamsim_similarity(
    image1: Image.Image,
    image2: Image.Image,
    dreamsim_type: str = "ensemble",
    use_patch_model: bool = False,
    cache_dir: str = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/checkpoint",
    report_memory: bool = False,
) -> float:
    """
    Return DreamSim similarity in [0, 1] (implemented as 1 - distance).

    This is designed to be callable from the RL env (VAREnv).
    """
    model, preprocess = _get_model(
        dreamsim_type=dreamsim_type,
        use_patch_model=use_patch_model,
        cache_dir=cache_dir,
        report_memory=report_memory,
    )

    # Robustly handle alpha channels
    if image1.mode != "RGB":
        image1 = image1.convert("RGB")
    if image2.mode != "RGB":
        image2 = image2.convert("RGB")

    x1 = preprocess(image1).to(device)
    x2 = preprocess(image2).to(device)

    if report_memory:
        report_vram("dreamsim after inputs to device")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            report_vram("dreamsim before inference")

    with torch.no_grad():
        # DreamSim returns a distance; higher distance => less similar.
        dist = model(x1, x2)

    if report_memory:
        report_vram("dreamsim after inference")

    sim = 1.0 - float(dist.detach().cpu().item())
    # Clamp for safety
    return float(max(min(sim, 1.0), 0.0))


def quality_func(x, a=0.88, b=0.98):
    """
    Linear mapping of DINOv3 cosine similarity:
        x in [a, b] -> [0, 1]
        x < a -> 0
        x > b -> 1
    """
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return float((x - a) / (b - a))

if __name__ == "__main__":
    # Interactive local test: press Enter to reload images from disk and re-score.
    # Type 'q' then Enter to quit.
    base_dir = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/training_tmp_results"
    image1_path = os.path.join(base_dir, "golden_img_12.png")
    image2_path = os.path.join(base_dir, "pruned_img_12.png")

    print(f"Reference: {image1_path}")
    print(f"Candidate: {image2_path}")
    print("Press Enter to recompute. Type 'q' then Enter to quit.")

    # Import DINOv3 scoring once so the model loads once (not every keypress).
    # NOTE: tools/dinov3_score.py loads the HF model at import time.
    from dinov3_score import get_dinov3_similarity as dinov3_get_similarity
    from dinov3_score import quality_func as dinov3_quality_func

    while True:
        cmd = input("> ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            break

        # Reload images every iteration (so you can overwrite files externally).
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        sim = get_dreamsim_similarity(img1, img2, report_memory=True)
        print(f"DreamSim similarity: {sim:.6f}")
        print(f"DreamSim quality score: {quality_func(sim, 0.935, 0.98):.6f}")

        dinov3_sim = float(dinov3_get_similarity(img1, img2, mode="cls"))
        print(f"DINOv3 (CLS) similarity: {dinov3_sim:.6f}")
        print(f"DINOv3 quality score: {dinov3_quality_func(dinov3_sim, 0.91, 0.98):.6f}")

