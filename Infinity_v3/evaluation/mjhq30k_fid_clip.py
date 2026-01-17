import os
import json
import argparse
import tempfile
import shutil
from collections import Counter

from cleanfid import fid
import torch
import clip
from PIL import Image
from tqdm import tqdm


def _normalize_categories(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [c for c in value if isinstance(c, str)]
    return []


def _resolve_category(meta_data, requested_category):
    if requested_category:
        return requested_category
    counts = Counter()
    for data in meta_data.values():
        for c in _normalize_categories(data.get("category")):
            counts[c] += 1
    categories = [c for c in counts.keys() if c]
    if not categories:
        return None
    if len(categories) == 1:
        return categories[0]
    raise ValueError(
        "meta_data has multiple categories. "
        "Please pass --category. Found: " + ", ".join(sorted(categories))
    )


def _index_images(run_dir):
    run_dir = os.path.abspath(run_dir)
    image_map = {}
    for name in os.listdir(run_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            image_map[stem] = os.path.join(run_dir, name)
    return image_map


def _resolve_clip_device(requested):
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _category_matches(data, category):
    cats = _normalize_categories(data.get("category"))
    return category in cats


def compute_clip_score(model, preprocess, device, image_path, text):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()
    return similarity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FID and CLIP score from an existing results_eval/runX directory."
    )
    parser.add_argument("--run-dir", required=True, help="Path to results_eval/runX")
    parser.add_argument(
        "--ref-root",
        default="Infinity_v3/evaluation/MJHQ30K/mjhq30k_imgs",
        help="Root directory for reference images (category subfolders inside).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Category name for FID (e.g. people/food/landscape). "
             "If omitted, will infer when meta_data has only one category.",
    )
    parser.add_argument(
        "--fid-device",
        default="cuda:0",
        help="Device for FID feature extractor (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=32,
        help="Batch size for FID feature extraction.",
    )
    parser.add_argument(
        "--fid-num-workers",
        type=int,
        default=4,
        help="DataLoader workers for FID feature extraction.",
    )
    parser.add_argument(
        "--fid-use-dataparallel",
        action="store_true",
        help="Enable DataParallel for FID feature extractor.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-L/14",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--clip-device",
        default="auto",
        help="CLIP device (auto/cpu/cuda:0/cuda:1...).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    meta_path = os.path.join(run_dir, "meta_data.json")
    if not os.path.isfile(meta_path):
        parent_dir = os.path.dirname(run_dir)
        parent_meta = os.path.join(parent_dir, "meta_data.json")
        if os.path.isfile(parent_meta):
            meta_path = parent_meta
        else:
            raise FileNotFoundError(f"Missing meta_data.json in {run_dir} or {parent_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    category = _resolve_category(meta_data, args.category)
    if category is None:
        raise ValueError("No category found in meta_data; please pass --category.")

    ref_dir = os.path.join(args.ref_root, category)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference dir not found: {ref_dir}")

    image_map = _index_images(run_dir)
    filtered_ids = [img_id for img_id, data in meta_data.items() if _category_matches(data, category)]
    if not filtered_ids:
        raise ValueError(f"No images found for category '{category}' in meta_data.")

    missing_images = 0
    linked_images = 0
    with tempfile.TemporaryDirectory(prefix="fid_gen_") as tmp_dir:
        for image_id in filtered_ids:
            image_path = image_map.get(image_id)
            if not image_path:
                missing_images += 1
                continue
            target_path = os.path.join(tmp_dir, os.path.basename(image_path))
            try:
                os.symlink(image_path, target_path)
            except OSError:
                shutil.copy2(image_path, target_path)
            linked_images += 1

        if linked_images == 0:
            raise ValueError("No images found for FID evaluation in the run directory.")
        fid_score = fid.compute_fid(
            ref_dir,
            tmp_dir,
            device=args.fid_device,
            batch_size=args.fid_batch_size,
            num_workers=args.fid_num_workers,
            use_dataparallel=args.fid_use_dataparallel,
        )
    print(f"FID score ({category}): {fid_score}")

    clip_device = _resolve_clip_device(args.clip_device)
    model, preprocess = clip.load(args.clip_model, device=clip_device)

    total_score = 0.0
    count = 0
    missing = missing_images

    for image_id in tqdm(filtered_ids, desc="CLIP"):
        data = meta_data[image_id]
        prompt = data.get("prompt")
        image_path = image_map.get(image_id)
        if not prompt or not image_path:
            missing += 1
            continue
        score = compute_clip_score(model, preprocess, clip_device, image_path, prompt)
        total_score += score
        count += 1

    if count > 0:
        average_clip_score = total_score / count
        print(f"Average CLIP Score: {average_clip_score}")
    else:
        print("No images were processed for CLIP score.")

    if missing:
        print(f"Missing prompts or images: {missing}")


if __name__ == "__main__":
    main()



