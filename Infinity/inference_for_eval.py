import argparse
import gc
import json
import os
import random
import re
from contextlib import contextmanager

import cv2
import numpy as np
import torch

from tools.run_infinity import *

model_path = '/nfs/home/tensore/pretrained/Infinity/infinity_2b_reg.pth'
vae_path = '/nfs/home/tensore/pretrained/Infinity/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/nfs/home/tensore/pretrained/Infinity/models--google--flan-t5-x'

DEFAULT_META_PATH = "/nfs/home/tensore/RL/FastRLVAR/Infinity/infinity/dataset/meta_data.json"
DEFAULT_CATEGORY = "people"
pruning_scales = "48:1.0,64:1.0"


def parse_pruning_scales(spec: str):
    if not spec:
        return None
    result = {}
    for entry in spec.split(','):
        entry = entry.strip()
        if not entry:
            continue
        scale_str, ratio_str = entry.split(':')
        result[int(scale_str)] = float(ratio_str)
    return result or None


parsed_prune_scales = None

model_args = argparse.Namespace(
    pn='1M',  # 1M, 0.60M, 0.25M, 0.06M
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    prune_scales=pruning_scales,
    prune_scale_list=parsed_prune_scales,
)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Generate multiple Infinity images from the meta_data prompts.")
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate in one run.",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="Category to filter prompts from meta_data.json.",
    )
    parser.add_argument(
        "--meta-data",
        dest="meta_data_path",
        default=DEFAULT_META_PATH,
        help="Path to the source meta_data.json file.",
    )
    parser.add_argument(
        "--base-output-dir",
        default="results",
        help="Base directory where run folders (run1, run2, ...) are stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for prompt selection and image generation.",
    )
    return parser.parse_args()


def get_scale_schedule(h_div_w: float):
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    schedule = dynamic_resolution_h_w[h_div_w_template_][model_args.pn]['scales']
    return [(1, h, w) for (_, h, w) in schedule]


def prepare_run_dir(base_output_dir: str) -> str:
    os.makedirs(base_output_dir, exist_ok=True)
    max_run_idx = 0
    for name in os.listdir(base_output_dir):
        match = re.match(r"run(\d+)$", name)
        if match:
            max_run_idx = max(max_run_idx, int(match.group(1)))
    run_dir = os.path.join(base_output_dir, f"run{max_run_idx + 1}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def load_candidates(meta_data_path: str, category: str):
    with open(meta_data_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    category_lower = category.lower()
    return {
        img_id: entry
        for img_id, entry in meta_data.items()
        if category_lower in str(entry.get("category", "")).lower()
    }


def select_prompts(candidates: dict, num_images: int, rng: random.Random):
    if num_images <= 0:
        raise ValueError("Number of images must be positive.")
    if len(candidates) < num_images:
        raise ValueError(f"Requested {num_images} images but only found {len(candidates)} matching prompts.")
    selected_ids = rng.sample(list(candidates.keys()), num_images)
    prompts = {img_id: candidates[img_id]["prompt"] for img_id in selected_ids}
    selected_metadata = {img_id: candidates[img_id] for img_id in selected_ids}
    return prompts, selected_metadata


def init_models():
    text_tokenizer, text_encoder = load_tokenizer(t5_path=model_args.text_encoder_ckpt)
    vae = load_visual_tokenizer(model_args)
    infinity = load_transformer(vae, model_args)
    return text_tokenizer, text_encoder, vae, infinity


# memory consumption evaluation
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')


def save_used_metadata(run_dir: str, used_metadata: dict):
    meta_output_path = os.path.join(run_dir, "meta_data.json")
    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(used_metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote metadata for generated prompts to {os.path.abspath(meta_output_path)}")


def main():
    cli_args = parse_cli_args()
    rng = random.Random(cli_args.seed)

    candidates = load_candidates(cli_args.meta_data_path, cli_args.category)
    prompts, selected_metadata = select_prompts(candidates, cli_args.num_images, rng)

    run_output_dir = prepare_run_dir(cli_args.base_output_dir)
    print(f"Saving outputs to {os.path.abspath(run_output_dir)}")

    text_tokenizer, text_encoder, vae, infinity = init_models()

    cfg_value = 4
    tau_value = 0.5
    h_div_w = 1 / 1  # aspect ratio, height:width
    enable_positive_prompt = 0
    scale_schedule = get_scale_schedule(h_div_w)
    print(scale_schedule)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    used_metadata = {}

    with torch.inference_mode():
        with measure_peak_memory():
            for idx, (img_id, prompt_text) in enumerate(prompts.items(), start=1):
                print(f"\n=== Generating for prompt '{img_id}': '{prompt_text}' ===")
                start_event.record()

                generated_image = gen_one_img(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt_text,
                    g_seed=cli_args.seed + idx,   # different seed per prompt
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=[cfg_value] * len(scale_schedule),
                    tau_list=[tau_value] * len(scale_schedule),
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[model_args.cfg_insertion_layer],
                    vae_type=model_args.vae_type,
                    sampling_per_bits=model_args.sampling_per_bits,
                    enable_positive_prompt=enable_positive_prompt,
                    save_intermediate_results=False,
                    save_dir=run_output_dir,
                    per_scale_infer=True,
                )

                img_path = os.path.join(run_output_dir, f"{img_id}.jpg")
                cv2.imwrite(img_path, generated_image.cpu().numpy())
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0
                print(f"Saved image for '{img_id}' to {os.path.abspath(img_path)} (took {elapsed:.2f}s)")

                used_metadata[img_id] = {
                    "prompt": prompt_text,
                    "category": selected_metadata[img_id].get("category", cli_args.category),
                }

    save_used_metadata(run_output_dir, used_metadata)


if __name__ == "__main__":
    main()
