import random
import os
import json
import torch
import cv2
import numpy as np
from tools.run_infinity import *
from contextlib import contextmanager
import gc
import argparse

model_path='/nfs/home/tensore/pretrained/Infinity/infinity_2b_reg.pth'
vae_path=  '/nfs/home/tensore/pretrained/Infinity/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/nfs/home/tensore/pretrained/Infinity/models--google--flan-t5-x'

# ------------ multi-prompt definition (name -> text) -------------
# prompts = {
#     # "cat":       "A cute cat on the grass.",
#     "city":      "A futuristic city skyline at night.",
#     "astronaut": "An astronaut painting on the moon.",
#     # "woman":     "An anime-style portrait of a woman.",
#     # "man":       "A detailed photo-realistic image of a man."
# }
with open("/nfs/home/tensore/RL/FastRLVAR/Infinity/infinity/dataset/meta_data.json") as f:
    meta_data = json.load(f)
# prompts = {
#     # "cat":       "A cute cat on the grass.",
#     "city":      "A futuristic city skyline at night.",
#     "astronaut": "An astronaut painting on the moon.",
#     # "woman":     "An anime-style portrait of a woman.",
#     # "man":       "A detailed photo-realistic image of a man."
# }
# with open("/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/evaluation/MJHQ30K/meta_data.json") as f:
#     meta_data = json.load(f)

# prompts = {}

# for img_id, data in meta_data.items():
#     if 'people' in data['category']:
#         prompts[img_id] = data['prompt']
#     if len(prompts) >= 5:
#         break

# Base results dir; each prompt gets its own subfolder
base_output_dir = "results"
os.makedirs(base_output_dir, exist_ok=True)

# si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64] 13 scales
# pruning_scales = "2:1.0,4:1.0,6:1.0,8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
# pruning_scales = "8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
# pruning_scales = "20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
pruning_scales = "32:0.4,40:0.5,48:1.0,64:1.0"
# pruning_scales = "64:1.0"


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


# parsed_prune_scales = parse_pruning_scales(pruning_scales)
parsed_prune_scales = None

args = argparse.Namespace(
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

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

cfg_value = 4
tau_value = 0.5
h_div_w = 1 / 1  # aspect ratio, height:width
seed = 42
enable_positive_prompt = 0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
print(scale_schedule)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


# memory consumption evaluation
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')


# global metadata for all prompts (optional, stored in base_output_dir)
global_prompt_records = []

with torch.inference_mode():
    with measure_peak_memory():
        # enumerate prompts with index for seeding
        for idx, (name, prompt_text) in enumerate(prompts.items(), start=1):
            # per-prompt output dir and metadata path
            root_output_dir = os.path.join(base_output_dir, f"gen_images_{name}")
            os.makedirs(root_output_dir, exist_ok=True)
            metadata_path = os.path.join(root_output_dir, "config.json")

            print(f"\n=== Generating for prompt '{name}': '{prompt_text}' ===")
            start_event.record()

            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt_text,
                g_seed=seed + idx,   # different seed per prompt
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=[cfg_value] * len(scale_schedule),
                tau_list=[tau_value] * len(scale_schedule),
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=enable_positive_prompt,
                save_intermediate_results=False,
                save_dir=root_output_dir,   # <<< all intermediate images & fourier here
                per_scale_infer=True,
            )

            # save main generated image as 1.jpg in that folder
            img_path = os.path.join(root_output_dir, "1.jpg")
            cv2.imwrite(img_path, generated_image.cpu().numpy())
            print(f"Saved image for '{name}' to {os.path.abspath(img_path)}")

            # per-prompt metadata
            prompt_record = {
                "prompt_name": name,
                "prompt": prompt_text,
                "image_path": img_path,
                "image_number": 1,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pruning_scales": parsed_prune_scales or {},
                        "pruning_scales_raw": pruning_scales,
                        "prompt": prompt_record,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved prompt metadata to {os.path.abspath(metadata_path)}")

            # add to global list
            global_prompt_records.append(prompt_record)

# # optional global config across all prompts
# global_metadata_path = os.path.join(base_output_dir, "config_all.json")
# with open(global_metadata_path, "w", encoding="utf-8") as f:
#     json.dump(
#         {
#             "pruning_scales": parsed_prune_scales or {},
#             "pruning_scales_raw": pruning_scales,
#             "prompts": global_prompt_records,
#         },
#         f,
#         ensure_ascii=False,
#         indent=2,
#     )
# print(f"\nSaved global prompt metadata to {os.path.abspath(global_metadata_path)}")
