import os
import sys
import random
import torch
import cv2
import numpy as np
import gc
import json
import argparse
from cleanfid import fid
from clip_score import clip_score
from tqdm import tqdm

# Ensure we can import from the project root (so that `tools.run_infinity` works
# when this script is run from within the `evaluation` directory).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_infinity import *
 
model_path = '/nfs/home/tensore/pretrained/Infinity/infinity_2b_reg.pth'
vae_path   = '/nfs/home/tensore/pretrained/Infinity/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/nfs/home/tensore/pretrained/Infinity/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001'

args=argparse.Namespace(
    pn='1M', # 1M, 0.60M, 0.25M, 0.06M
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
    save_file='tmp.jpg'
)

# Set device for Infinity pipeline (all models must be on same GPU)
infinity_device = 'cuda:0'

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt, device=infinity_device)
# load vae
vae = load_visual_tokenizer(args, device=infinity_device)
# load infinity
infinity = load_transformer(vae, args, device=infinity_device)

# 16GB memo
cfg = 4
tau = 1.0
h_div_w = 1/1 # aspect ratio, height:width
# seed = random.randint(0, 10000)
seed = 42
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]


with open("/nfs/home/tensore/RL/FastRLVAR/Infinity_v3/evaluation/MJHQ30K/meta_data.json") as f:
    meta_data = json.load(f)

save_root_dir = "/nfs/home/tensore/RL/FastRLVAR/Infinity_v3/evaluation/MJHQ30K/output/"
os.makedirs(save_root_dir,exist_ok=True)

num_sampe = 0
for img_id,data in tqdm(meta_data.items()):
    prompt = data['prompt']
    category = data['category']
    if 'people' not in category:
        continue
    with torch.inference_mode():
        generated_image = gen_one_img(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=[cfg] * len(scale_schedule),
            tau_list=[tau] * len(scale_schedule),
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=enable_positive_prompt,
        )
        os.makedirs(os.path.join(save_root_dir, category), exist_ok=True)
        cv2.imwrite(os.path.join(save_root_dir, category, f"{img_id}.png"), generated_image.cpu().numpy())


# test fid
ref_dir = "/nfs/home/tensore/RL/FastRLVAR/Infinity_v3/evaluation/MJHQ30K/mjhq30k_imgs/people"
gen_dir = save_root_dir
fid_score = fid.compute_fid(ref_dir,gen_dir)
print(f'FID score:{fid_score}')

# clip_score

import torch
import clip
from PIL import Image

# Load the CLIP model on GPU 1
clip_device = "cuda:2" if torch.cuda.device_count() > 1 else "cuda:0"
model, preprocess = clip.load("ViT-L/14", device=clip_device)
def compute_clip_score(image_path, text):
    # Load and preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(clip_device)

    # Tokenize the text
    text = clip.tokenize([text],truncate=True).to(clip_device)

    # Compute the feature vectors
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize the feature vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute the cosine similarity
    similarity = (image_features @ text_features.T).item()
    return similarity
import json

# Load


import os

generated_dir = save_root_dir
total_score = 0
count = 0


for root, _, files in os.walk(generated_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure you're processing image files
            image_path = os.path.join(root, file)
            image_id = os.path.splitext(file)[0]
            if image_id in meta_data:
                prompt = meta_data[image_id]["prompt"]
                score = compute_clip_score(image_path, prompt)
                total_score += score
                count += 1
            else:
                print(f"No prompt found for image {image_id}")

if count > 0:
    average_clip_score = total_score / count
    print(f"Average CLIP Score: {average_clip_score}")
else:
    print("No images were processed.")








