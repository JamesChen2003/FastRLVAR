import warnings
import time
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from types import MethodType

from gymnasium.envs.registration import register

# import wandb
# from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import random
import os
import json
import torch
import cv2
import numpy as np
import re
from tools.run_infinity import *
from contextlib import contextmanager
import gc
import argparse

from tools.VAREnv import VAREnv


class FastVARCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128, width: int = 32):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        code_channels = c - 1

        self.trunk = nn.Sequential(
            nn.Conv2d(code_channels, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8 * 8, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

        self.width = width
        self._features_dim = features_dim

    def forward(self, obs):
        obs = obs.float()
        codes = obs[:, :-1]
        scale_plane = obs[:, -1]

        cnn_map = self.trunk(codes)
        cnn_map = self.pool(cnn_map)

        scale_scalar = scale_plane.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        scale_emb = self.scale_mlp(scale_scalar)

        fused_map = cnn_map + scale_emb.view(-1, self.width, 1, 1)
        return self.cnn_head(fused_map)


warnings.filterwarnings("ignore")
register(
    id='var-v0',
    entry_point='tools:VAREnv'
)

my_config = {
    "run_id": "PPO5",
    "algorithm": PPO,
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model/PPO",
    "num_train_envs": 1,
    "epoch_num": 50,
    "eval_episode_num": 5,
    "prompt_pool_size": 1000,
    "iterations_per_prompt": 1,
}

my_config["timesteps_per_epoch"] = my_config["prompt_pool_size"] * 13 * my_config["iterations_per_prompt"]


def make_env(infinity, vae, scale_schedule, text_tokenizer, text_encoder, prompt):
    # env = gym.make('var-v0')
    env = VAREnv(
        infinity,
        vae, 
        scale_schedule, 
        text_tokenizer, 
        text_encoder, 
        prompt)
    env = Monitor(
        env,
        info_keywords=(
            "quality_reward",
            "speed_reward",
            "quality_score",
            "speed_score",
            "total_reward",
            "psnr",
            "ms_ssim",
            "lpips",
            "prune_ratio",
        ),
    )
    return env

def eval(env, model, eval_episode_num):
    """
    Evaluate the model on the vectorized FastVAR environment.
    Returns:
        avg_return: average episode return
        avg_quality: average per-step quality_score
        avg_speed: average per-step speed_score
    """
    avg_return = 0.0
    avg_quality = 0.0
    avg_speed = 0.0

    for _ in range(eval_episode_num):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_quality = 0.0
        ep_speed = 0.0
        steps = 0

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            r = float(rewards[0])
            info = infos[0]

            ep_return += r
            ep_quality += info.get("quality_score", 0.0)
            ep_speed += info.get("speed_score", 0.0)
            steps += 1

            done = bool(dones[0])

        avg_return += ep_return
        if steps > 0:
            avg_quality += ep_quality / steps
            avg_speed += ep_speed / steps

    avg_return /= eval_episode_num
    avg_quality /= eval_episode_num
    avg_speed /= eval_episode_num

    return avg_return, avg_quality, avg_speed

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_return = -float("inf")

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()
        
        # Enable wandb logging with both WandbCallback and custom episode logger
        # callback_list = [
        #     WandbCallback(
        #         gradient_save_freq=0,  # disable gradient logging to reduce artifact size
        #         verbose=0,
        #     ),
        #     WandbEpisodeLogger()
        # ]

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=callback_list,
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        eval_start = time.time()
        avg_return, avg_quality, avg_speed = eval(eval_env, model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance:")
        print(f"   - Avg Return: {avg_return:.3f}")
        print(f"   - Avg Quality Score: {avg_quality:.3f}")
        print(f"   - Avg Speed Score: {avg_speed:.3f}")
        
        # wandb.log({
        #     "eval/avg_return": avg_return,
        #     "eval/avg_quality": avg_quality,
        #     "eval/avg_speed": avg_speed,
        #     "epoch": epoch + 1,
        # })
        
        ### Save best model (by average return)
        if avg_return > current_best_return:
            print("Saving New Best Model")
            print(f"   - Previous best return: {current_best_return:.3f} → {avg_return:.3f}")
            current_best_return = avg_return
            save_path = config["save_path"]
            model.save(f"{save_path}/best")
        print("-"*60)
            
    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

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

# memory consumption evaluation
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')


def prepare_run_dir(base_output_dir: str) -> str:
    """
    Create a new run directory under base_output_dir, e.g. results/run1, run2, ...
    (same pattern as inference_for_eval.py)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    max_run_idx = 0
    for name in os.listdir(base_output_dir):
        match = re.match(r"run(\d+)$", name)
        if match:
            max_run_idx = max(max_run_idx, int(match.group(1)))
    run_dir = os.path.join(base_output_dir, f"run{max_run_idx + 1}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_used_metadata(run_dir: str, used_metadata: dict):
    """
    Save a meta_data.json describing all prompts used in this run.
    """
    meta_output_path = os.path.join(run_dir, "meta_data.json")
    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(used_metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote metadata for generated prompts to {os.path.abspath(meta_output_path)}")

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run VAR inference and save eval outputs.")
    parser.add_argument(
        "-3c",
        "--three-class",
        dest="three_class",
        action="store_true",
        help="Save outputs into landscape/food/people subfolders with their own JSON files.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_cli_args()

    # Create wandb session
    # run = wandb.init(
    #     project="FastVAR_Infinity",
    #     name=my_config["run_id"],
    #     config=my_config,
    #     sync_tensorboard=True,
    #     id=my_config["run_id"],
    #     resume="allow",
    # )

    model_path='/nfs/home/tensore/pretrained/Infinity/infinity_2b_reg.pth'
    vae_path=  '/nfs/home/tensore/pretrained/Infinity/infinity_vae_d32reg.pth'
    text_encoder_ckpt = '/nfs/home/tensore/pretrained/Infinity/models--google--flan-t5-x'

    # ------------ multi-prompt definition (name -> text) -------------
    with open("/nfs/home/tensore/RL/FastRLVAR/Infinity/infinity/dataset/meta_data.json") as f:
        meta_data = json.load(f)

    three_class_categories = ["landscape", "food", "people"]
    three_class_enabled = cli_args.three_class
    prompt_limit = int(my_config.get("prompt_pool_size", 0))
    if three_class_enabled:
        # Run all prompts when using -3c.
        prompt_limit = 0

    if three_class_enabled:
        prompts_by_category = {cat: [] for cat in three_class_categories}
        for img_id, data in meta_data.items():
            prompt = data.get("prompt")
            if not prompt:
                continue
            cats = data.get("category", [])
            if isinstance(cats, str):
                cats_list = [cats]
            else:
                cats_list = list(cats)
            cats_norm = [str(c).lower() for c in cats_list]
            target_cat = None
            for cat in three_class_categories:
                if cat in cats_norm:
                    target_cat = cat
                    break
            if target_cat is None:
                continue
            if prompt_limit and len(prompts_by_category[target_cat]) >= prompt_limit:
                continue
            prompts_by_category[target_cat].append((img_id, prompt))
    else:
        prompts = {}
        for img_id, data in meta_data.items():
            # if 'people' in data['category']:
            # if 'fashion' in data['category']:
            prompt = data.get("prompt")
            if not prompt:
                continue
            prompts[img_id] = prompt
            if prompt_limit and len(prompts) >= prompt_limit:
                break

    # prompts = {
    #     # "cat":       "A cute cat on the grass.",
    #     "city":      "A futuristic city skyline at night.",
    #     # "astronaut": "An astronaut painting on the moon.",
    #     # "woman":     "An anime-style portrait of a woman.",
    #     # "man":       "A detailed photo-realistic image of a man."
    # }

    # Base results dir; each run gets its own subfolder (run1, run2, ...)
    base_output_dir = "results"
    os.makedirs(base_output_dir, exist_ok=True)
    run_output_dir = prepare_run_dir(base_output_dir)
    print(f"Saving outputs to {os.path.abspath(run_output_dir)}")
    if three_class_enabled:
        output_dirs = {}
        for cat in three_class_categories:
            cat_dir = os.path.join(run_output_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)
            output_dirs[cat] = cat_dir
    else:
        output_dirs = None

    # si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    # pruning_scales = "2:1.0,4:1.0,6:1.0,8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    # pruning_scales = "8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    # pruning_scales = "20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    pruning_scales = "32:0.4,40:0.5,48:1.0,64:1.0"
    # pruning_scales = "64:1.0"
    parsed_prune_scales = parse_pruning_scales(pruning_scales)

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

    #### load PPO model
    load_model = False
    trained_model_path = None
    if load_model:

        OBS_H, OBS_W = 64, 64
        OBS_CHANNELS = 32 + 1  # 32 code channels + 1 scale channel
        SKIP_FIRST_N_SCALES = 9

        trained_model_path = './checkpoint/best_PPO_v4.zip'
        print(f"Loading PPO Agent from {trained_model_path}...")
        # 強制使用自訂的 CNN feature extractor，避免預設 NatureCNN 對 Box(-inf, inf, (33, 64, 64)) 報錯
        policy_kwargs = dict(
            features_extractor_class=FastVARCNNExtractor,
            # features_extractor_kwargs=dict(features_dim=128, width=32),
            features_extractor_kwargs=dict(features_dim=256, width=64),
            normalize_images=False,
        )
        custom_objects = {
            "policy_kwargs": policy_kwargs,
        }
        model = PPO.load(
            trained_model_path,
            device='cuda',
            custom_objects=custom_objects,
        )

    if trained_model_path:
        ppo_name = os.path.splitext(os.path.basename(trained_model_path))[0]
        ctime_filename = f"ctime_{ppo_name}.txt"
    else:
        ctime_filename = "consume_time.txt"

    cfg_value = 4
    tau_value = 0.5
    h_div_w = 1 / 1  # aspect ratio, height:width
    seed = 0
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # print(scale_schedule)

    torch.cuda.synchronize()

    # global metadata for all prompts (optional, stored in base_output_dir)
    global_prompt_records = []

    
    # Create a single model that learns across all prompts
    if three_class_enabled:
        total_prompts = sum(len(items) for items in prompts_by_category.values())
        print(f"Creating environments for {total_prompts} prompts across 3 classes...")
        prompt_items = []
        for cat in three_class_categories:
            prompt_items.extend([(img_id, prompt, cat) for img_id, prompt in prompts_by_category[cat]])
    else:
        print(f"Creating environments for {len(prompts)} prompts...")
        # Keep both id and text for each prompt so we can save {Id}.jpg and metadata
        prompt_items = list(prompts.items())

    cfg_list=[cfg_value] * len(scale_schedule)
    tau_list=[tau_value] * len(scale_schedule)
    negative_prompt=''
    scale_schedule=scale_schedule
    top_k=900
    top_p=0.97
    cfg_sc=3
    cfg_exp_k=0.0
    cfg_insertion_layer=-5
    vae_type=1
    gumbel=0
    softmax_merge_topk=-1
    gt_leak=0
    gt_ls_Bl=None
    g_seed=0
    sampling_per_bits=1
    enable_positive_prompt=0
    save_intermediate_results=False
    save_dir=None
    per_scale_infer=True

    if three_class_enabled:
        used_metadata = {cat: {} for cat in three_class_categories}
    else:
        used_metadata = {}

    for image_num, item in enumerate(prompt_items):
        if three_class_enabled:
            img_id, prompt, category = item
            output_dir = output_dirs[category]
            used_metadata_store = used_metadata[category]
        else:
            img_id, prompt = item
            category = None
            output_dir = run_output_dir
            used_metadata_store = used_metadata
        start_time_total = time.time()
        consume_time = 0

        if not isinstance(cfg_list, list):
            cfg_list = [cfg_list] * len(scale_schedule)
        if not isinstance(tau_list, list):
            tau_list = [tau_list] * len(scale_schedule)
        text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
        if negative_prompt:
            negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
        else:
            negative_label_B_or_BLT = None
        # print(f'cfg: {cfg_list}, tau: {tau_list}')
            if per_scale_infer:
                # Use step-wise generation via `infer_pruned_per_scale`, iterating all scales.
                # This matches `autoregressive_infer_cfg` behavior (VAE / bit-label path)
                # while allowing dynamic per-scale pruning.

                if vae_type == 0 or not getattr(infinity, "use_bit_label", True):
                    raise NotImplementedError(
                        "per_scale_infer=True is currently supported only for vae_type != 0 "
                        "and use_bit_label == True."
                    )

                # Ensure cfg_insertion_layer is a list, as expected by the model.
                if not isinstance(cfg_insertion_layer, (list, tuple)):
                    cfg_insertion_layer_ = [cfg_insertion_layer]
                else:
                    cfg_insertion_layer_ = list(cfg_insertion_layer)

                state = None
                summed_codes = None
                img = None

                num_scales = len(scale_schedule)

                for si in range(num_scales):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    decode_img = si == num_scales - 1
                    if load_model is not True:
                        prune_ratio = get_pruning_ratio(si, num_scales)
                    else:
                        if summed_codes is not None:
                            # Match VAREnv interpolation logic
                            interp_mode = getattr(
                                getattr(vae, "quantizer", None),
                                "z_interplote_up",
                                "area",
                            )
                            # Resize [B, C, H, W] -> [B, C, 64, 64]
                            resized = F.interpolate(
                                summed_codes, 
                                size=(1, OBS_H, OBS_W), 
                                mode=interp_mode
                            ) 
                            # Extract single item [32, 64, 64] and move to CPU/Float
                            codes_resized = resized.squeeze(-3)[0].cpu().float()
                        else:
                            # First step: Empty codes
                            codes_resized = torch.zeros(OBS_CHANNELS - 1, OBS_H, OBS_W, dtype=torch.float32)
                        norm_scale = float(si / max(num_scales - 1, 1))
                        scale_plane = torch.full(
                            (1, OBS_H, OBS_W),
                            norm_scale,
                            dtype=codes_resized.dtype,
                            device=codes_resized.device,
                        )

                        # C. Concatenate and Convert to Numpy for PPO
                        full_obs_tensor = torch.cat([codes_resized, scale_plane], dim=0)
                        obs = full_obs_tensor.numpy().astype(np.float32)
                        if si < SKIP_FIRST_N_SCALES:
                            prune_ratio = 0.0
                        else:
                            # print(obs)
                            # Predict action
                            action, _ = model.predict(obs, deterministic=True)
                            # print(action)
                            # Convert Action [-1, 1] -> Pruning Ratio [0, 1]
                            prune_ratio = (float(action[0]) + 1) * 0.5
                            prune_ratio = max(0.0, min(1.0, prune_ratio))
                            if prune_ratio > 0.95:
                                prune_ratio = 1.0
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                        codes, summed_codes, img, state = infinity.infer_pruned_per_scale(
                            vae=vae,
                            scale_schedule=scale_schedule,
                            label_B_or_BLT=text_cond_tuple,
                            scale_ind=si,
                            prune_ratio=prune_ratio,
                            B=1,
                            negative_label_B_or_BLT=negative_label_B_or_BLT,
                            g_seed=g_seed if si == 0 else None,
                            cfg_list=cfg_list,
                            tau_list=tau_list,
                            cfg_sc=cfg_sc,
                            top_k=top_k,
                            top_p=top_p,
                            returns_vemb=1,
                            cfg_insertion_layer=cfg_insertion_layer_,
                            vae_type=vae_type,
                            trunk_scale=1000,
                            gt_leak=gt_leak if gt_leak >= 0 else 0,
                            gt_ls_Bl=gt_ls_Bl,
                            inference_mode=True,
                            sampling_per_bits=sampling_per_bits,
                            save_intermediate_results=save_intermediate_results,
                            save_dir=save_dir,
                            state=state,
                            decode_img=decode_img,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    consume_time += time.perf_counter() - wall_start

                if img is None and isinstance(summed_codes, torch.Tensor):
                    img = vae.decode(summed_codes.squeeze(-3))
                    img = (img + 1) / 2
                    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))

                # One full image (all scales) is done; reset print cache so pruning
                # ratio logs are emitted again for the next image.
                reset_prune_print_cache()

                # `img` is already the decoded uint8 image for the final scale.
                img_list = [img]

            else:
                _, intermidiate_list, img_list = infinity.autoregressive_infer_cfg(
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple,
                    g_seed=g_seed,
                    B=1,
                    negative_label_B_or_BLT=negative_label_B_or_BLT,
                    force_gt_Bhw=None,
                    cfg_sc=cfg_sc,
                    cfg_list=cfg_list,
                    tau_list=tau_list,
                    top_k=top_k,
                    top_p=top_p,
                    returns_vemb=1,
                    ratio_Bl1=None,
                    gumbel=gumbel,
                    norm_cfg=False,
                    cfg_exp_k=cfg_exp_k,
                    cfg_insertion_layer=cfg_insertion_layer,
                    vae_type=vae_type,
                    softmax_merge_topk=softmax_merge_topk,
                    ret_img=True,
                    trunk_scale=1000,
                    gt_leak=gt_leak,
                    gt_ls_Bl=gt_ls_Bl,
                    inference_mode=True,
                    sampling_per_bits=sampling_per_bits,
                    save_intermediate_results=save_intermediate_results,
                    save_dir=save_dir,
                )

        # img_list elements are batched tensors [B, H, W, C]; for our single-image
        # inference we return the first item in the batch.
        img = img_list[0]
        if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[0] == 1:
            img = img[0]

        with open(ctime_filename, "a") as file:
            file.write(f'{img_id}: {consume_time}\n')

        total_elapsed = time.time() - start_time_total
        with open("total_consume_time.txt", "a") as file:
            file.write(f'{img_id}: {total_elapsed}\n')

        img_path = os.path.join(output_dir, f"{img_id}.jpg")
        cv2.imwrite(img_path, img.cpu().numpy())
        print(f"Saved image for {img_id} to {os.path.abspath(img_path)} (total {total_elapsed:.2f}s, infer-only {consume_time:.2f}s)")

        # Collect metadata for this image id so we can write a meta_data.json
        # that mirrors the structure used by inference_for_eval.py
        if img_id in meta_data:
            used_metadata_store[img_id] = meta_data[img_id]
        else:
            used_metadata_store[img_id] = {"prompt": prompt}

    # Save metadata for all generated images in this run
    if three_class_enabled:
        for cat in three_class_categories:
            save_used_metadata(output_dirs[cat], used_metadata[cat])
    else:
        save_used_metadata(run_output_dir, used_metadata)
