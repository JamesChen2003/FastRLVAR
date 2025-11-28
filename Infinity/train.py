import warnings
import time
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from types import MethodType

from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback
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
    "prompt_pool_size": 100,
    "iterations_per_prompt": 1,
}

my_config["timesteps_per_epoch"] = my_config["prompt_pool_size"] * 13 * my_config["iterations_per_prompt"]

class WandbEpisodeLogger(BaseCallback):
    """Callback to log custom episode metrics to wandb"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # Check if any episode ended
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                episode_info = info['episode']
                wandb.log(
                    {
                        # Episode-level aggregates from Monitor
                        'custom/ep_rew': episode_info.get('r', 0),
                        'custom/ep_len': episode_info.get('l', 0),
                        # Per-episode averages/last values of our custom metrics
                        'custom/quality_reward': episode_info.get('quality_reward', 0),
                        'custom/speed_reward': episode_info.get('speed_reward', 0),
                        'custom/quality_score': episode_info.get('quality_score', 0),
                        'custom/speed_score': episode_info.get('speed_score', 0),
                        'custom/total_reward': episode_info.get('total_reward', 0),
                        'custom/psnr': episode_info.get('psnr', 0),
                        'custom/ms_ssim': episode_info.get('ms_ssim', 0),
                        'custom/lpips': episode_info.get('lpips', 0),
                        'custom/prune_ratio': episode_info.get('prune_ratio', 0),
                    },
                    step=self.num_timesteps,
                )
        
        return True

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
        callback_list = [
            WandbCallback(
                gradient_save_freq=0,  # disable gradient logging to reduce artifact size
                verbose=0,
            ),
            WandbEpisodeLogger()
        ]

        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=callback_list,
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
        
        wandb.log({
            "eval/avg_return": avg_return,
            "eval/avg_quality": avg_quality,
            "eval/avg_speed": avg_speed,
            "epoch": epoch + 1,
        })
        
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

if __name__ == "__main__":
    # Create wandb session
    run = wandb.init(
        project="FastVAR_Infinity",
        name=my_config["run_id"],
        config=my_config,
        sync_tensorboard=True,
        id=my_config["run_id"],
        resume="allow",
    )

    model_path = '/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/checkpoint/infinity_2b_reg.pth'
    vae_path   = '/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/checkpoint/infinity_vae_d32reg.pth'
    text_encoder_ckpt = 'google/flan-t5-xl'

    # ------------ multi-prompt definition (name -> text) -------------
    with open("/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/evaluation/MJHQ30K/meta_data.json") as f:
        meta_data = json.load(f)

    prompts = {}

    for img_id, data in meta_data.items():
        # if 'people' in data['category']:
        if 'fashion' in data['category']:
            prompts[img_id] = data['prompt']
        if len(prompts) >= my_config["prompt_pool_size"]:
            break

    # prompts = {
    #     # "cat":       "A cute cat on the grass.",
    #     "city":      "A futuristic city skyline at night.",
    #     # "astronaut": "An astronaut painting on the moon.",
    #     # "woman":     "An anime-style portrait of a woman.",
    #     # "man":       "A detailed photo-realistic image of a man."
    # }

    # Base results dir; each prompt gets its own subfolder
    base_output_dir = "results"
    os.makedirs(base_output_dir, exist_ok=True)

    # si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
    # pruning_scales = "2:1.0,4:1.0,6:1.0,8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    # pruning_scales = "8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    # pruning_scales = "20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
    # pruning_scales = "48:1.0,64:1.0"
    pruning_scales = "64:1.0"
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

    # global metadata for all prompts (optional, stored in base_output_dir)
    global_prompt_records = []

    # with torch.inference_mode():
    #     with measure_peak_memory():
            # enumerate prompts with index for seeding
    #         for idx, (name, prompt_text) in enumerate(prompts.items(), start=1):
    #             # per-prompt output dir and metadata path
    #             root_output_dir = os.path.join(base_output_dir, f"gen_images_{name}")
    #             os.makedirs(root_output_dir, exist_ok=True)
    #             metadata_path = os.path.join(root_output_dir, "config.json")

    #             print(f"\n=== Generating for prompt '{name}': '{prompt_text}' ===")
    #             start_event.record()

    #             generated_image = gen_one_img(
    #                 infinity,
    #                 vae,
    #                 text_tokenizer,
    #                 text_encoder,
    #                 prompt_text,
    #                 g_seed=seed + idx,   # different seed per prompt
    #                 gt_leak=0,
    #                 gt_ls_Bl=None,
    #                 cfg_list=[cfg_value] * len(scale_schedule),
    #                 tau_list=[tau_value] * len(scale_schedule),
    #                 scale_schedule=scale_schedule,
    #                 cfg_insertion_layer=[args.cfg_insertion_layer],
    #                 vae_type=args.vae_type,
    #                 sampling_per_bits=args.sampling_per_bits,
    #                 enable_positive_prompt=enable_positive_prompt,
    #                 save_intermediate_results=True,
    #                 save_dir=root_output_dir,   # <<< all intermediate images & fourier here
    #                 per_scale_infer=True,
    #             )

    #             # save main generated image as 1.jpg in that folder
    #             img_path = os.path.join(root_output_dir, "1.jpg")
    #             cv2.imwrite(img_path, generated_image.cpu().numpy())
    #             print(f"Saved image for '{name}' to {os.path.abspath(img_path)}")

    #             # per-prompt metadata
    #             prompt_record = {
    #                 "prompt_name": name,
    #                 "prompt": prompt_text,
    #                 "image_path": img_path,
    #                 "image_number": 1,
    #             }
    #             with open(metadata_path, "w", encoding="utf-8") as f:
    #                 json.dump(
    #                     {
    #                         "pruning_scales": parsed_prune_scales or {},
    #                         "pruning_scales_raw": pruning_scales,
    #                         "prompt": prompt_record,
    #                     },
    #                     f,
    #                     ensure_ascii=False,
    #                     indent=2,
    #                 )
    #             print(f"Saved prompt metadata to {os.path.abspath(metadata_path)}")

    #             # add to global list
    #             global_prompt_records.append(prompt_record)

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

    # Create a single model that learns across all prompts
    print(f"Creating environments for {len(prompts)} prompts...")
    
    # Create training environments: each env receives the full prompt list and
    # VAREnv will rotate to a new prompt on each episode via reset().
    prompt_list = list(prompts.values())
    
    def make_env_factory(prompts_for_env):
        """Factory function to create env with a prompt pool (fixes closure issue)."""
        def _make():
            return make_env(infinity, vae, scale_schedule, text_tokenizer, text_encoder, prompts_for_env)
        return _make
    
    train_env = DummyVecEnv([
        make_env_factory(prompt_list)
        for _ in range(my_config["num_train_envs"])
    ])
    
    # For eval, also use the same prompt pool (episodes will cycle through prompts)
    eval_env = DummyVecEnv([
        make_env_factory(prompt_list)
    ])
    
    print("Creating model...")
    device = 'cuda'
    policy_kwargs = dict(
        features_extractor_class=FastVARCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256, width=64),
        normalize_images=False,
    )
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env,
        verbose=0,
        tensorboard_log=my_config["run_id"],
        policy_kwargs=policy_kwargs,
        device = device,
        n_steps=my_config["timesteps_per_epoch"]
    )
    print(f"Starting training on {len(prompts)} prompts...")
    train(eval_env, model, my_config)