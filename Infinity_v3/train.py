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
    "run_id": "hybrid_PPO_3",
    "algorithm": PPO,
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model/PPO",
    "num_train_envs": 1,
    "epoch_num": 50,
    # "eval_episode_num": 30,  # Evaluate over multiple prompts for more stable metrics
    "training_prompt_pool_size": 1000,
    "eval_prompt_pool_size": 30,
    "num_training_classes": 10,
    # Shuffle training prompt order (reproducible)
    "prompt_shuffle_seed": 0,
    "iterations_per_prompt": 1,
    # VAREnv optimization: we pre-run the first N scales inside reset() and only
    # learn pruning decisions on later scales.
    "skip_first_N_scales": 9,

    "base_steps_per_episode": 4,
    "ppo_batch_size_base": 32,          
    "sac_learning_starts_base": 40,     
}

# NOTE: rollout_steps and timesteps_per_epoch depend on the number of scales,
# which is only known after scale_schedule is constructed. We set these later.

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
                        'custom/dinov3': episode_info.get('dinov3', 0),
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
        prompt,
        skip_first_N_scales=my_config["skip_first_N_scales"],
    )
    env = Monitor(
        env,
        info_keywords=(
            "quality_reward",
            "speed_reward",
            "quality_score",
            "speed_score",
            "total_reward",
            "psnr",
            "dinov3",
            "prune_ratio",
        ),
    )
    return env

def eval(env, model, eval_prompt_pool_size):
    """
    Evaluate the model on the vectorized FastVAR environment.
    Always evaluates on the first N prompts to ensure fair comparison across epochs.
    Returns:
        avg_return: average episode return
        avg_quality: average per-step quality_score
        avg_speed: average per-step speed_score
    """
    # Reset prompt index to ensure we always evaluate on the same set/order of prompts.
    # NOTE: env.envs[0] is a Monitor wrapper; use .unwrapped to reach VAREnv.
    base_env = env.envs[0].unwrapped
    base_env.prompt_idx = -1  # Reset to start from prompt 0
    base_env._first_reset = True  # Reset the first_reset flag
    
    avg_return = 0.0
    avg_quality = 0.0
    avg_speed = 0.0

    for _ in range(eval_prompt_pool_size):
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

    avg_return /= eval_prompt_pool_size
    avg_quality /= eval_prompt_pool_size
    avg_speed /= eval_prompt_pool_size

    return avg_return, avg_quality, avg_speed

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_return = -float("inf")
    current_best_quality = -float("inf")
    current_best_speed = -float("inf")

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()
        steps_before = int(getattr(model, "num_timesteps", 0))
        
        # Enable wandb logging with both WandbCallback and custom episode logger
        callback_list = [
            WandbCallback(
                gradient_save_freq=0,  # disable gradient logging to reduce artifact size
                verbose=0,
            ),
            WandbEpisodeLogger()
        ]

        model.learn(
            # Match SB3 recommended usage (see RL/HW3/train.py):
            # with reset_num_timesteps=False, SB3 will internally add current num_timesteps
            # so this trains for an additional `timesteps_per_epoch` each epoch.
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=callback_list,
        )
        steps_after = int(getattr(model, "num_timesteps", 0))
        epoch_env_steps = steps_after - steps_before
        if epoch_env_steps != int(config["timesteps_per_epoch"]):
            print(
                f"WARNING: epoch_env_steps={epoch_env_steps} != timesteps_per_epoch={int(config['timesteps_per_epoch'])}. "
                "This indicates something changed the expected step accounting."
            )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        eval_start = time.time()
        avg_return, avg_quality, avg_speed = eval(eval_env, model, config["eval_prompt_pool_size"])
        eval_duration = time.time() - eval_start

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance (averaged over {config['eval_prompt_pool_size']} prompts):")
        print(f"   - Avg Return: {avg_return:.3f}")
        print(f"   - Avg Quality Score: {avg_quality:.3f}")
        print(f"   - Avg Speed Score: {avg_speed:.3f}")
        print(f"   - Env steps this epoch: {epoch_env_steps}")
        
        wandb.log({
            "eval/avg_return": avg_return,
            "eval/avg_quality": avg_quality,
            "eval/avg_speed": avg_speed,
            "epoch": epoch + 1,
            "train/epoch_env_steps": epoch_env_steps,
        }, step=model.num_timesteps)  # Use actual environment steps for consistent x-axis
        
        save_path = config["save_path"]
        saved_any = False
        
        ### Save best model by average return
        if avg_return > current_best_return:
            print("Saving New Best Model (by Return)")
            print(f"   - Previous best return: {current_best_return:.3f} → {avg_return:.3f}")
            current_best_return = avg_return
            model.save(f"{save_path}/best_reward")
            saved_any = True
        
        ### Save best model by average quality
        if avg_quality > current_best_quality:
            print("Saving New Best Model (by Quality)")
            print(f"   - Previous best quality: {current_best_quality:.3f} → {avg_quality:.3f}")
            current_best_quality = avg_quality
            model.save(f"{save_path}/best_quality")
            saved_any = True
        
        ### Save best model by average speed
        if avg_speed > current_best_speed:
            print("Saving New Best Model (by Speed)")
            print(f"   - Previous best speed: {current_best_speed:.3f} → {avg_speed:.3f}")
            current_best_speed = avg_speed
            model.save(f"{save_path}/best_speed")
            saved_any = True
        
        if not saved_any:
            print("No new best models this epoch")
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

    model_path = '/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/checkpoint/infinity_2b_reg.pth'
    vae_path   = '/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/checkpoint/infinity_vae_d32reg.pth'
    text_encoder_ckpt = 'google/flan-t5-xl'

    # ------------ multi-prompt definition (name -> text) -------------
    with open("/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/evaluation/MJHQ30K/meta_data.json") as f:
        meta_data = json.load(f)

    category_list = ['animals', 'art', 'fashion', 'food', 'indoor', 'landscape', 'logo', 'people', 'plants', 'vehicles']

    # ----------------- build disjoint, class-balanced train/eval prompt pools -----------------
    num_classes = int(my_config["num_training_classes"])
    if num_classes != len(category_list):
        raise ValueError(
            f"num_training_classes={num_classes} must match len(category_list)={len(category_list)} "
            "for uniform sampling across all classes."
        )

    train_pool_size = int(my_config["training_prompt_pool_size"])
    eval_pool_size = int(my_config["eval_prompt_pool_size"])
    if train_pool_size < num_classes:
        raise ValueError(
            f"training_prompt_pool_size={train_pool_size} must be >= num_training_classes={num_classes} "
            "to sample from all classes."
        )
    if eval_pool_size < num_classes:
        raise ValueError(
            f"eval_prompt_pool_size={eval_pool_size} must be >= num_training_classes={num_classes} "
            "to sample from all classes."
        )

    def _nearly_uniform_counts(total: int, classes: int) -> List[int]:
        """
        Allocate `total` items across `classes` buckets with counts differing by at most 1.
        Deterministic: extra items go to earlier buckets.
        """
        base = total // classes
        rem = total % classes
        return [base + (1 if i < rem else 0) for i in range(classes)]

    per_class_train_list = _nearly_uniform_counts(train_pool_size, num_classes)
    per_class_eval_list = _nearly_uniform_counts(eval_pool_size, num_classes)

    prompts_by_cat = {c: [] for c in category_list}
    for img_id, data in meta_data.items():
        cats = data.get("category", [])
        # meta_data may store category as a string or a list; normalize to list[str].
        if isinstance(cats, str):
            cats_list = [cats]
        else:
            cats_list = list(cats)

        for c in category_list:
            if c in cats_list:
                prompt = data.get("prompt", None)
                if prompt:
                    prompts_by_cat[c].append(prompt)

    # Shuffle per-category prompt order to avoid bias from JSON iteration order.
    rng = random.Random(int(my_config.get("prompt_shuffle_seed", 0)))
    for c in category_list:
        rng.shuffle(prompts_by_cat[c])

    # Ensure we have enough prompts per class to make train/eval disjoint.
    for i, c in enumerate(category_list):
        need = per_class_train_list[i] + per_class_eval_list[i]
        have = len(prompts_by_cat[c])
        if have < need:
            raise ValueError(
                f"Not enough prompts for category '{c}': need {need} "
                f"(train {per_class_train_list[i]} + eval {per_class_eval_list[i]}), have {have}."
            )

    # Deterministic selection: per class, take first K for train, next M for eval.
    train_prompt_list = []
    eval_prompt_list = []
    for i, c in enumerate(category_list):
        k_train = per_class_train_list[i]
        k_eval = per_class_eval_list[i]
        train_prompt_list.extend(prompts_by_cat[c][:k_train])
        eval_prompt_list.extend(prompts_by_cat[c][k_train:k_train + k_eval])

    # Shuffle training prompt order so VAREnv's round-robin prompt cycling
    # sees a randomized class-mixed sequence.
    rng.shuffle(train_prompt_list)

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

    pruning_scales = "64:0.0"
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

    # ----------------- step accounting (must match VAREnv) -----------------
    # VAREnv will run the first skip_first_N_scales internally during reset().
    steps_per_episode = len(scale_schedule) - int(my_config["skip_first_N_scales"])
    if steps_per_episode <= 0:
        raise ValueError(
            f"Invalid configuration: len(scale_schedule)={len(scale_schedule)} "
            f"<= skip_first_N_scales={my_config['skip_first_N_scales']}."
        )

    # Keep the original heuristic: 16 episodes per PPO rollout, but with the
    # new per-episode step count.
    my_config["rollout_steps"] = steps_per_episode * 16

    # ----------------- scale step-coupled hyperparams -----------------
    base_steps = int(my_config.get("base_steps_per_episode", 13))
    if base_steps <= 0:
        raise ValueError(f"base_steps_per_episode must be > 0, got {base_steps}")
    scale_factor = float(steps_per_episode) / float(base_steps)

    # PPO batch size: scale and keep it divisible by steps_per_episode, and <= rollout_steps * n_envs.
    ppo_batch_target = int(round(int(my_config.get("ppo_batch_size_base", 104)) * scale_factor))
    ppo_batch_target = max(steps_per_episode, ppo_batch_target)
    # snap down to a multiple of steps_per_episode
    ppo_batch_target = (ppo_batch_target // steps_per_episode) * steps_per_episode
    # hard constraint in SB3 PPO: batch_size <= n_steps * n_envs
    max_ppo_batch = int(my_config["rollout_steps"]) * int(my_config["num_train_envs"])
    if ppo_batch_target > max_ppo_batch:
        ppo_batch_target = max(steps_per_episode, (max_ppo_batch // steps_per_episode) * steps_per_episode)
    my_config["ppo_batch_size"] = int(max(1, ppo_batch_target))

    # SAC learning_starts: scale and align to whole episodes (multiple of steps_per_episode).
    sac_ls_target = int(round(int(my_config.get("sac_learning_starts_base", 130)) * scale_factor))
    sac_ls_target = max(steps_per_episode, sac_ls_target)
    sac_ls_target = (sac_ls_target // steps_per_episode) * steps_per_episode
    my_config["sac_learning_starts"] = int(max(0, sac_ls_target))

    # Train for prompt_pool_size episodes per epoch (times iterations_per_prompt),
    # using the new per-episode step count.
    my_config["timesteps_per_epoch"] = (
        my_config["training_prompt_pool_size"] * steps_per_episode * my_config["iterations_per_prompt"]
    )

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # global metadata for all prompts (optional, stored in base_output_dir)
    global_prompt_records = []

    # Create a single model that learns across all prompts
    print(f"Creating environments for {len(train_prompt_list)} training prompts...")
    
    # Create training environments: each env receives the full prompt list and
    # VAREnv will rotate to a new prompt on each episode via reset().
    prompt_list = train_prompt_list
    
    def make_env_factory(prompts_for_env):
        """Factory function to create env with a prompt pool (fixes closure issue)."""
        def _make():
            return make_env(infinity, vae, scale_schedule, text_tokenizer, text_encoder, prompts_for_env)
        return _make
    
    train_env = DummyVecEnv([
        make_env_factory(prompt_list)
        for _ in range(my_config["num_train_envs"])
    ])
    
    # For eval, use a fixed prompt pool that is DISJOINT from training.
    # We still run `eval_prompt_pool_size` episodes, which will round-robin over this fixed list.
    eval_env = DummyVecEnv([
        make_env_factory(eval_prompt_list)
    ])
    
    print("Creating model...")
    device = 'cuda'
    policy_kwargs = dict(
        features_extractor_class=FastVARCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256, width=64),
        normalize_images=False,
    )

    if my_config["algorithm"] == PPO:
        model = my_config["algorithm"](
            my_config["policy_network"],
            train_env,
            verbose=0,
            tensorboard_log=my_config["run_id"],
            policy_kwargs=policy_kwargs,
            device=device,
            # PPO-specific hyperparams
            n_steps=my_config["rollout_steps"],  
            batch_size=my_config["ppo_batch_size"],
            n_epochs=4,                          
            learning_rate=1e-4,                  
            ent_coef=0.001,
            gamma=0.99,                          
        )
    elif my_config["algorithm"] == SAC:
        model = my_config["algorithm"](
            my_config["policy_network"],
            train_env,
            verbose=0,
            tensorboard_log=my_config["run_id"],
            policy_kwargs=policy_kwargs,
            device=device,
            # SAC-specific hyperparams
            batch_size=256,
            train_freq=1,
            gradient_steps=4,     
            learning_rate=3e-4,
            buffer_size=20000,      # small, safe
            learning_starts=my_config["sac_learning_starts"],
            tau=0.005,
            gamma=0.99,
        )

    print(f"Starting training on {len(train_prompt_list)} prompts...")
    train(eval_env, model, my_config)