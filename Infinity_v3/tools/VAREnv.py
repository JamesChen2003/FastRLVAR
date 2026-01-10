import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse
import shutil
import re

import cv2
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

from infinity.models.infinity import Infinity
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import torchvision.utils as vutils
from infinity.models.fastvar_utils import reset_prune_print_cache

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, prompt_idx=0):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'\n\n📌 prompt no.{prompt_idx}: {prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

class VAREnv(gym.Env):
    def __init__(self, infinity_test, vae, scale_schedule, text_tokenizer, text_encoder, prompt, 
                 alpha: float = 0.7, beta: float = 4.0, 
                 golden_dir: str = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/golden_images",
                 # Inference configuration
                 B: int = 1,
                 negative_label_B_or_BLT=None,
                 g_seed=42,
                 cfg_list=None,
                 tau_list=None,
                 cfg_sc: float = 3,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 returns_vemb: int = 1,
                 cfg_insertion_layer=[-5],
                 vae_type: int = 1,
                 trunk_scale: int = 1000,
                 gt_leak: int = 0,
                 gt_ls_Bl=None,
                 skip_first_N_scales: int = 9,
                 # IMPORTANT: infer_pruned_per_scale expects inference_mode=True in this repo's Infinity implementation.
                 # Passing False can send it down a different codepath and crash (e.g., ModuleList has no attribute 'module').
                 inference_mode: bool = True,
                 sampling_per_bits: int = 1,
                 save_intermediate_results: bool = False,
                 save_dir: str = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/training_tmp_results",
                 ): # r14_jameschen-1000043/FastVAR/Infinity/golden_images #d10_rick_huang-1000011/RL/RL_final_project/FastRLVAR/golden_images
        self.infinity_test = infinity_test
        self.vae = vae
        self.scale_schedule = scale_schedule
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.alpha = alpha  # weight for quality vs speed reward
        self.beta = beta    # weight for quality vs speed reward
        self.golden_dir = golden_dir

        # Store inference configuration
        self.B = B
        self.negative_label_B_or_BLT = negative_label_B_or_BLT
        self.g_seed = g_seed
        self.cfg_list = cfg_list
        self.tau_list = tau_list
        self.cfg_sc = cfg_sc
        self.top_k = top_k
        self.top_p = top_p
        self.returns_vemb = returns_vemb
        self.cfg_insertion_layer = cfg_insertion_layer
        self.vae_type = vae_type
        self.trunk_scale = trunk_scale
        self.gt_leak = gt_leak
        self.gt_ls_Bl = gt_ls_Bl
        self.skip_first_N_scales = skip_first_N_scales
        self.inference_mode = inference_mode
        self.sampling_per_bits = sampling_per_bits
        self.save_intermediate_results = save_intermediate_results
        self.save_dir = save_dir

        self.label_B_or_BLT = None
        # Continuous action [-1, 1], later squashed to [0, 1] via (action+1) / 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: [channels, H, W]; we use 32 VAE channels + 1 scale-index channel, resized to 64x64
        self.obs_channels = 32 + 32 + 1
        self.obs_H = 64
        self.obs_W = 64
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_channels, self.obs_H, self.obs_W),
            dtype=np.float32,
        )
        # Prompt pool: support either a single prompt or a list/tuple of prompts.
        if isinstance(prompt, (list, tuple)):
            self.prompts = list(prompt)
        else:
            self.prompts = [prompt]
        # Start before the first prompt; first reset() will select index 0.
        self.prompt_idx = -1
        self._first_reset = True

        self.seed()
        # Prompt-dependent state will be initialized on the first reset call.
        self.label_B_or_BLT = None
        self._golden_imgs = {}
        self._golden_codes = {}

        # TODO: Initialization (ensure golden images are precomputed)

    def _set_prompt(self, prompt: str):
        """Set current prompt, re-encode text condition, and clear golden cache."""
        self.prompt = prompt
        # Derive a stable prompt id and base seed from the prompt text so that
        # golden (unpruned) and pruned generations share the same noise.
        prompt_md5 = hashlib.md5(self.prompt.encode("utf-8")).hexdigest()
        self.prompt_id = prompt_md5
        # Use a portion of the hash as a deterministic base seed
        self.base_seed = int(prompt_md5[:8], 16)

        text_cond_tuple = encode_prompt(
            self.text_tokenizer,
            self.text_encoder,
            self.prompt,
            prompt_idx=self.prompt_idx,
        )
        self.label_B_or_BLT = text_cond_tuple
        # Cache for unpruned (golden) images per scale index, tied to this prompt.
        self._golden_imgs = {}
        # Optional cache for golden summed_codes per scale (loaded/saved to disk)
        self._golden_codes = {}

    def _ensure_golden_images(self, g_seed=42, cfg_list=None, tau_list=None,
                              cfg_sc: float = 3, top_k: int = 0, top_p: float = 0.0,
                              cfg_insertion_layer=[-5], vae_type: int = 1,
                              trunk_scale: int = 1000, gt_leak: int = 0, gt_ls_Bl=None,
                              sampling_per_bits: int = 1):
        """
        Precompute and cache unpruned (golden) images and summed_codes for each
        scale index for this prompt. For each prompt+scale, we either load from
        disk (if already saved) or run a full per-scale pass with
        prune_ratio=0 and store both decoded images and summed_codes.
        """
        num_scales = len(self.scale_schedule)
        # Already fully cached in memory for this prompt
        if self._golden_imgs and len(self._golden_imgs) == num_scales:
            return True

        # If no explicit seed is provided, fall back to the prompt-based base seed
        if g_seed is None:
            g_seed_local = self.base_seed
        else:
            g_seed_local = g_seed

        # Use a stable ID per prompt to separate caches on disk.
        # Include seed in the directory name to avoid using cached images from a different seed
        prompt_dir = os.path.join(self.golden_dir, f"{self.prompt_id}_seed{g_seed_local}")
        os.makedirs(prompt_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Detect whether we have a *complete* on-disk cache for this prompt
        # and seed. Mixing partially cached scales with newly generated ones
        # would break the stateful contract of `infer_pruned_per_scale`
        # (it requires `state is None` only for `scale_ind == 0`).
        # ------------------------------------------------------------------
        all_cached = True
        for si in range(num_scales):
            img_path = os.path.join(prompt_dir, f"scale_{si}_golden.png")
            codes_path = os.path.join(prompt_dir, f"scale_{si}_golden_codes.pt")
            if not (os.path.exists(img_path) and os.path.exists(codes_path)):
                all_cached = False
                break

        # Fast path: all scales cached → just load into memory and return True.
        if all_cached:
            print(
                f"[VAREnv] ✅ Golden cache hit: prompt_idx={self.prompt_idx} "
                f"prompt_id={self.prompt_id[:8]} seed={g_seed_local} dir={prompt_dir}"
            )
            with torch.no_grad():
                for si in range(num_scales):
                    img_path = os.path.join(prompt_dir, f"scale_{si}_golden.png")
                    codes_path = os.path.join(prompt_dir, f"scale_{si}_golden_codes.pt")

                    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        golden = torch.from_numpy(img_rgb).float() / 255.0  # [H, W, C], RGB
                        self._golden_imgs[si] = golden.cpu()

                    try:
                        codes = torch.load(codes_path, map_location="cpu")
                        self._golden_codes[si] = codes
                    except Exception:
                        pass
            return True

        # If the cache is incomplete, drop any stale files to avoid mixing
        # old and new golden images, then regenerate everything in a single,
        # consistent forward pass starting from scale 0.
        print(
            f"[VAREnv] 🧱 Golden cache miss → generating: prompt_idx={self.prompt_idx} "
            f"prompt_id={self.prompt_id[:8]} seed={g_seed_local} dir={prompt_dir}"
        )
        t0 = time.time()
        for si in range(num_scales):
            img_path = os.path.join(prompt_dir, f"scale_{si}_golden.png")
            codes_path = os.path.join(prompt_dir, f"scale_{si}_golden_codes.pt")
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except OSError:
                    pass
            if os.path.exists(codes_path):
                try:
                    os.remove(codes_path)
                except OSError:
                    pass

        # Save RNG state to prevent side effects on the pruned generation process
        # This ensures that generating golden images (which consumes random numbers)
        # doesn't desynchronize the RNG for the next step of the pruned generation.
        rng_state = self.infinity_test.rng.get_state()

        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                state = None
                for si in range(num_scales):
                    img_path = os.path.join(prompt_dir, f"scale_{si}_golden.png")
                    codes_path = os.path.join(prompt_dir, f"scale_{si}_golden_codes.pt")

                    # Run unpruned per-scale inference sequentially so that
                    # `state` is correctly carried across scales.
                    codes, summed_codes, img, state = self.infinity_test.infer_pruned_per_scale(
                        vae=self.vae,
                        scale_schedule=self.scale_schedule,
                        label_B_or_BLT=self.label_B_or_BLT,
                        scale_ind=si,
                        prune_ratio=0.0,
                        B=1,
                        negative_label_B_or_BLT=None,
                        g_seed=g_seed_local if si == 0 else None,
                        cfg_list=cfg_list,
                        tau_list=tau_list,
                        cfg_sc=cfg_sc,
                        top_k=top_k,
                        top_p=top_p,
                        returns_vemb=1,
                        cfg_insertion_layer=cfg_insertion_layer,
                        vae_type=vae_type,
                        trunk_scale=trunk_scale,
                        gt_leak=max(gt_leak, 0),
                        gt_ls_Bl=gt_ls_Bl,
                        inference_mode=True,
                        sampling_per_bits=sampling_per_bits,
                        save_intermediate_results=False,
                        save_dir=prompt_dir,
                        state=state,
                    )
                    if isinstance(img, torch.Tensor):
                        # img: [B, H, W, C], uint8 BGR (matches Infinity + cv2.imwrite)
                        img_bgr = img[0].cpu().numpy()  # uint8 BGR for saving
                        # Convert to RGB float in [0, 1] for PSNR
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        golden = torch.from_numpy(img_rgb).float() / 255.0  # [H, W, C], RGB
                        self._golden_imgs[si] = golden.cpu()
                        # Save golden image to disk (BGR)
                        cv2.imwrite(img_path, img_bgr)
                    if isinstance(summed_codes, torch.Tensor):
                        # Save golden summed_codes and cache in memory
                        torch.save(summed_codes.cpu(), codes_path)
                        self._golden_codes[si] = summed_codes.cpu()
        finally:
            # Restore RNG state so the pruned generation process continues uninterrupted
            self.infinity_test.rng.set_state(rng_state)

        # If we reach here without error, golden images were freshly generated
        # for this prompt/seed (not loaded from an existing cache).
        # Return False to signal "not pre-existing" to the caller.
        print(
            f"[VAREnv] ✅ Golden generation done in {time.time() - t0:.2f}s: "
            f"prompt_idx={self.prompt_idx} prompt_id={self.prompt_id[:8]} seed={g_seed_local}"
        )
        return False

    @staticmethod
    def _psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
        """
        Compute PSNR between two images in [0,1], shape [H, W, C].
        """
        mse = torch.mean((x - y) ** 2)
        if mse.item() < eps:
            return 100.0
        psnr = 20.0 * torch.log10(torch.tensor(1.0, device=x.device)) - 10.0 * torch.log10(mse)
        return float(psnr.item())

    @staticmethod
    def _to_pil_rgb(img: torch.Tensor) -> Image.Image:
        """
        Convert an image tensor [H, W, C] in RGB float [0,1] to PIL.Image (RGB).
        """
        img_uint8 = (img * 255.0).clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(img_uint8, mode="RGB")

    def get_similarity(
        self,
        pruned_img_resized: torch.Tensor,
        golden_img: torch.Tensor,
        scale_index: int,
        skip_first_N_scales: int,
    ) -> dict:
        """
        Compute similarity between pruned and golden images using DINOv3 (cosine
        similarity of pooled features). Also returns PSNR for debugging.

        Returns a dict with:
        - psnr (float)
        - dinov3 (float, cosine similarity in [-1, 1])
        - similarity (float, same as dinov3 for downstream reward)
        """
        if scale_index < skip_first_N_scales:
            return {
                "psnr": 0.0,
                "dinov3": 0.0,
                "similarity": 0.0,
            }
        else:
            # Lazy import to avoid loading the DINO model unless needed.
            from tools.dinov3_score import get_dinov3_similarity

            psnr_val = self._psnr(pruned_img_resized, golden_img)
            pruned_pil = self._to_pil_rgb(pruned_img_resized)
            golden_pil = self._to_pil_rgb(golden_img)
            dinov3_val = float(get_dinov3_similarity(pruned_pil, golden_pil))
            dinov3_val = float(max(min(dinov3_val, 1.0), -1.0))
            return {
                "psnr": psnr_val,
                "dinov3": dinov3_val,
                "similarity": dinov3_val,
            }

    # def quality_func(self, x, a=0.68, b=0.92):
    def quality_func(self, x, a=0.88, b=0.98):
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

    def step(self, action):
        # Work with a local copy of the current scale index to avoid off-by-one
        # confusion between the model call, golden cache indexing, and logging.
        current_scale = self.scale_index

        # Map action [-1, 1] to prune_ratio [0, 1]
        prune_ratio = (float(action[0]) + 1) * 0.5
        prune_ratio = max(0.0, min(1.0, prune_ratio))
        
        # linear speedup approximation
        slope_scale =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0424, 0.0801, 0.0847, 0.1561]
        intercept_scale = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0165, -0.0024, -0.0287, -0.0333]
        skip_scale =      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1043, 0.1646, 0.1250, 0.2371]

        if prune_ratio <= 0.0: # no pruning overhead for current scale
            speed_score = 0.0
        elif prune_ratio >= 1.0: # skipping current scale
            speed_score = skip_scale[current_scale]
        else: # pruning current scale
            speed_score = prune_ratio * slope_scale[current_scale] + intercept_scale[current_scale]

        if current_scale == len(self.scale_schedule) - 1:
            done = True
        else:
            done = False

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            if self.vae_type == 0 or not getattr(self.infinity_test, "use_bit_label", True):
                raise NotImplementedError(
                    "per_scale_infer=True is currently supported only for vae_type != 0 "
                    "and use_bit_label == True."
                )

            print("="*100)
            print(f"Scale index: {current_scale}, Prune ratio: {prune_ratio:.4f}")
            with torch.no_grad():
                codes, summed_codes, img, state = self.infinity_test.infer_pruned_per_scale(
                    vae=self.vae,
                    scale_schedule=self.scale_schedule,
                    label_B_or_BLT=self.label_B_or_BLT,
                    scale_ind=current_scale,
                    prune_ratio=prune_ratio,
                    B=self.B,
                    negative_label_B_or_BLT=self.negative_label_B_or_BLT,
                    # Use the prompt-based base seed so that golden and pruned
                    # images share the same initial noise at scale 0.
                    g_seed=self.base_seed if current_scale == 0 else None,
                    cfg_list=self.cfg_list,
                    tau_list=self.tau_list,
                    cfg_sc=self.cfg_sc,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    returns_vemb=self.returns_vemb,
                    cfg_insertion_layer=self.cfg_insertion_layer,
                    vae_type=self.vae_type,
                    trunk_scale=self.trunk_scale,
                    gt_leak=self.gt_leak if self.gt_leak >= 0 else 0,
                    gt_ls_Bl=self.gt_ls_Bl,
                    inference_mode=True,
                    sampling_per_bits=self.sampling_per_bits,
                    save_intermediate_results=self.save_intermediate_results,
                    save_dir=self.save_dir,
                    state=self.state,
                )

        self.state = state
        # Move to next scale for the following step
        self.scale_index = current_scale + 1
        
        # Compute similarity (DINOv3 cosine similarity) against golden image
        # for this scale (if available) and derive a quality_score.
        quality_score = 0.0
        psnr_val = 0.0
        dinov3_val = 0.0

        if isinstance(img, torch.Tensor) and current_scale in self._golden_imgs:
            # Convert pruned image from BGR uint8 to RGB float in [0, 1]
            pruned_bgr = img[0].cpu().numpy()                     # uint8 BGR
            pruned_rgb = cv2.cvtColor(pruned_bgr, cv2.COLOR_BGR2RGB)
            pruned_img = torch.from_numpy(pruned_rgb).float() / 255.0  # [H, W, C], RGB

            golden_img = self._golden_imgs[current_scale]         # RGB float in [0,1]
            # Resize to match if needed
            if pruned_img.shape != golden_img.shape:
                pruned_img_resized = torch.nn.functional.interpolate(
                    pruned_img.permute(2, 0, 1).unsqueeze(0),
                    size=golden_img.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)
            else:
                pruned_img_resized = pruned_img

            metrics = self.get_similarity(
                pruned_img_resized,
                golden_img,
                scale_index=current_scale,
                skip_first_N_scales=self.skip_first_N_scales,
            )
            psnr_val = metrics["psnr"]
            dinov3_val = metrics["dinov3"]
            quality_score = metrics["similarity"]

            # Optional debug dump at the last scale
            if current_scale == len(self.scale_schedule) - 1:
                # create save_dir if not exists
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                # Save pruned image as BGR uint8
                cv2.imwrite(os.path.join(self.save_dir, f"pruned_img_{current_scale}.png"), pruned_bgr)

                # Save golden image as BGR uint8 (convert from stored RGB [0,1])
                golden_img_uint8 = (golden_img * 255.0).clamp(0, 255).byte().numpy()
                golden_img_bgr = cv2.cvtColor(golden_img_uint8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.save_dir, f"golden_img_{current_scale}.png"), golden_img_bgr)
                print(
                    f"quality_score (similarity): {quality_score:.2f}, "
                    f"PSNR: {psnr_val:.2f}, DINOv3: {dinov3_val:.4f}"
                )

        quality_reward = self.quality_func(quality_score)
        speed_reward = self.beta * speed_score
        # increase alpha for stronger speed reward
        # increase beta for weaker speed reward
        reward = quality_reward * ( (1-self.alpha) + self.alpha * speed_reward) + (self.alpha * quality_reward if current_scale == len(self.scale_schedule) - 1 else 0.0)

        # Build observation: resize summed_codes with same interpolation mode as Infinity,
        # then append a scale-index channel. Everything is moved to CPU before
        # converting to NumPy for SB3.
        if isinstance(summed_codes, torch.Tensor):
            # Match Infinity's interpolation mode when resizing codes
            interp_mode = getattr(
                getattr(self.vae, "quantizer", None),
                "z_interplote_up",
                "area",
            )
            resized = torch.nn.functional.interpolate(
                summed_codes,
                size=(1, self.obs_H, self.obs_W),
                mode=interp_mode,
            )  # [B, d, 1, H, W]
            codes_4d = resized.squeeze(-3)[0]  # [d, H, W]
            codes_resized = codes_4d.cpu().float()
        else:
            codes_resized = torch.zeros(self.obs_channels - 1, self.obs_H, self.obs_W, dtype=torch.float32)
        # Scale-index channel (normalized), on same device/dtype as codes_resized
        num_scales = len(self.scale_schedule)
        norm_scale = float(current_scale / max(num_scales - 1, 1))
        scale_plane = torch.full(
            (1, self.obs_H, self.obs_W),
            norm_scale,
            dtype=codes_resized.dtype,
            device=codes_resized.device,
        )
        # try:
        #     print("Pre_Obs:", self.pre_obs.shape)
        # except:
        #     print("Pre_Obs: None")
        if(self.pre_obs is not None):
            full_obs = torch.cat([torch.from_numpy(self.pre_obs[0:32,:,:]), codes_resized, scale_plane], dim=0)
        else:
            full_obs = torch.cat([0*codes_resized, codes_resized, scale_plane], dim=0)  # [C+1, H, W]
        
        obs = full_obs.numpy().astype(np.float32)
        
        info = {
            "scale_index": current_scale,
            "prune_ratio": prune_ratio,
            "quality_score": quality_score,
            "speed_score": speed_score,
            "quality_reward": quality_reward,
            "speed_reward": speed_reward,
            "total_reward": reward,
            "psnr": psnr_val,
            "dinov3": dinov3_val,
            "alpha": self.alpha,
            "reward_components": {
                "alpha_quality": quality_reward,
                "beta_speed": speed_reward,
            },
        }
        self.pre_obs = obs
        print(f"quality_reward: {quality_reward:.2f}, speed_reward: {speed_reward:.2f}, reward: {reward:.2f}")
        # Return observation, reward, done, truncate and info dict
        return obs, reward, done, False, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        reset_prune_print_cache()
        self.scale_index = 0
        self.state = None
        self.pre_obs = None
        self.seed(seed=seed)
        # Select prompt for this episode.
        if self._first_reset:
            self._first_reset = False
            if self.prompt_idx < 0:
                self.prompt_idx = 0
        else:
            # Rotate to a new prompt each episode (round-robin over prompt pool).
            if len(self.prompts) > 1:
                self.prompt_idx = (self.prompt_idx + 1) % len(self.prompts)
        self._set_prompt(self.prompts[self.prompt_idx])

        # Ensure golden (unpruned) images are available for quality_score
        # Use self.base_seed to ensure golden images match the pruned generation's seed
        self._ensure_golden_images(
            g_seed=self.base_seed,
            cfg_list=self.cfg_list,
            tau_list=self.tau_list,
            cfg_sc=self.cfg_sc,
            top_k=self.top_k,
            top_p=self.top_p,
            cfg_insertion_layer=self.cfg_insertion_layer,
            vae_type=self.vae_type,
            trunk_scale=self.trunk_scale,
            gt_leak=self.gt_leak,
            gt_ls_Bl=self.gt_ls_Bl,
            sampling_per_bits=self.sampling_per_bits,
        )

        # Skip the first skip_first_N_scales by running fixed inference
        summed_codes = None
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            for si in range(self.skip_first_N_scales):
                codes, summed_codes, img, state = self.infinity_test.infer_pruned_per_scale(
                    vae=self.vae,
                    scale_schedule=self.scale_schedule,
                    label_B_or_BLT=self.label_B_or_BLT,
                    scale_ind=si,
                    prune_ratio=0.0,
                    B=self.B,
                    negative_label_B_or_BLT=self.negative_label_B_or_BLT,
                    g_seed=self.base_seed if si == 0 else None,
                    cfg_list=self.cfg_list,
                    tau_list=self.tau_list,
                    cfg_sc=self.cfg_sc,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    returns_vemb=self.returns_vemb,
                    cfg_insertion_layer=self.cfg_insertion_layer,
                    vae_type=self.vae_type,
                    trunk_scale=self.trunk_scale,
                    gt_leak=self.gt_leak if self.gt_leak >= 0 else 0,
                    gt_ls_Bl=self.gt_ls_Bl,
                    inference_mode=True,
                    sampling_per_bits=self.sampling_per_bits,
                    save_intermediate_results=self.save_intermediate_results,
                    save_dir=self.save_dir,
                    state=self.state,
                )
                self.state = state
                self.scale_index = si + 1

        # Construct initial observation from the result of the last skipped scale
        if isinstance(summed_codes, torch.Tensor):
            # Match Infinity's interpolation mode when resizing codes
            interp_mode = getattr(
                getattr(self.vae, "quantizer", None),
                "z_interplote_up",
                "area",
            )
            resized = torch.nn.functional.interpolate(
                summed_codes,
                size=(1, self.obs_H, self.obs_W),
                mode=interp_mode,
            )  # [B, d, 1, H, W]
            codes_resized = resized.squeeze(-3)[0].cpu().float()
        else:
            codes_resized = torch.zeros(self.obs_channels - 1, self.obs_H, self.obs_W, dtype=torch.float32)

        # Scale-index channel (normalized) for the last skipped scale
        num_scales = len(self.scale_schedule)
        norm_scale = float((self.scale_index - 1) / max(num_scales - 1, 1))
        scale_plane = torch.full(
            (1, self.obs_H, self.obs_W),
            norm_scale,
            dtype=codes_resized.dtype,
        )
        
        # Following the original logic where first 32 channels are zeros if pre_obs was None
        full_obs = torch.cat([0 * codes_resized, codes_resized, scale_plane], dim=0)
        obs = full_obs.numpy().astype(np.float32)
        self.pre_obs = obs
        
        return obs, {}
    