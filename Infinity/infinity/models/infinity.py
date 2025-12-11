"""
Definition of Infinity transformer model.
"""

import math
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt

# from torch.nn.attention.flex_attention import flex_attention
try:
    from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
    from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
except:
    flash_attn_func, flash_attn_varlen_kvpacked_func = None, None
import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import (
    flash_fused_op_installed,
    AdaLNBeforeHead,
    CrossAttnBlock,
    SelfAttnBlock,
    CrossAttention,
    FastRMSNorm,
    precompute_rope2d_freqs_grid,
)
from infinity.models.fastvar_basic import FastVARCrossAttnBlock

from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

import cv2
import torchvision.transforms as transforms


# ---------------- Fourier helpers ---------------- #

def shift(x: torch.Tensor) -> torch.Tensor:
    """fftshift for 4D tensor [B, C, H, W]."""
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(h // 2, w // 2), dims=(2, 3))


def radial_profile(latent: torch.Tensor) -> torch.Tensor:
    """
    Compute azimuthally averaged (radial) Δ log-amplitude of a 2D spectrum.

    latent: [H, W] log-magnitude spectrum with DC at the center.
    Returns:
        1D tensor [N_r] = Δ log amplitude from low→high frequency.
    """
    H, W = latent.shape
    cy, cx = H // 2, W // 2

    yy, xx = torch.meshgrid(
        torch.arange(H, device=latent.device),
        torch.arange(W, device=latent.device),
        indexing="ij",
    )
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # radius from center
    r_flat = r.view(-1)
    vals = latent.view(-1)

    r_int = r_flat.long()
    max_r = int(r_int.max().item())

    hist_sum = torch.zeros(max_r + 1, device=latent.device)
    hist_cnt = torch.zeros(max_r + 1, device=latent.device)

    hist_sum.index_add_(0, r_int, vals)
    hist_cnt.index_add_(0, r_int, torch.ones_like(vals))

    radial_mean = hist_sum / (hist_cnt + 1e-8)

    # Keep only center→Nyquist
    radial_mean = radial_mean[: max_r // 2 + 1]

    # Δ log amplitude, relative to DC
    radial_mean = radial_mean - radial_mean[0]
    return radial_mean


def fourior_plot(x: torch.Tensor):
    """
    Debug helper: print 1D Δ log-amplitude radial spectrum of an intermediate feature.

    x: [B, C, T, H, W] or [B, C, H, W]. If 5D, we use the first frame T=0.
    """
    if x.dim() == 5:
        x = x[:, :, 0, :, :]  # [B, C, H, W]
    elif x.dim() != 4:
        raise ValueError(f"Expected x to be 4D or 5D, got shape {x.shape}")

    B, C, H, W = x.shape

    # FFT over spatial dims
    f = torch.fft.fft2(x)                      # [B, C, H, W], complex
    f = f.abs().clamp_min(1e-6).log()         # log-magnitude

    f_shift = shift(f)                         # DC at center
    latent = f_shift.mean(dim=(0, 1))          # [H, W]

    spec_1d = radial_profile(latent)           # [N_r]
    print(list(spec_1d.cpu().numpy()))


def save_intermediate_results_func(summed_codes, vae, scale_index, save_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    img = vae.decode(summed_codes.squeeze(-3)).detach().cpu()  # [B, 3, H, W]
    img = (img + 1) / 2
    img = (
        img.permute(0, 2, 3, 1)   # [B, H, W, 3]
        .mul_(255)
        .to(torch.uint8)
        .flip(dims=(3,))          # RGB -> BGR for cv2
    )
    generated_image = img[0]
    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{save_dir}/intermediate_scale_img_{scale_index}.png"
    cv2.imwrite(save_file, generated_image.cpu().numpy())
    print(f"Saved intermediate scale image to {save_file}")

    # Also return a 256x256 tensor for potential debug use
    generated_image = generated_image.cpu().numpy()
    generated_image = transform(generated_image[:, :, ::-1])  # back to RGB
    return generated_image


def dft_results(
    codes: torch.Tensor,
    vae,
    vae_scale_schedule,
    scale_index: int,
    save_dir: str,
    mode: str = "pixel",
    resize_to: int = 256,
):
    """
    Compute 1D Δ log-amplitude *radial* spectrum for a given scale and save it.

    Args:
        codes:
            - If mode == "pixel": latent codes for VAE,
              shape [B, D, 1, H, W] or [B, D, H, W].
            - If mode == "latent": feature map,
              shape [B, C, 1, H, W] or [B, C, H, W].
        vae:              VAE model with .decode().
        vae_scale_schedule: list of (t, h, w) for the VAE scales;
                            only the last entry is used for pixel mode.
        scale_index:      index of current scale (for file naming).
        save_dir:         directory to save the 1D spectrum (.npy).
        mode:             "pixel" or "latent".
        resize_to:        (currently unused; kept for future consistency).
    Returns:
        spec_1d: numpy array [N_freq] containing Δ log-amplitude (radial), low→high.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _compute_delta_log_amp(x_4d: torch.Tensor) -> np.ndarray:
        """
        x_4d: [B, C, H, W] in spatial domain.
        Returns:
            1D numpy array of Δ log-amplitude (radial average).
        """
        # make absolutely sure we are in float32 for FFT
        x_4d = x_4d.to(torch.float32)

        f = torch.fft.fft2(x_4d)                      # [B, C, H, W], complex
        f = f.abs().clamp_min(1e-6).log()            # log-magnitude

        f_shift = shift(f)                           # [B, C, H, W], DC at center
        latent = f_shift.mean(dim=(0, 1))            # [H, W]

        spec_1d_ = radial_profile(latent)            # [N_r]
        return spec_1d_.cpu().numpy()

    with torch.no_grad():
        # turn OFF CUDA autocast here, so nothing is in bfloat16 during FFT
        with torch.amp.autocast("cuda", enabled=False):
            if mode == "pixel":
                # Normalize to 5D [B, D, 1, H, W]
                if codes.dim() == 4:  # [B, D, H, W] -> add dummy t=1
                    codes_5d = codes.unsqueeze(2)
                elif codes.dim() == 5:
                    codes_5d = codes
                else:
                    raise ValueError(
                        f"[dft_results/pixel] Expected codes to be 4D or 5D, got {codes.shape}"
                    )

                # Upsample latent to final VAEs spatial/temporal scale (t, H, W)
                target_t, target_h, target_w = vae_scale_schedule[-1]
                codes_5d = F.interpolate(
                    codes_5d,
                    size=(target_t, target_h, target_w),
                    mode=vae.quantizer.z_interplote_up,
                )

                # Decode to pixels: [B, 3, H, W] in float32
                x = vae.decode(codes_5d.squeeze(2))
                x = x.to(torch.float32)
                x = (x + 1) / 2                      # [-1,1] -> [0,1]

                # Convert to grayscale for 2D FFT
                x = x.mean(dim=1, keepdim=True)      # [B, 1, H, W]
                x_4d = x
            elif mode == "latent":
                # No extra rescale here: we want the spectrum at the *current* latent resolution.
                if codes.dim() == 5:                 # [B, C, 1, H, W] or [B, C, T, H, W]
                    x_4d = codes.squeeze(2)          # [B, C, H, W] if t=1
                elif codes.dim() == 4:               # [B, C, H, W]
                    x_4d = codes
                else:
                    raise ValueError(
                        f"[dft_results/latent] Expected codes to be 4D or 5D, got {codes.shape}"
                    )
                x_4d = x_4d.to(torch.float32)
            else:
                raise ValueError(f"Unknown mode '{mode}', use 'pixel' or 'latent'.")

            spec_1d = _compute_delta_log_amp(x_4d)

    mode_tag = "pix" if mode == "pixel" else "lat"
    out_path = os.path.join(save_dir, f"dft_scale_{scale_index:02d}_{mode_tag}.npy")
    np.save(out_path, spec_1d)
    print(f"[DFT] Saved 1D Δ log-amplitude spectrum to: {out_path}")
    print(f"[DFT] scale {scale_index}, mode={mode}, spectrum (first 8 vals): {spec_1d[:8]}")

    return spec_1d



def plot_fourier_scales(
    npy_paths,
    labels=None,
    save_path=None,
    title="Fourier Analysis of Large-Scale Steps"
):
    """
    Plot stacked Δ log-amplitude spectra over frequency for multiple scales.

    Args:
        npy_paths: list of paths to .npy files, each storing a 1D spectrum
                   (Δ log-amplitude from low→high freq).
                   The last path is treated as the latest step (Step K).
        labels:    list of curve labels, same length as npy_paths.
                   If None, defaults to Step 1, Step 2, ...
        save_path: where to save the figure (PNG). If None, just shows.
        title:     figure title.
    """
    if len(npy_paths) == 0:
        print("[plot_fourier_scales] No npy_paths provided, skip.")
        return

    # Load curves
    curves = [np.load(p) for p in npy_paths]

    # Ensure same length (crop to min length)
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    # Frequency axis: normalize [0, 1] → [0, π]
    x = np.linspace(0.0, 1.0, min_len)

    # Default labels if not provided
    if labels is None or len(labels) != len(curves):
        labels = [f"Step {i+1}" for i in range(len(curves))]

    # Plot: older steps lighter, Step K darker (like FastVAR fig)
    cmap = plt.cm.Blues
    num = len(curves)

    plt.figure(figsize=(4.5, 3.5), dpi=150)
    ax = plt.gca()

    for i, (curve, label) in enumerate(zip(curves, labels)):
        # i goes 0..num-1, we want last one darkest
        color = cmap(0.3 + 0.6 * i / max(num - 1, 1))
        lw = 1.5 + 0.5 * i / max(num - 1, 1)
        ax.plot(x * np.pi, curve, label=label, color=color, linewidth=lw)

    # Axes labels & grid
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Δ Log Amplitude")
    ax.set_title(title)

    # x-ticks: 0, 0.17π, 0.33π, 0.50π, 0.67π, 0.83π, 1.00π
    xticks = np.array([0.00, 0.17, 0.33, 0.50, 0.67, 0.83, 1.00]) * np.pi
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.2f}π" for v in [0.00, 0.17, 0.33, 0.50, 0.67, 0.83, 1.00]])

    # Horizontal zero line like in the paper
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    # Legend on the right side
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[FOURIER PLOT] Saved figure to: {save_path}")
        plt.close()
    else:
        plt.show()

        

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h

class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        prune_scale_list=None,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.prune_scale_list = prune_scale_list
        self._prune_skip_logged = set()
        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        flash_attn_has_code = flash_attn_func is not None and hasattr(flash_attn_func, '__code__')
        if flash_attn_has_code:
            customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        else:
            customized_kernel_installed = False
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            if flash_attn_has_code:
                file_path = inspect.getsourcefile(flash_attn_func)
                line_number = inspect.getsourcelines(flash_attn_func)[1]
                varnames_info = f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            else:
                file_path = 'N/A'
                line_number = 'N/A'
                varnames_info = ''
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'{varnames_info}'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5 ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(f'======= WARNING: selecting_idx not specified, set to 1/{num_classes} @ {dist.get_device()} =======')
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                #fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (FastVARCrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                prune_scale_list=self.prune_scale_list,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash if self.t2i else b.attn.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
    

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "infinity_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
            return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        SelfAttnBlock.forward, FastVARCrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def infer_pruned_per_scale(
        self,
        vae,
        scale_schedule,
        label_B_or_BLT,
        scale_ind: int,
        prune_ratio=None,
        # Core config (mirrors a subset of autoregressive_infer_cfg)
        B: int = 1,
        negative_label_B_or_BLT=None,
        g_seed=None,
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
        inference_mode: bool = False,
        sampling_per_bits: int = 1,
        save_intermediate_results: bool = False,
        save_dir: str = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/results/gen_images_v3",
        # Stateful handle carried across scales; pass None on the first call.
        state: dict | None = None,
        # Decode uint8 image; set False when you only need latent codes and will decode once at the end.
        decode_img: bool = True,
    ):
        """
        Step-wise version of `autoregressive_infer_cfg` that runs **one scale step**
        and returns `(codes, summed_codes, img, state)`.

        - Call with `state=None` and `scale_ind == 0` to start a new generation run.
        - Re-use the returned `state` when calling for the next `scale_ind`.
        - If you iterate over all scales in order with fixed arguments, the
          final `summed_codes` and `img` will match `autoregressive_infer_cfg`
          (for `vae_type != 0`).
        - `prune_ratio` directly controls pruning at the current scale. If
          `None`, it falls back to `self.prune_scale_list` (if defined).
        """

        if vae_type == 0:
            raise NotImplementedError(
                "infer_pruned_per_scale currently supports only vae_type != 0 (VAE path)."
            )

        assert 0 <= scale_ind < len(scale_schedule), f"scale_ind {scale_ind} out of range."

        # Default cfg/tau lists if not provided
        if cfg_list is None:
            cfg_list = [1.0 for _ in range(len(scale_schedule))]
        if tau_list is None:
            tau_list = [1.0 for _ in range(len(scale_schedule))]

        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # ------------------------------------------------------------------ #
        # 1) Initialization on the first scale (equivalent to the top part
        #    of `autoregressive_infer_cfg`)
        # ------------------------------------------------------------------ #
        if state is None:
            if scale_ind != 0:
                raise ValueError("state must be None only for scale_ind == 0")

            # RNG
            if g_seed is None:
                rng = None
            else:
                self.rng.manual_seed(g_seed)
                rng = self.rng

            # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify,
            # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
            if self.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule

            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

            # CFG preparation (same as in autoregressive_infer_cfg)
            cfg_array = np.array(cfg_list)
            if any(cfg_array != 1):
                bs = 2 * B
                if not negative_label_B_or_BLT:
                    kv_compact_un = kv_compact.clone()
                    total = 0
                    for le in lens:
                        kv_compact_un[total : total + le] = (self.cfg_uncond)[:le]
                        total += le
                    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    cu_seqlens_k = torch.cat(
                        (cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0
                    )
                else:
                    kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    cu_seqlens_k = torch.cat(
                        (cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0
                    )
                    max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            else:
                bs = B

            kv_compact = self.text_norm(kv_compact)
            sos = cond_BD = self.text_proj_for_sos(
                (kv_compact, cu_seqlens_k, max_seqlen_k)
            )  # sos shape: [2, 4096]
            kv_compact = self.text_proj_for_ca(kv_compact)  # kv_compact shape: [10, 2048]
            ca_kv = (kv_compact, cu_seqlens_k, max_seqlen_k)
            last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(
                bs, 1, -1
            )

            with torch.amp.autocast("cuda", enabled=False):
                cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()

            accu_BChw, cur_L, ret = None, 0, []
            idx_Bl_list, idx_Bld_list = [], []

            # Enable KV caching (same logic as autoregressive_infer_cfg)
            if inference_mode:
                for b in self.unregistered_blocks:
                    (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(
                        True
                    )
            else:
                assert self.num_block_chunks > 1
                for block_chunk_ in self.block_chunks:
                    for module in block_chunk_.module.module:
                        (
                            module.sa
                            if isinstance(module, FastVARCrossAttnBlock)
                            else module.attn
                        ).kv_caching(True)

            abs_cfg_insertion_layers = []
            add_cfg_on_logits, add_cfg_on_probs = False, False
            leng = len(self.unregistered_blocks)
            for item in cfg_insertion_layer:
                if item == 0:  # add cfg on logits
                    add_cfg_on_logits = True
                elif item == 1:  # add cfg on probs
                    add_cfg_on_probs = True
                elif item < 0:  # determine to add cfg at item-th layer's output
                    assert (
                        leng + item > 0
                    ), f"cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}"
                    abs_cfg_insertion_layers.append(leng + item)
                else:
                    raise ValueError(f"cfg_insertion_layer: {item} is not valid")

            num_stages_minus_1 = len(scale_schedule) - 1
            summed_codes = 0

            state = dict(
                rng=rng,
                vae_scale_schedule=vae_scale_schedule,
                kv_compact=kv_compact,
                lens=lens,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
                bs=bs,
                ca_kv=ca_kv,
                sos=sos,
                cond_BD=cond_BD,
                cond_BD_or_gss=cond_BD_or_gss,
                last_stage=last_stage,
                accu_BChw=accu_BChw,
                cur_L=cur_L,
                ret=ret,
                idx_Bl_list=idx_Bl_list,
                idx_Bld_list=idx_Bld_list,
                abs_cfg_insertion_layers=abs_cfg_insertion_layers,
                add_cfg_on_logits=add_cfg_on_logits,
                add_cfg_on_probs=add_cfg_on_probs,
                num_stages_minus_1=num_stages_minus_1,
                summed_codes=summed_codes,
                kv_caching_enabled=True,
            )
        else:
            # Recover state from previous step
            rng = state["rng"]
            vae_scale_schedule = state["vae_scale_schedule"]
            bs = state["bs"]
            ca_kv = state["ca_kv"]
            cond_BD = state["cond_BD"]
            cond_BD_or_gss = state["cond_BD_or_gss"]
            last_stage = state["last_stage"]
            accu_BChw = state["accu_BChw"]
            cur_L = state["cur_L"]
            ret = state["ret"]
            idx_Bl_list = state["idx_Bl_list"]
            idx_Bld_list = state["idx_Bld_list"]
            abs_cfg_insertion_layers = state["abs_cfg_insertion_layers"]
            add_cfg_on_logits = state["add_cfg_on_logits"]
            add_cfg_on_probs = state["add_cfg_on_probs"]
            num_stages_minus_1 = state["num_stages_minus_1"]
            summed_codes = state["summed_codes"]

        # ------------------------------------------------------------------ #
        # 2) One-scale body (equivalent to a single iteration of the loop in
        #    `autoregressive_infer_cfg`)
        # ------------------------------------------------------------------ #
        pn = scale_schedule[scale_ind]

        # Respect trunk_scale limit (same semantics as `break` in the loop)
        if scale_ind >= trunk_scale:
            # Do nothing for this and further scales.
            img = None
            if summed_codes is not None and not isinstance(summed_codes, int):
                img = vae.decode(summed_codes.squeeze(-3))
                img = (img + 1) / 2
                img = (
                    img.permute(0, 2, 3, 1)
                    .mul_(255)
                    .to(torch.uint8)
                    .flip(dims=(3,))
                )
            return None, summed_codes, img, state

        # Dynamic pruning ratio control.
        #
        # We support two semantics:
        # - Stage-level skipping when the *effective* prune ratio >= 1.0
        #   (mirrors `autoregressive_infer_cfg` but keeps sequence lengths
        #   consistent by propagating `summed_codes` to the next scale).
        # - Token-level pruning when the ratio is in [0, 1), by injecting it
        #   into `self.prune_scale_list` so that `compute_merge` uses it.

        # 1) Determine an effective ratio for this scale, preferring the
        #    per-call argument and falling back to the model config.
        ratio_val: Optional[float] = None
        if prune_ratio is not None:
            ratio_val = float(prune_ratio)
        elif getattr(self, "prune_scale_list", None) is not None and isinstance(
            self.prune_scale_list, dict
        ):
            cfg_ratio = self.prune_scale_list.get(pn[2])
            if cfg_ratio is not None:
                ratio_val = float(cfg_ratio)

        # 2) Stage-level skip when ratio >= 1.0
        if ratio_val is not None and ratio_val >= 1.0:
            total_tokens = pn[1] * pn[2]
            print(
                f"[{pn[1]}x{pn[2]}] Pruning ratio = {ratio_val * 100:.0f}%, Remaining Tokens: 0/{total_tokens} (skipping stage)"
            )
            # No new codes are added at this scale. For non-final scales we still
            # need to propagate `summed_codes` to the next spatial resolution so
            # that `last_stage` has the expected sequence length and channel dim.
            if scale_ind != num_stages_minus_1 and isinstance(summed_codes, torch.Tensor):
                next_scale = vae_scale_schedule[scale_ind + 1]
                last_stage = F.interpolate(
                    summed_codes,
                    size=next_scale,
                    mode=vae.quantizer.z_interplote_up,
                )  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                last_stage = last_stage.squeeze(-3)  # [B, d, h, w] or [B, d, 2h, 2w]
                if self.apply_spatial_patchify:  # patchify operation
                    last_stage = torch.nn.functional.pixel_unshuffle(
                        last_stage, 2
                    )  # [B, 4d, h, w]
                last_stage = last_stage.reshape(
                    *last_stage.shape[:2], -1
                )  # [B, d, h*w] or [B, 4d, h*w]
                last_stage = torch.permute(
                    last_stage, [0, 2, 1]
                )  # [B, h*w, d] or [B, h*w, 4d]

                # Project from VAE space into transformer embedding space and
                # duplicate for CFG batches, mirroring the non-skip path.
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs // B, 1, 1)

            # Update state and decode image from the current `summed_codes`.
            state["last_stage"] = last_stage
            state["summed_codes"] = summed_codes

            img = None
            if decode_img and isinstance(summed_codes, torch.Tensor):
                img = vae.decode(summed_codes.squeeze(-3))
                img = (img + 1) / 2
                img = (
                    img.permute(0, 2, 3, 1)
                    .mul_(255)
                    .to(torch.uint8)
                    .flip(dims=(3,))
                )

            return None, summed_codes, img, state

        # 3) Token-level pruning when ratio in [0, 1)
        if ratio_val is not None and 0.0 <= ratio_val < 1.0:
            if self.prune_scale_list is None or not isinstance(self.prune_scale_list, dict):
                self.prune_scale_list = {}
            self.prune_scale_list[pn[2]] = ratio_val
            # Ensure all backbone blocks see the same dict reference.
            for b in self.unregistered_blocks:
                b.prune_scale_list = self.prune_scale_list

        cfg = cfg_list[scale_ind]
        cur_L += np.array(pn).prod()

        need_to_pad = 0
        attn_fn = None
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[: (scale_ind + 1)]), None)

        layer_idx = 0
        for block_idx, b in enumerate(self.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if self.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = self.add_lvl_embeding(
                    last_stage, scale_ind, scale_schedule, need_to_pad=need_to_pad
                )
            if not self.add_lvl_embeding_only_first_block:
                last_stage = self.add_lvl_embeding(
                    last_stage, scale_ind, scale_schedule, need_to_pad=need_to_pad
                )

            for m in b.module:
                last_stage = m(
                    x=last_stage,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=None,
                    attn_fn=attn_fn,
                    scale_schedule=scale_schedule,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                    scale_ind=scale_ind,
                    layer_idx=layer_idx,
                    x_shape=pn,
                )
                if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                    last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                    last_stage = torch.cat((last_stage, last_stage), 0)
                layer_idx += 1

        # Logits & sampling (only bit-label + VAE path is supported here)
        if (cfg != 1) and state["add_cfg_on_logits"]:
            logits_BlV = self.get_logits(last_stage, cond_BD).mul(1 / tau_list[scale_ind])
            logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
        else:
            logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(
                1 / tau_list[scale_ind]
            )

        if self.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV,
                rng=rng,
                top_k=top_k or self.top_k,
                top_p=top_p or self.top_p,
                num_samples=1,
            )[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            raise NotImplementedError(
                "infer_pruned_per_scale is currently implemented for `use_bit_label=True` only."
            )

        assert returns_vemb, "VAE path in infer_pruned_per_scale expects returns_vemb != 0"

        if scale_ind < gt_leak:
            idx_Bld = gt_ls_Bl[scale_ind]
        else:
            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
            if self.apply_spatial_patchify:  # unpatchify operation
                idx_Bld = idx_Bld.permute(0, 3, 1, 2)  # [B, 4d, h, w]
                idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)  # [B, d, 2h, 2w]
                idx_Bld = idx_Bld.permute(0, 2, 3, 1)  # [B, 2h, 2w, d]
            idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

        idx_Bld_list.append(idx_Bld)
        codes = vae.quantizer.lfq.indices_to_codes(
            idx_Bld, label_type="bit_label"
        )  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]

        if scale_ind != num_stages_minus_1:
            rescale_codes = F.interpolate(
                codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up
            )
            summed_codes = summed_codes + rescale_codes
            last_stage = F.interpolate(
                summed_codes,
                size=vae_scale_schedule[scale_ind + 1],
                mode=vae.quantizer.z_interplote_up,
            )  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(
                -3
            )  # [B, d, h, w] or [B, d, 2h, 2w]
            if self.apply_spatial_patchify:  # patchify operation
                last_stage = torch.nn.functional.pixel_unshuffle(
                    last_stage, 2
                )  # [B, 4d, h, w]
            last_stage = last_stage.reshape(
                *last_stage.shape[:2], -1
            )  # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(
                last_stage, [0, 2, 1]
            )  # [B, h*w, d] or [B, h*w, 4d]
        else:
            rescale_codes = codes
            summed_codes = summed_codes + codes

        if save_intermediate_results:
            save_intermediate_results_func(
                summed_codes, vae, scale_ind, f"{save_dir}/summed_codes"
            )
            # save_intermediate_results_func(
            #     rescale_codes, vae, scale_ind, f"{save_dir}/rescale_codes"
            # )

            # pixel-space spectra
            # dft_results(
            #     summed_codes,
            #     vae,
            #     vae_scale_schedule,
            #     scale_index=scale_ind,
            #     save_dir=f"{save_dir}/fourier_pixel_summed",
            #     mode="pixel",
            # )
            # dft_results(
            #     rescale_codes,
            #     vae,
            #     vae_scale_schedule,
            #     scale_index=scale_ind,
            #     save_dir=f"{save_dir}/fourier_pixel_residual",
            #     mode="pixel",
            # )

            # # latent-space spectra
            # dft_results(
            #     summed_codes,
            #     vae,
            #     vae_scale_schedule,
            #     scale_index=scale_ind,
            #     save_dir=f"{save_dir}/fourier_latent_summed",
            #     mode="latent",
            # )
            # dft_results(
            #     rescale_codes,
            #     vae,
            #     vae_scale_schedule,
            #     scale_index=scale_ind,
            #     save_dir=f"{save_dir}/fourier_latent_residual",
            #     mode="latent",
            # )
            # dft_results(
            #     codes,
            #     vae,
            #     vae_scale_schedule,
            #     scale_index=scale_ind,
            #     save_dir=f"{save_dir}/fourier_latent_unscaled",
            #     mode="latent",
            # )

        # Prepare last_stage for the next scale if not the last one
        if scale_ind != num_stages_minus_1:
            last_stage = self.word_embed(self.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs // B, 1, 1)

        # ------------------------------------------------------------------ #
        # 3) Optionally disable KV caching if this was the last scale
        # ------------------------------------------------------------------ #
        if scale_ind == num_stages_minus_1 and state.get("kv_caching_enabled", False):
            if inference_mode:
                for b in self.unregistered_blocks:
                    (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(
                        False
                    )
            else:
                assert self.num_block_chunks > 1
                for block_chunk_ in self.block_chunks:
                    for module in block_chunk_.module.module:
                        (
                            module.sa
                            if isinstance(module, FastVARCrossAttnBlock)
                            else module.attn
                        ).kv_caching(False)
            state["kv_caching_enabled"] = False

        # ------------------------------------------------------------------ #
        # 4) Update state and decode image
        # ------------------------------------------------------------------ #
        state["last_stage"] = last_stage
        state["accu_BChw"] = accu_BChw
        state["cur_L"] = cur_L
        state["ret"] = ret
        state["idx_Bl_list"] = idx_Bl_list
        state["idx_Bld_list"] = idx_Bld_list
        state["summed_codes"] = summed_codes

        # Decode current image from summed_codes (VAE path, uint8, B x H x W x C)
        img = None
        if decode_img:
            img = vae.decode(summed_codes.squeeze(-3))
            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))

        return codes, summed_codes, img, state

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        save_intermediate_results=True,
        save_dir="/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity/results/gen_images_v3",
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [10, 2048]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, FastVARCrossAttnBlock) else module.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')


        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0

        for si, pn in enumerate(scale_schedule): # si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
            prune_ratio = None
            if self.prune_scale_list:
                prune_ratio = self.prune_scale_list.get(pn[2])
            if prune_ratio is not None and prune_ratio >= 1.0:
                if pn[2] not in self._prune_skip_logged:
                    total_tokens = pn[1] * pn[2]
                    print(
                        f"[{pn[1]}x{pn[2]}] Pruning ratio = {prune_ratio * 100:.0f}%, Remaining Tokens: 0/{total_tokens} (skipping stage)"
                    )
                    self._prune_skip_logged.add(pn[2])
                continue
 
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # Uncomment these lines to benchmark inference time speedup at different scales:
            # torch.cuda.synchronize()
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()


            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks): # 8
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
                for m in b.module: # 4
                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si,layer_idx=layer_idx,x_shape=pn)
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1

            # Uncomment these lines to benchmark inference time speedup at different scales:
            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_time = start_event.elapsed_time(end_event)
            # print('scale:', pn ,"total forward running time:", int(elapsed_time), "ms")
            # torch.cuda.empty_cache()

            if (cfg != 1) and add_cfg_on_logits:
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    rescale_codes = F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    summed_codes += rescale_codes
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    rescale_codes = codes
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            
                
            if save_intermediate_results:
                save_intermediate_results_func(summed_codes, vae, si, f"{save_dir}/summed_codes")
                # save_intermediate_results_func(rescale_codes, vae, si, f"{save_dir}/rescale_codes")

                # # pixel-space spectra
                # dft_results(
                #     summed_codes, vae, vae_scale_schedule,
                #     scale_index=si,
                #     save_dir=f"{save_dir}/fourier_pixel_summed",
                #     mode="pixel",
                # )
                # dft_results(
                #     rescale_codes, vae, vae_scale_schedule,
                #     scale_index=si,
                #     save_dir=f"{save_dir}/fourier_pixel_residual",
                #     mode="pixel",
                # )

                # # latent-space spectra
                # dft_results(
                #     summed_codes, vae, vae_scale_schedule,
                #     scale_index=si,
                #     save_dir=f"{save_dir}/fourier_latent_summed",
                #     mode="latent",
                # )
                # dft_results(
                #     rescale_codes, vae, vae_scale_schedule,
                #     scale_index=si,
                #     save_dir=f"{save_dir}/fourier_latent_residual",
                #     mode="latent",
                # )
                # dft_results(
                #     codes, vae, vae_scale_schedule,
                #     scale_index=si,
                #     save_dir=f"{save_dir}/fourier_latent_unscaled",
                #     mode="latent",
                # )

            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)


        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, FastVARCrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, FastVARCrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, FastVARCrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
