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
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

from infinity.models.infinity import Infinity
from infinity.models.basic import *
from infinity.models.fastvar_utils import reset_prune_print_cache
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import torchvision.utils as vutils


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
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

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee',
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt



image_num = 0
def get_pruning_ratio(scale: int, num_scales: int) -> float:
    """
    Example pruning schedule:
    - No pruning on earlier scales
    - Stronger pruning on the last few scales
    This function is just a placeholder; in your project you can replace it
    with an RL agent or any other controller.
    """
    prune_scale_list = [0.0] * num_scales
    global image_num
    # Apply pruning only to the last few scales.
    N = min(5, num_scales)
    tail_pattern = [0.0, 0.4, 0.5, 1.0, 1.0][:N]

    # tail_pattern = [0.0, 0.0, 0.0, 0.0, 0.0][:N]
    
    prune_scale_list[-N:] = tail_pattern
    return prune_scale_list[scale]

def gen_one_img(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    save_intermediate_results=False,
    save_dir=None,
    per_scale_infer=False,
):
    sstt = time.time()
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
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        if per_scale_infer:
            # Use step-wise generation via `infer_pruned_per_scale`, iterating all scales.
            # This matches `autoregressive_infer_cfg` behavior (VAE / bit-label path)
            # while allowing dynamic per-scale pruning.

            if vae_type == 0 or not getattr(infinity_test, "use_bit_label", True):
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
                prune_ratio = get_pruning_ratio(si, num_scales)

                codes, summed_codes, img, state = infinity_test.infer_pruned_per_scale(
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
                )

            # One full image (all scales) is done; reset print cache so pruning
            # ratio logs are emitted again for the next image.
            reset_prune_print_cache()

            # `img` is already the decoded uint8 image for the final scale.
            img_list = [img]

        else:
            _, intermidiate_list, img_list = infinity_test.autoregressive_infer_cfg(
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

    return img

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file


def _find_tokenizer_in_hf_cache(repo_like: str):
    """
    Try to locate a cached model dir under ~/.cache/huggingface/hub
    Return a path (str) or None.
    """
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache_root):
        return None
    # normalize search token: e.g. "google/flan-t5-xl" -> "google--flan-t5-xl" or check substring
    token = repo_like.replace("/", "--")
    for name in os.listdir(cache_root):
        # names look like "models--google--flan-t5-xl"
        if token in name or repo_like in name:
            root = os.path.join(cache_root, name)
            # root may have subfolders (commit hashes). search for dir that has tokenizer files.
            for sub in os.listdir(root):
                subdir = os.path.join(root, sub)
                if _dir_has_tokenizer_files(subdir):
                    return subdir
            # sometimes tokenizer files are directly under root
            if _dir_has_tokenizer_files(root):
                return root
    return None

def load_tokenizer(t5_path: str = ""):
    """
    Robust tokenizer + encoder loader.
    Accepts:
      - local folder path (with tokenizer files)
      - huggingface repo id like "google/flan-t5-xl"
      - a local folder that only contains model weights (will try cache or fallback to hub)
    Returns: (text_tokenizer, text_encoder)
    """
    print(f'[Loading tokenizer and text encoder] t5_path={t5_path!r}')
    t5_path = os.path.expanduser(str(t5_path)) if t5_path else ""

    # 1) If an explicit local dir with tokenizer files -> load locally
    if t5_path and os.path.isdir(t5_path) and _dir_has_tokenizer_files(t5_path):
        print(f'Loading tokenizer from local folder: {t5_path}')
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
            t5_path, local_files_only=True, revision=None, legacy=True
        )
        text_tokenizer.model_max_length = 512
        print('Loading encoder from local folder (weights)...')
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            t5_path, torch_dtype=torch.float16, local_files_only=True
        )
        text_encoder.to('cuda')
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        return text_tokenizer, text_encoder

    # 2) Try to find matching tokenizer in HF cache
    if t5_path:
        cached = _find_tokenizer_in_hf_cache(t5_path)
        if cached:
            print(f'Found tokenizer in HF cache: {cached}')
            text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
                cached, local_files_only=True, revision=None, legacy=True
            )
            text_tokenizer.model_max_length = 512
            # For encoder weights, attempt to load from same cached dir; if that fails,
            # fallback to using repo id below.
            try:
                text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
                    cached, torch_dtype=torch.float16, local_files_only=True
                )
                text_encoder.to('cuda')
                text_encoder.eval()
                text_encoder.requires_grad_(False)
                return text_tokenizer, text_encoder
            except Exception as e:
                print(f'Warning: loading encoder from cached dir failed: {e}. Will try repo id below.')

    # 3) Fallback: try to treat t5_path as a repo id or use official id
    # Try the provided t5_path first (may be repo id), then fallback to google/flan-t5-xl.
    tried = []
    for repo_candidate in filter(None, [t5_path, "google/flan-t5-xl"]):
        try:
            print(f'Trying AutoTokenizer.from_pretrained("{repo_candidate}") (may download)...')
            text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(
                repo_candidate, revision=None, legacy=True
            )
            text_tokenizer.model_max_length = 512
            print(f'Loaded tokenizer from {repo_candidate}')
            print(f'Loading encoder from {repo_candidate} (this may download large files)...')
            text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
                repo_candidate, torch_dtype=torch.float16
            )
            text_encoder.to('cuda')
            text_encoder.eval()
            text_encoder.requires_grad_(False)
            return text_tokenizer, text_encoder
        except Exception as e:
            print(f'Failed to load from {repo_candidate!r}: {e}')
            tried.append((repo_candidate, e))

    # If we reach here, nothing worked
    raise RuntimeError(f"Failed to load tokenizer/encoder. Tried: {tried}")


def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        model_kwargs = model_kwargs or {}
        default_prune_cfg = model_kwargs.pop("prune_scale_list", None)

        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            prune_scale_list=default_prune_cfg,
            **model_kwargs,
        ).to(device=device)
        # print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        state_dict = torch.load(model_path, map_location=device)
        print(infinity_test.load_state_dict(state_dict))
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, infinity_input = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    gt_img = (inp[0] + 1) / 2
    gt_img = gt_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices

def load_visual_tokenizer(args, device='cuda'):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args, device='cuda'):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        # assert ('ar-' in model_path) or ('slim-' in model_path)
        # copy large model to local, save slim to local, and copy slim to nas, and load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        slim_model_path = model_path.replace('ar-', 'slim-')
        print(f'load checkpoint from {slim_model_path}')

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    prune_cfg = getattr(args, 'prune_scale_list', None)
    if prune_cfg:
        kwargs_model["prune_scale_list"] = prune_cfg
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
    )
    return infinity

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument(
        '--prune_scales',
        type=str,
        default='',
        help='Comma-separated list of scale:ratio entries (e.g. "32:0.4,40:0.5").'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    args = parser.parse_args()

    if args.prune_scales:
        prune_scale_list = {}
        for entry in args.prune_scales.split(','):
            entry = entry.strip()
            if not entry:
                continue
            try:
                scale_str, ratio_str = entry.split(':')
                prune_scale_list[int(scale_str)] = float(ratio_str)
            except ValueError as exc:
                raise ValueError(f"Invalid prune_scales entry '{entry}'. Expected format scale:ratio") from exc
        args.prune_scale_list = prune_scale_list
    else:
        args.prune_scale_list = None

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
