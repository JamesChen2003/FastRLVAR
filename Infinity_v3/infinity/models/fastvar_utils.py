import torch
from typing import Tuple, Callable, Dict, Iterable, Union
import torch
import math
from typing import Type, Any


def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    B, L, c = cur_x.shape
    mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x,(1,1)).permute(0, 2, 3, 1).view(B, 1,c)
    mse_difference = torch.sum((cur_x - mean_x)**2,dim=-1,keepdim=True)
    select_indices = torch.argsort(mse_difference,dim=1,descending=True)
    filted_select_indices=select_indices[:,:num_remain,:]

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x,dim=1,index=filted_select_indices.repeat(1,1,c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        if unmerged_cache_x is None or cached_hw is None:
            base = torch.zeros(B, L, c, device=unmerged_cur_x.device, dtype=unmerged_cur_x.dtype)
        else:
            base = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
            base = torch.nn.functional.interpolate(base, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        base.scatter_(dim=1, index=filted_select_indices.repeat(1,1,c), src=unmerged_cur_x)
        return base

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx

def masked_previous_scale_cache_entropy(cur_x, num_remain, cur_shape, attn_entropy):
    B, L, c = cur_x.shape
    # mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    # mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x,(1,1)).permute(0, 2, 3, 1).view(B, 1,c)
    # mse_difference = torch.sum((cur_x - mean_x)**2,dim=-1,keepdim=True)
    # print("MSE Difference:", mse_difference.shape)
    # print("Attn Entropy:", attn_entropy.shape)
    attn_entropy = torch.mean(attn_entropy, dim=1, keepdim=True).permute(0,2,1)
    select_indices = torch.argsort(attn_entropy,dim=1,descending=True)
    filted_select_indices=select_indices[:,:num_remain,:]

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x,dim=1,index=filted_select_indices.repeat(1,1,c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        if unmerged_cache_x is None or cached_hw is None:
            base = torch.zeros(B, L, c, device=unmerged_cur_x.device, dtype=unmerged_cur_x.dtype)
        else:
            base = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
            base = torch.nn.functional.interpolate(base, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        base.scatter_(dim=1, index=filted_select_indices.repeat(1,1,c), src=unmerged_cur_x)
        return base

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx


def masked_previous_scale_cache_hybrid(cur_x, num_remain, cur_shape, attn_entropy, local_ratio):
    B, L, c = cur_x.shape
    
    # Entropy-style importance
    if attn_entropy.dim() == 3:
        entropy_scores = torch.mean(attn_entropy, dim=1, keepdim=True).permute(0, 2, 1) # (B, L, 1)
    else:
        entropy_scores = attn_entropy.unsqueeze(-1) # (B, L, 1)
        
    # FastVAR-style importance
    mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x, (1, 1)).permute(0, 2, 3, 1).view(B, 1, c)
    mse_difference = torch.sum((cur_x - mean_x)**2, dim=-1, keepdim=True) # (B, L, 1)
    
    num_remain_entropy = int(num_remain * local_ratio)
    num_remain_fastvar = num_remain - num_remain_entropy
    
    # Pick top indices from entropy
    entropy_indices = torch.argsort(entropy_scores, dim=1, descending=True)
    selected_entropy = entropy_indices[:, :num_remain_entropy, :] # (B, k1, 1)
    
    # Mask out already selected tokens from MSE pool
    mse_temp = mse_difference.clone()
    mask = torch.zeros_like(mse_temp)
    mask.scatter_(dim=1, index=selected_entropy, value=1.0)
    mse_temp.masked_fill_(mask > 0.5, -1e9)
    
    # Pick the remaining tokens from FastVAR importance
    remaining_indices = torch.argsort(mse_temp, dim=1, descending=True)
    selected_fastvar = remaining_indices[:, :num_remain_fastvar, :] # (B, k2, 1)
    
    # Combine to get exactly num_remain indices
    filted_select_indices = torch.cat([selected_entropy, selected_fastvar], dim=1) # (B, num_remain, 1)

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x, dim=1, index=filted_select_indices.repeat(1, 1, c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        if unmerged_cache_x is None or cached_hw is None:
            base = torch.zeros(B, L, c, device=unmerged_cur_x.device, dtype=unmerged_cur_x.dtype)
        else:
            base = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
            base = torch.nn.functional.interpolate(base, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        base.scatter_(dim=1, index=filted_select_indices.repeat(1, 1, c), src=unmerged_cur_x)
        return base

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx


DEFAULT_PRUNE_RATIOS: Dict[int, float] = {32: 0.4, 40: 0.5}
_printed_prune_scales: Dict[int, bool] = {}


def reset_prune_print_cache() -> None:
    """
    Reset the cache that tracks which scales have had their pruning ratios
    printed. Useful when generating multiple images so that logs are emitted
    per image instead of only once per process.
    """
    _printed_prune_scales.clear()


def _normalise_prune_config(
    prune_cfg: Union[None, Dict[int, float], Iterable[Union[int, Tuple[int, float]]]]
) -> Dict[int, float]:
    if prune_cfg is None:
        return DEFAULT_PRUNE_RATIOS

    if isinstance(prune_cfg, dict):
        return prune_cfg

    ratios: Dict[int, float] = {}
    for item in prune_cfg:
        if isinstance(item, tuple):
            scale, ratio = item
            ratios[int(scale)] = float(ratio)
        else:
            scale = int(item)
            if scale not in DEFAULT_PRUNE_RATIOS:
                raise ValueError(
                    f"Pruning ratio for scale {scale} is not provided. Supply a (scale, ratio) tuple."
                )
            ratios[scale] = DEFAULT_PRUNE_RATIOS[scale]
    return ratios


# 1/2 : [... (1, 23, 46), (1, 30, 60), (1, 37, 74), (1, 45, 90), (1, 60, 120)]
# 1.333/1  (1, 36, 27), (1, 48, 36), (1, 60, 45), (1, 72, 54) (1,84,63)
# 2/1:  (1, 46, 23), (1, 60, 30), (1, 74, 37), (1, 90, 45) (1,120,60)
# 1/1 , (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64), (1, 84, 84)]
def compute_merge(
    x: torch.Tensor,
    prune_scale_list: Union[None, Dict[int, float], Iterable[Union[int, Tuple[int, float]]]] = None,
    is_later_layer: bool = False,
    x_shape=None,
    override_log_ratio: float = None,
) -> Tuple[Callable, ...]:
    _, original_h, original_w = x_shape

    prune_ratios = _normalise_prune_config(prune_scale_list)
    ratio = prune_ratios.get(original_w)

    if ratio is not None and is_later_layer:
        # Clamp ratio into [0, 1] for safety and ensure we always keep at least
        # one token, otherwise many downstream ops (e.g. rotary embedding) fail
        # on zero-length sequences.
        ratio = float(max(min(ratio, 1.0), 0.0))
        remaining_tokens = int(x.shape[1] * (1.0 - ratio))
        if remaining_tokens <= 0:
            remaining_tokens = 1
        
        log_ratio = override_log_ratio if override_log_ratio is not None else ratio
        if not _printed_prune_scales.get(original_w):
            print(
                f"[{original_h}x{original_w}] Pruning ratio = {log_ratio * 100:.0f}%, "
                f"Remaining Tokens: {remaining_tokens}/{original_h * original_w}"
            )
            _printed_prune_scales[original_w] = True
        m, u, id_fn = masked_previous_scale_cache(x, remaining_tokens, x_shape)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn  # Okay this is probably not very good

def compute_merge_entropy(
    x: torch.Tensor,
    attn_entropy: torch.Tensor = None,  # NEW: Input entropy from previous layer/step
    prune_scale_list: Union[None, Dict[int, float], Iterable[Union[int, Tuple[int, float]]]] = None,
    is_later_layer: bool = False,
    x_shape=None,
    override_log_ratio: float = None,
) -> Tuple[Callable, Callable, torch.Tensor]:
    
    B, N, C = x.shape
    _, original_h, original_w = x_shape

    prune_ratios = _normalise_prune_config(prune_scale_list) 
    ratio = prune_ratios.get(original_w)

    if ratio is not None and is_later_layer and attn_entropy is not None:
        remaining_tokens = int(x.shape[1] * (1.0 - ratio))
        if remaining_tokens <= 0:
            remaining_tokens = 1
        
        log_ratio = override_log_ratio if override_log_ratio is not None else ratio
        if not _printed_prune_scales.get(original_w):
            print(
                f"[{original_h}x{original_w}] Pruning ratio = {log_ratio * 100:.0f}%, "
                f"Remaining Tokens: {remaining_tokens}/{original_h * original_w}"
            )
            _printed_prune_scales[original_w] = True
        m, u, id_fn = masked_previous_scale_cache_entropy(x, remaining_tokens, x_shape, attn_entropy)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn  # Okay this is probably not very good


def compute_merge_hybrid(
    x: torch.Tensor,
    attn_entropy: torch.Tensor = None,
    local_ratio: float = 0.5,
    prune_scale_list: Union[None, Dict[int, float], Iterable[Union[int, Tuple[int, float]]]] = None,
    is_later_layer: bool = False,
    x_shape=None,
    override_log_ratio: float = None,
) -> Tuple[Callable, Callable, torch.Tensor]:
    
    B, N, C = x.shape
    _, original_h, original_w = x_shape

    prune_ratios = _normalise_prune_config(prune_scale_list) 
    ratio = prune_ratios.get(original_w)

    if ratio is not None and is_later_layer and attn_entropy is not None:
        remaining_tokens = int(x.shape[1] * (1.0 - ratio))
        if remaining_tokens <= 0:
            remaining_tokens = 1
        
        log_ratio = override_log_ratio if override_log_ratio is not None else ratio
        if not _printed_prune_scales.get(original_w):
            print(
                f"[{original_h}x{original_w}] Pruning ratio = {log_ratio * 100:.0f}%, "
                f"Remaining Tokens: {remaining_tokens}/{original_h * original_w}"
            )
            _printed_prune_scales[original_w] = True
        m, u, id_fn = masked_previous_scale_cache_hybrid(x, remaining_tokens, x_shape, attn_entropy, local_ratio)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)

    return m, u, id_fn
