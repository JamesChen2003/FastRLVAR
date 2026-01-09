import os
import json
import torch
import numpy as np
import cv2
import argparse
import glob
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List

# Global storage for pruned token indices during a single generation run.
# Keys are (scale_h, scale_w), values are list of (layer_idx, pruned_indices_tensor)
_pruned_data_cache = {}
_current_output_dir = None

def set_pruned_output_dir(output_dir: str):
    global _current_output_dir
    _current_output_dir = output_dir

def reset_pruned_data():
    global _pruned_data_cache
    _pruned_data_cache = {}

def save_pruned_tokens(attn_entropy: Optional[torch.Tensor], x_shape: Tuple[int, int, int], prune_scale_list: Dict[int, float], layer_idx: int = -1):
    """
    Captures which tokens would be pruned at this layer/scale.
    This function is called from FastVARCrossAttnBlock.
    """
    if attn_entropy is None or _current_output_dir is None:
        return

    _, original_h, original_w = x_shape
    B = attn_entropy.shape[0]
    if original_w not in prune_scale_list:
        return

    # Determine remaining tokens (matching logic in fastvar_utils.py)
    ratio = prune_scale_list[original_w]
    L = original_h * original_w
    num_remain = int(L * (1.0 - ratio))
    if num_remain <= 0: num_remain = 1
    if num_remain >= L: return # No pruning

    # Get entropy scores (averaged over heads)
    entropy_avg = torch.mean(attn_entropy, dim=1) # (B, L)
    
    # Identify pruned indices (those with SMALLEST entropy in entropy-based pruning? 
    # compute_merge_entropy uses HIGHEST entropy to keep. So we sort descending and take the tail.)
    indices = torch.argsort(entropy_avg, dim=1, descending=True)
    pruned_indices = indices[:, num_remain:] # (B, L - num_remain)

    scale_key = (original_h, original_w)
    if scale_key not in _pruned_data_cache:
        _pruned_data_cache[scale_key] = []
    
    _pruned_data_cache[scale_key].append({
        'layer_idx': layer_idx,
        'pruned_indices': pruned_indices.detach().cpu().numpy().tolist()
    })

def finalize_pruned_tokens():
    """Saves the cached pruned data to a JSON file in the output directory."""
    if _current_output_dir is None or not _pruned_data_cache:
        return

    json_path = os.path.join(_current_output_dir, "pruned_tokens.json")
    
    summary = {}
    for (h, w), layers in _pruned_data_cache.items():
        if not layers: continue
        summary[f"{h}x{w}"] = layers

    with open(json_path, 'w') as f:
        json.dump(summary, f)
    
    return json_path

def plot_pruned_tokens(img_path: Optional[str] = None, json_path: Optional[str] = None, output_path: Optional[str] = None, batch_idx: int = 0):
    """
    Reads the pruned_tokens.json and overlays a mask on the generated image.
    Generates one image per scale and per layer.
    """
    if img_path is None and _current_output_dir is not None:
        img_path = os.path.join(_current_output_dir, "1.jpg")
    if json_path is None and _current_output_dir is not None:
        json_path = os.path.join(_current_output_dir, "pruned_tokens.json")

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        if not os.path.exists(json_path) and img_path is not None:
            alt_json_path = os.path.join(os.path.dirname(img_path), "pruned_tokens.json")
            if os.path.exists(alt_json_path):
                json_path = alt_json_path
            else:
                return
        else:
            return

    img = cv2.imread(img_path)
    if img is None: return
    H, W, _ = img.shape

    with open(json_path, 'r') as f:
        pruned_data = json.load(f)

    for scale_str, layers in pruned_data.items():
        h_s, w_s = map(int, scale_str.split('x'))
        for layer_info in layers:
            layer_idx = layer_info['layer_idx']
            batch_pruned = layer_info['pruned_indices']
            if batch_idx >= len(batch_pruned): continue
            
            pruned_indices = batch_pruned[batch_idx]
            scale_mask = np.zeros((h_s, w_s), dtype=np.float32)
            for idx in pruned_indices:
                r, c = divmod(idx, w_s)
                if r < h_s and c < w_s: scale_mask[r, c] = 1.0
            
            resized_mask = cv2.resize(scale_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            vis_img = img.astype(np.float32)
            vis_img[resized_mask > 0.5] *= 0.3
            vis_img[resized_mask > 0.5, 2] += 100 
            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)

            layer_output_path = img_path.replace(".jpg", f"_pruned_{scale_str}_layer{layer_idx}.jpg")
            cv2.imwrite(layer_output_path, vis_img)

def save_heatmap(matrix, path, title="Similarity Matrix", x_label="Index", y_label="Index"):
    """
    Creates a heatmap visualization using matplotlib to match the style of plot_entropy.py.
    - Range: 0.3 (Red) to 1.0 (Blue)
    - Values < 0.3 are clipped to Red.
    """
    plt.figure(figsize=(10, 8))
    
    # Use 'RdBu' colormap: 0.3 is Red, 1.0 is Blue.
    # By setting vmin=0.3 and vmax=1.0, values are mapped accordingly.
    im = plt.imshow(matrix, aspect='auto', cmap='RdBu', vmin=-1.0, vmax=1.0, interpolation='nearest')
    
    # Add colorbar with a label
    cbar = plt.colorbar(im)
    cbar.set_label('Similarity Value', rotation=270, labelpad=15)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    if max(matrix.shape) <= 32:
        plt.xticks(np.arange(matrix.shape[1]))
        plt.yticks(np.arange(matrix.shape[0]))

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[evaluate] Saved heatmap with range [0.5, 1.0] to {path}")

def save_bar_chart(x_values, y_values, path, title, x_label, y_label):
    """Creates a bar chart to visualize average similarity across layers."""
    plt.figure(figsize=(12, 6))
    plt.bar(x_values, y_values, color='skyblue', edgecolor='navy')
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust Y limits based on data
    y_min = min(min(y_values), 0)
    y_max = max(max(y_values), 1.0)
    plt.ylim(y_min - 0.1, y_max + 0.1)
    
    if len(x_values) <= 32:
        plt.xticks(x_values)
        
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[evaluate] Saved bar chart to {path}")

def get_full_mask(pruned_indices, total_tokens):
    """Converts pruned indices to a vector of -1 (pruned) and 1 (unpruned)."""
    mask = np.ones(total_tokens, dtype=np.float32)
    mask[pruned_indices] = -1.0
    return mask

def evaluate_similarities(results_root: str):
    """
    Evaluates cross-prompt and cross-layer similarity across all generated samples.
    """
    json_paths = glob.glob(os.path.join(results_root, "**/pruned_tokens.json"), recursive=True)
    if not json_paths:
        print(f"[evaluate] No pruned_tokens.json found in {results_root}")
        return

    # Data structure: all_data[scale][layer][prompt_name] = mask_vector
    all_data = {}
    
    print(f"[evaluate] Loading {len(json_paths)} files...")
    for jp in json_paths:
        prompt_name = os.path.basename(os.path.dirname(jp))
        with open(jp, 'r') as f:
            data = json.load(f)
            
        for scale_str, layers in data.items():
            h_s, w_s = map(int, scale_str.split('x'))
            total_tokens = h_s * w_s
            
            if scale_str not in all_data: all_data[scale_str] = {}
            
            for layer_info in layers:
                l_idx = layer_info['layer_idx']
                if l_idx not in all_data[scale_str]: all_data[scale_str][l_idx] = {}
                
                # Assume batch size 1 for evaluation or take first
                pruned_indices = layer_info['pruned_indices'][0]
                mask_vec = get_full_mask(pruned_indices, total_tokens)
                all_data[scale_str][l_idx][prompt_name] = mask_vec

    # 1. Cross-Prompt Similarity (per scale)
    eval_dir = os.path.join(results_root, "evaluation_pruning")
    cp_dir = os.path.join(eval_dir, "cross_prompt")
    os.makedirs(cp_dir, exist_ok=True)
    
    for scale_str, layers_dict in all_data.items():
        scale_subdir = os.path.join(cp_dir, f"scale{scale_str}")
        os.makedirs(scale_subdir, exist_ok=True)
        
        layer_indices = sorted(layers_dict.keys())
        avg_similarities = []
        
        for l_idx in layer_indices:
            prompts_dict = layers_dict[l_idx]
            prompt_names = sorted(prompts_dict.keys())
            N = len(prompt_names)
            if N < 2: 
                avg_similarities.append(0.0)
                continue
            
            masks = [prompts_dict[name] for name in prompt_names]
            L = len(masks[0])
            
            # Vectorized computation of similarity matrix
            stacked_masks = np.stack(masks) # (N, L)
            sim_matrix = (stacked_masks @ stacked_masks.T) / L
            
            # Average of off-diagonal elements (i != j)
            # Total sum minus diagonal, divided by (N*N - N)
            if N > 1:
                off_diag_avg = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (N * N - N)
            else:
                off_diag_avg = 0.0
            avg_similarities.append(off_diag_avg)
            
        bar_path = os.path.join(scale_subdir, f"cross_prompt_{scale_str}_bar.jpg")
        save_bar_chart(layer_indices, avg_similarities, bar_path, 
                       title=f"Avg Cross-Prompt Similarity [{scale_str}]", 
                       x_label="Layer Index", y_label="Avg Similarity (i != j)")

    # 2. Cross-Layer Similarity (per prompt, per scale)
    cl_dir = os.path.join(eval_dir, "cross_layer")
    os.makedirs(cl_dir, exist_ok=True)

    for scale_str, layers_dict in all_data.items():
        # Reorganize to: prompts_layers[prompt_name][layer_idx] = mask
        prompts_layers = {}
        all_l_indices = sorted(layers_dict.keys())
        
        for l_idx in all_l_indices:
            for p_name, mask in layers_dict[l_idx].items():
                if p_name not in prompts_layers: prompts_layers[p_name] = {}
                prompts_layers[p_name][l_idx] = mask
        
        for p_name, l_dict in prompts_layers.items():
            l_indices = sorted(l_dict.keys())
            K = len(l_indices)
            if K < 2: continue
            
            masks = [l_dict[idx] for idx in l_indices]
            L = len(masks[0])
            stacked_masks = np.stack(masks) # (K, L)
            sim_matrix = (stacked_masks @ stacked_masks.T) / L
            
            out_subdir = os.path.join(cl_dir, p_name)
            os.makedirs(out_subdir, exist_ok=True)
            out_path = os.path.join(out_subdir, f"cross_layer_{scale_str}.jpg")
            save_heatmap(sim_matrix, out_path, title=f"Cross-Layer Similarity [{scale_str}] {p_name}", x_label="Layer Index", y_label="Layer Index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True, help="Path to the results directory containing gen_images_* folders")
    args = parser.parse_args()
    
    evaluate_similarities(args.results_root)

'''
python infinity/utils/plot_pruned_tokens.py --results_root results_fashion
'''