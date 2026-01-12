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
from infinity.utils.plot_pruned_tokens import set_pruned_output_dir, reset_pruned_data, finalize_pruned_tokens, plot_pruned_tokens

model_path = '/nfs/home/tensore/pretrained/Infinity/infinity_2b_reg.pth'
vae_path   = '/nfs/home/tensore/pretrained/Infinity/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/nfs/home/tensore/pretrained/Infinity/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001'

# Benchmark settings
batch_size = 1
skip_last_cfg = True
debug_bs = False
per_scale_infer = False
base_output_dir = "results_speed_test"
os.makedirs(base_output_dir, exist_ok=True)

# Generate pruning configurations
# Last 4 scales: 32, 40, 48, 64
target_scales = [32, 40, 48, 64]
steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

configs = []

# 1. Baseline (all zeros)
configs.append({32: 0.0, 40: 0.0, 48: 0.0, 64: 0.0})

# 3. Variations
for s in target_scales:
    for v in steps:
        conf = {32: 0.0, 40: 0.0, 48: 0.0, 64: 0.0}
        conf[s] = v
        configs.append(conf)

def dict_to_str(d):
    return ",".join([f"{k}:{v:.2f}" for k, v in sorted(d.items())])

args = argparse.Namespace(
    pn='1M',
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
    start_entropy_scale=24,
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

# Settings for generation
cfg_value = 4
tau_value = 0.5
h_div_w = 1 / 1
seed = 42
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

prompt_text = "A beautiful sunset over a tranquil lake, highly detailed."
print(f"Benchmarking with prompt: {prompt_text}")

benchmark_results = {}

with torch.inference_mode():
    for conf_dict in configs:
        conf_str = dict_to_str(conf_dict)
        print(f"\n>>> Benchmarking Config: {conf_str}")
        
        latencies = []
        # Update pruning ratios on the model and its blocks directly
        infinity.prune_scale_list = conf_dict
        for blk in infinity.unregistered_blocks:
            if hasattr(blk, "prune_scale_list"):
                blk.prune_scale_list = conf_dict

        for i in range(4):  # 4 images per config
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt_text,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=[cfg_value] * len(scale_schedule),
                tau_list=[tau_value] * len(scale_schedule),
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                skip_last_cfg=skip_last_cfg,
                debug_bs=False,
                enable_positive_prompt=0,
                save_intermediate_results=False,
                per_scale_infer=per_scale_infer
            )
            end_event.record()
            torch.cuda.synchronize()
            
            latency = start_event.elapsed_time(end_event)
            latencies.append(latency)
            print(f"  Run {i+1}: {latency:.2f} ms")
            
        # Average later 3
        avg_latency = sum(latencies[1:]) / 3
        benchmark_results[conf_str] = avg_latency
        print(f"  Average (last 3): {avg_latency:.2f} ms")

# Save to JSON
output_json = os.path.join(base_output_dir, "latency_benchmark.json")
with open(output_json, "w") as f:
    json.dump(benchmark_results, f, indent=4)

# Generate relative speedup spreadsheet (CSV)
import csv
baseline_key = dict_to_str({32: 0.0, 40: 0.0, 48: 0.0, 64: 0.0})
if baseline_key in benchmark_results:
    baseline_latency = benchmark_results[baseline_key]
    output_csv = os.path.join(base_output_dir, "speedup_spreadsheet.csv")
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: Ratio \ Scale, 32, 40, 48, 64
        writer.writerow(["Ratio \\ Scale"] + [str(s) for s in target_scales])
        
        for r in steps:
            row_data = [f"{r:.2f}"]
            for s in target_scales:
                conf = {32: 0.0, 40: 0.0, 48: 0.0, 64: 0.0}
                conf[s] = r
                k = dict_to_str(conf)
                if k in benchmark_results:
                    speedup = baseline_latency / benchmark_results[k]
                    row_data.append(f"{speedup:.4f}")
                else:
                    row_data.append("N/A")
            writer.writerow(row_data)
    print(f"Speedup spreadsheet saved to {output_csv}")

print(f"\nDone! Results saved to {output_json}")
