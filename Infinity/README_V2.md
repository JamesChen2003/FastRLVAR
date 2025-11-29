## Updated Inference Guide (V2)

This document explains how to run the updated `inference.py` with dynamic pruning and per-scale inference.

All paths below are relative to the `FastVAR/Infinity` root.

---

## 1. Control the prompts (`inference.py` line 17)

Prompts are defined near the top of `inference.py`:

```17:23:FastVAR/Infinity/inference.py
prompts = {
    # "cat":       "A cute cat on the grass.",
    "city":      "A futuristic city skyline at night.",
    # "astronaut": "An astronaut painting on the moon.",
    # "woman":     "An anime-style portrait of a woman.",
    # "man":       "A detailed photo-realistic image of a man."
}
```

- **To change or add prompts**, edit this `prompts` dict:
  - Keys are prompt names (used for folder names).
  - Values are the text prompts actually passed to the model.

Example:

```python
prompts = {
    "cat":  "A cute cat on the grass.",
    "city": "A futuristic city skyline at night.",
}
```

When you run `inference.py`, each entry in `prompts` will generate one image under:

- `results/gen_images_<name>/1.jpg`

---

## 2. Control `pruning_scales` (line 34) for `per_scale_infer = False`

Near the top of `inference.py`:

```30:35:FastVAR/Infinity/inference.py
# si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
# pruning_scales = "2:1.0,4:1.0,6:1.0,8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
# pruning_scales = "8:1.0,12:1.0,16:1.0,20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
# pruning_scales = "20:1.0,24:1.0,32:1.0,40:1.0,48:1.0,64:1.0"
# pruning_scales = "48:1.0,64:1.0"
pruning_scales = "64:1.0"
```

This string is parsed into a dict that is passed to the model as `prune_scale_list`.  
It is used in the **original one-shot inference path** (`per_scale_infer=False`, i.e. using `autoregressive_infer_cfg`) to:

- Apply **token-level pruning** at specified spatial scales (widths).

Format:

- `"scale:ratio"` pairs, comma-separated, e.g.:
  - `"32:0.4,40:0.5"` → prune 40% at 32×32, 50% at 40×40.
  - `"64:1.0"` → fully prune (effectively skip) the 64×64 scale in the original path.

To use this configuration you should:

1. Set `pruning_scales` as desired.
2. In your calling code, make sure `per_scale_infer=False` when you call `gen_one_img` so that `autoregressive_infer_cfg` is used.

> Note: In the distributed `inference.py` we currently set `per_scale_infer=True` to use the new step-wise API. The `pruning_scales` string is still logged and stored in metadata for reproducibility.

---

## 3. Per-scale inference and `get_pruning_ratio` (`run_infinity.py`)

When `per_scale_infer=True` (as in the updated `inference.py`), generation uses the **step-wise** API:

- `infinity.infer_pruned_per_scale(...)` is called once per scale.
- The pruning ratio for each scale is **not taken from `pruning_scales`**, but instead from `get_pruning_ratio` in `tools/run_infinity.py`:

```71:84:FastVAR/Infinity/tools/run_infinity.py
def get_pruning_ratio(scale: int, num_scales: int) -> float:
    """
    Example pruning schedule:
    - No pruning on earlier scales
    - Stronger pruning on the last few scales
    This function is just a placeholder; in your project you can replace it
    with an RL agent or any other controller.
    """
    prune_scale_list = [0.0] * num_scales
    # Apply pruning only to the last few scales.
    N = min(6, num_scales)
    tail_pattern = [0.3, 0.4, 1.0, 1.0, 0.8, 1.0][:N]
    prune_scale_list[-N:] = tail_pattern
    return prune_scale_list[scale]
```

- `scale` is the stage index `si` (`0..num_scales-1`).
- `num_scales` is `len(scale_schedule)`.
- The returned value is interpreted as:
  - `0.0 <= ratio < 1.0` → **token-level pruning** at that scale.
  - `ratio >= 1.0` → **stage skip** at that scale in the step-wise API.

To define your own schedule:

- Edit `tail_pattern` (or the whole function) to match your experiment.
- For RL-based control, this is the single point where you would query your agent and return the chosen ratio.

---

## 4. Saving intermediate images (`save_intermediate_results=True`)

The `gen_one_img` call in `inference.py` passes `save_intermediate_results` and `save_dir` into the model:

```128:146:FastVAR/Infinity/inference.py
generated_image = gen_one_img(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    prompt_text,
    g_seed=seed + idx,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=[cfg_value] * len(scale_schedule),
    tau_list=[tau_value] * len(scale_schedule),
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=enable_positive_prompt,
    save_intermediate_results=True,
    save_dir=root_output_dir,
    per_scale_infer=True,
)
```

When `save_intermediate_results=True`, `infer_pruned_per_scale` (and the original path) call:

- `save_intermediate_results_func(summed_codes, vae, si, f"{save_dir}/summed_codes")`
- `save_intermediate_results_func(rescale_codes, vae, si, f"{save_dir}/rescale_codes")`

Result:

- For each non-skipped scale `si`, PNGs are written to:
  - `results/gen_images_<name>/summed_codes/intermediate_scale_img_<si>.png`
  - `results/gen_images_<name>/rescale_codes/intermediate_scale_img_<si>.png`

If a scale is fully skipped (pruning ratio ≥ 1.0), there are no new `codes` for that scale, so no new PNG is written for that `si`.

---

## 5. DFT analysis with `analyze_fourier_npy.py`

Inside `infer_pruned_per_scale` (and the original `autoregressive_infer_cfg`) there are calls to `dft_results` that are **currently commented out** to save compute:

```968:1007:FastVAR/Infinity/infinity/models/infinity.py
#             # pixel-space spectra
#             dft_results(
#                 summed_codes,
#                 vae,
#                 vae_scale_schedule,
#                 scale_index=scale_ind,
#                 save_dir=f"{save_dir}/fourier_pixel_summed",
#                 mode="pixel",
#             )
#             dft_results(
#                 rescale_codes,
#                 vae,
#                 vae_scale_schedule,
#                 scale_index=scale_ind,
#                 save_dir=f"{save_dir}/fourier_pixel_residual",
#                 mode="pixel",
#             )
#
#             # latent-space spectra
#             dft_results(
#                 summed_codes,
#                 vae,
#                 vae_scale_schedule,
#                 scale_index=scale_ind,
#                 save_dir=f"{save_dir}/fourier_latent_summed",
#                 mode="latent",
#             )
#             dft_results(
#                 rescale_codes,
#                 vae,
#                 vae_scale_schedule,
#                 scale_index=scale_ind,
#                 save_dir=f"{save_dir}/fourier_latent_residual",
#                 mode="latent",
#             )
#             dft_results(
#                 codes,
#                 vae,
#                 vae_scale_schedule,
#                 scale_index=scale_ind,
#                 save_dir=f"{save_dir}/fourier_latent_unscaled",
#                 mode="latent",
#             )
```

To enable DFT analysis:

1. **Uncomment** the relevant `dft_results(...)` blocks in `infinity/models/infinity.py`.
2. Run `inference.py` again.  
   - DFT outputs will be written under:
     - `results/gen_images_<name>/fourier_pixel_*`
     - `results/gen_images_<name>/fourier_latent_*`
3. Use `scripts/analyze_fourier_npy.py` to visualize or further analyze these spectra.

This lets you compare frequency content **with and without pruning**, and in particular:

- Behavior when certain scales are skipped entirely (pruning ratio ≥ 1.0).
- Behavior at the last non-skipped scale.

---

## 6. Running the pipeline

From the `FastVAR/Infinity` directory:

```bash
conda activate fastvar
python inference.py
```

Outputs:

- Per-prompt images: `results/gen_images_<name>/1.jpg`
- Per-prompt metadata: `results/gen_images_<name>/config.json`
- Global metadata: `results/config_all.json`
- Optional intermediates/DFT results as described above.


---

## 6. 評估指標 (`evaluate_agent.py`)

`evaluate_agent.py` 用來比較三組影像結果：Original / FAST / RL，並計算多種影像與文字相關指標。

- 支援指標：
  - Similarity：PSNR、SSIM、LPIPS（需要 `--orig` 對照圖）。
  - Text-Image Quality：CLIP Score（必須有 prompt）。
  - 可選：ImageReward、VIEScore（需額外安裝與 API key，預設關閉或實驗性使用）。
- Prompt 來源：
  - 優先從輸入資料夾中的 `meta_data.json` 或 `metadata.json` 取得（格式為 `{image_id: {"prompt": "...", ...}}`）。
  - 若找不到，才退回用檔名推測 prompt（`xxx_yyy.jpg -> "xxx yyy"`）。

基本使用方式（從 `Infinity/` 底下執行）：

```bash
python evaluate_agent.py \
  --orig path/to/orig_images \
  --fast path/to/fast_images \
  --rl   path/to/rl_images