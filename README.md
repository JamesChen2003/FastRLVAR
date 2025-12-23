# FastRLVAR

This README focuses on installation and the three main workflows: training, inference, and evaluation.

## Installation

Choose one:

### Use requirements.txt

```bash
pip install -r requirements.txt
```

### Use environment.yml

```bash
conda env create -f environment.yml
conda activate eval_env
```

### FlashAttention (important)

Install `flash-attention` by downloading the wheel that matches your CUDA / PyTorch versions from:

https://github.com/Dao-AILab/flash-attention/releases

Example:

```bash
pip install /path/to/flash_attn-*.whl
```

## Checkpoints

### Download T5 (text encoder)

Run the helper script to fetch `google/flan-t5-xl` into your Hugging Face cache:

```bash
python Infinity/checkpoint/download_t5.py
```

### Download Infinity checkpoint

```bash
mkdir -p Infinity/checkpoint
curl -L https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_2b_reg.pth \
  -o Infinity/checkpoint/infinity_2b_reg.pth
```
### Download VAE checkpoint

```bash
mkdir -p Infinity/checkpoint
curl -L https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32.pth \
  -o Infinity/checkpoint/infinity_vae_d32.pth

### Dataset
only use meat_data.json don't need to download

### Training (Infinity_v2/train.py)

```bash
python Infinity_v2/train.py
```

This script has no CLI arguments. Configure these in `Infinity_v2/train.py`:

- `model_path`: path to the Infinity checkpoint
- `vae_path`: path to the VAE checkpoint
- `text_encoder_ckpt`: T5 checkpoint path (or HF cache dir)
- metadata load path: JSON file used to build the prompt pool
- `my_config["run_id"]`: W&B run name
- `my_config["save_path"]`: model output path
- `my_config["epoch_num"]`: number of epochs
- `my_config["prompt_pool_size"]`: prompt pool size

Note: this script initializes W&B by default (login required).

## Inference (Infinity_v2/inference_var_for_eval_format.py)

```bash
python Infinity_v2/inference_var_for_eval_format.py
```

This script has no CLI arguments. Configure these in `Infinity_v2/inference_var_for_eval_format.py`:

- `model_path`: path to the Infinity checkpoint
- `vae_path`: path to the VAE checkpoint
- `text_encoder_ckpt`: T5 checkpoint path (or HF cache dir)
- metadata load path: JSON file used to build prompts (default `Infinity_v2/meta_data.json` class people total 1000 testcase )
- `my_config["prompt_pool_size"]`: number of prompts to run
- `load_model`: whether to load a PPO agent (True/False)
- `trained_model_path`: PPO checkpoint path (when `load_model` is True)

Outputs are written under `results/` in an auto-incremented subfolder.

## Evaluation (Infinity_v2/eval_hpsv3.py)

```bash
python Infinity_v2/eval_hpsv3.py \
  --original /path/to/original \
  --fastvar /path/to/fastvar \
  --RL /path/to/RL
```
--f --o also accepted

Arguments:

- `--original` (required): folder with original images and `meta_data.json`
- `--fastvar` (optional): folder with FastVAR outputs
- `--RL` (optional): folder with RL outputs

Notes:

- CUDA GPU is required for HPSv3 inference
- Results are saved to `hpsv3_comparison_results.csv`
