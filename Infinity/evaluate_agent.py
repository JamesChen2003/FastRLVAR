import argparse
import os
import glob
import json
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# --- Metrics Libraries ---
# 為了避免未安裝所有庫導致報錯，使用 try-import
try:
    import lpips
except ImportError:
    lpips = None

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    psnr_metric = None

try:
    import clip
except ImportError:
    clip = None
try:
    import ImageReward as RM
except ImportError:
    RM = None
sys.path.append("/nfs/home/tensore/RL/FastRLVAR/VIEScore")

import viescore
# ==========================================
# CONFIGURATION: 設定要測的項目
# ==========================================
ENABLE_METRICS = {
    # Similarity (需要 Original 對照)
    "psnr": True,
    "ssim": True,
    "lpips": True,
    
    # Quality (無需 Original，需要 Prompt)
    "clip_score": False,
    "image_reward": False,  # 需安裝 ImageReward 環境有點問題
    "viescore": False,     # 需自定義模型
    "geneval": False       # 需自定義模型
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMAGE_REWARD_MODEL = "zai-org/ImageReward"  # HF repo id
DEFAULT_VIESCORE_MODEL = "TIGER-Lab/VIEScore"

# ==========================================
# Metric Helper Classes
# ==========================================

def load_viescore_model(model_name: str, device: str):
    """
    使用本地 VIEScore repo 的 VIEScore 類別，不走 HuggingFace。
    model_name 目前會被忽略，只是為了相容舊參數。
    """
    cls = getattr(viescore, "VIEScore", None)
    if cls is None:
        print("Warning: viescore.VIEScore not found, VIEScore metric will be 0.")
        return None

    try:
        print("Initializing local VIEScore(backbone='gemini', task='t2i') from viescore.VIEScore ...")
        model = cls(backbone="gemini", task="t2i")
        print("VIEScore model initialized.")
        return model
    except Exception as e:
        print(f"Warning: failed to initialize local VIEScore: {e}")
        return None


class MetricsEvaluator:
    def __init__(self, device, reward_model_path=None, viescore_model_path=None):
        self.device = device
        self.lpips_fn = None
        self.clip_model = None
        self.clip_preprocess = None
        self.reward_model = None
        self.viescore_model = None

        print(f"Initializing Metrics on {device}...")

        # 1. Initialize LPIPS
        if ENABLE_METRICS["lpips"]:
            if lpips is None:
                print("Warning: lpips not installed, LPIPS metric will be 0.")
            else:
                print("Loading LPIPS model...")
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # 2. Initialize CLIP
        if ENABLE_METRICS["clip_score"] or ENABLE_METRICS["viescore"]:
            if clip is None:
                print("Warning: openai-clip not installed, CLIP-based metrics will be 0.")
            else:
                print("Loading CLIP model...")
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 3. Initialize ImageReward
        if ENABLE_METRICS["image_reward"]:

            if RM is None:
                print("Warning: ImageReward/image-reward not importable, ImageReward metric will be 0.")
                print("  Try:  pip install ImageReward  或  pip install image-reward")
            else:
                model_to_load = reward_model_path or DEFAULT_IMAGE_REWARD_MODEL
                try:
                    print(f"Loading ImageReward model ({model_to_load})...")
                    # 盡量相容不同版本：優先用 .load，其次用 .ImageReward 類別
                    if hasattr(RM, "load"):
                        self.reward_model = RM.load(model_to_load)  # 自動下載或讀 cache / 本地
                    elif hasattr(RM, "ImageReward"):
                        cls = RM.ImageReward
                        try:
                            self.reward_model = cls(model_to_load, device=device)
                        except TypeError:
                            # 有些版本只接 device
                            self.reward_model = cls(device=device)
                    else:
                        raise RuntimeError("ImageReward module has no 'load' or 'ImageReward' attribute")
                    print("ImageReward model loaded.")
                except Exception as e:
                    print(f"Warning: failed to load ImageReward model '{model_to_load}': {e}")
                    self.reward_model = None

        # 4. Initialize VIEScore (no proxy/fallback)
        if ENABLE_METRICS["viescore"]:
            model_to_load = viescore_model_path or DEFAULT_VIESCORE_MODEL
            self.viescore_model = load_viescore_model(model_to_load, device=device)

    def calc_psnr(self, img_true, img_test):
        # Input: Numpy arrays [H, W, C], range [0, 255]
        return psnr_metric(img_true, img_test, data_range=255)

    def calc_ssim(self, img_true, img_test):
        # Input: Numpy arrays [H, W, C], range [0, 255]
        # win_size 設為 3 或 7，視圖片大小而定，multichannel=True
        return ssim_metric(img_true, img_test, data_range=255, channel_axis=2, win_size=3)

    def calc_lpips(self, img_true_pil, img_test_pil):
        # Input: PIL Images
        # LPIPS expects tensors in range [-1, 1]
        if self.lpips_fn is None: return 0.0
        
        t_true = lpips.im2tensor(np.array(img_true_pil)).to(self.device)
        t_test = lpips.im2tensor(np.array(img_test_pil)).to(self.device)
        
        with torch.no_grad():
            dist = self.lpips_fn(t_true, t_test)
        return dist.item()

    def calc_clip_score(self, img_pil, prompt_text):
        if self.clip_model is None: return 0.0
        if prompt_text is None: return 0.0
        
        image = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
        text = clip.tokenize([prompt_text], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # 這裡回傳 cosine similarity 本身比較直觀
            score = torch.cosine_similarity(image_features, text_features).item()
        return score

    def calc_image_reward(self, img_pil, prompt_text):
        if self.reward_model is None: return 0.0
        if prompt_text is None: return 0.0
        with torch.no_grad():
            score = self.reward_model.score(prompt_text, img_pil)
        return score

    def calc_viescore(self, img_pil, prompt_text):
        if self.viescore_model is None: 
            return 0.0
        if prompt_text is None:
            return 0.0
        # 使用官方 VIEScore 介面：evaluate(image, text, ...)
        try:
            with torch.no_grad():
                score = self.viescore_model.evaluate(
                    img_pil,
                    prompt_text,
                    extract_overall_score_only=True,
                    extract_all_score=False,
                    echo_output=False,
                )
        except Exception as e:
            print(f"Warning: VIEScore.evaluate failed: {e}; returning 0.")
            return 0.0
        # evaluate 可能回傳單一分數或 list
        if isinstance(score, (list, tuple)) and len(score) > 0:
            return float(score[-1])
        try:
            return float(score)
        except Exception:
            return 0.0

    def calc_geneval(self, img_pil, prompt_text):
        # TODO: GenEval usually requires running object detection 
        # against parsing of the prompt.
        return 0.0


# ==========================================
# Utils
# ==========================================

def get_image_paths(folder):
    """Get all image paths sorted by filename."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files

def load_metadata(folder):
    """
    嘗試從資料夾讀取 meta_data/metadata 檔案，用來取得 prompt。
    """
    for name in ("meta_data.json", "metadata.json"):
        meta_path = os.path.join(folder, name)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: failed to load metadata from {meta_path}: {e}")
    return {}


def get_prompt_from_filename(filepath, metadata):
    """
    先從 metadata 找 prompt，找不到再退回用檔名推測。
    metadata 允許兩種格式：
      1) {<id>: {"prompt": "...", ...}}
      2) {<id>: "<prompt>"}
    """
    basename = os.path.basename(filepath)
    file_id = os.path.splitext(basename)[0]

    entry = metadata.get(file_id, None) if isinstance(metadata, dict) else None
    if isinstance(entry, dict):
        prompt = entry.get("prompt")
        if prompt:
            return prompt
    elif isinstance(entry, str):
        return entry

    # fallback: derive from filename
    return file_id.replace('_', ' ')

# ==========================================
# Main Evaluation Loop
# ==========================================

def evaluate(args):
    evaluator = MetricsEvaluator(
        DEVICE,
        reward_model_path=args.reward_model,
        viescore_model_path=args.viescore_model,
    )

    # 依據 ENABLE_METRICS 建立 CSV 欄位
    metric_flags = [
        ("PSNR", "psnr"),
        ("SSIM", "ssim"),
        ("LPIPS", "lpips"),
        ("CLIP", "clip_score"),
        ("ImageReward", "image_reward"),
        ("VIEScore", "viescore"),
    ]
    metric_columns = [name for name, flag in metric_flags if ENABLE_METRICS.get(flag, False)]
    csv_columns = ["id"] + metric_columns

    fast_metadata = load_metadata(args.fast)
    rl_metadata = load_metadata(args.rl)
    orig_metadata = load_metadata(args.orig)
    
    # 1. 讀取檔案列表
    orig_files = get_image_paths(args.orig)
    fast_files = get_image_paths(args.fast)
    rl_files = get_image_paths(args.rl)

    # 確保數量一致 (以 FAST 為基準，因為 Original 可能是參考用)
    print(f"Found images -> Orig: {len(orig_files)}, FAST: {len(fast_files)}, RL: {len(rl_files)}")
    
    # 用於儲存每張圖的結果
    per_image_rows = {
        "ORIGINAL": [],
        "FAST": [],
        "RL": []
    }

    def init_row(file_id):
        row = {"id": file_id}
        for col in metric_columns:
            row[col] = None
        return row

    # 以 FAST 資料夾的檔案名稱為主 key
    for idx, fast_path in enumerate(tqdm(fast_files, desc="Evaluating Images")):
        filename = os.path.basename(fast_path)
        file_id = os.path.splitext(filename)[0]
        
        # 尋找對應的 Orig 和 RL 檔案
        orig_path = os.path.join(args.orig, filename)
        rl_path = os.path.join(args.rl, filename)

        if not os.path.exists(orig_path):
            # print(f"Warning: Original missing for {filename}, skipping similarity metrics.")
            orig_img = None
        else:
            orig_img = Image.open(orig_path).convert("RGB")
            orig_np = np.array(orig_img)

        if not os.path.exists(rl_path):
            rl_img = None
        else:
            rl_img = Image.open(rl_path).convert("RGB")

        fast_img = Image.open(fast_path).convert("RGB")
        fast_prompt = get_prompt_from_filename(fast_path, fast_metadata)
        rl_prompt = get_prompt_from_filename(rl_path, rl_metadata) if rl_img is not None else None
        orig_prompt = get_prompt_from_filename(orig_path, orig_metadata) if orig_img is not None else None

        # Original 圖片的品質指標
        if orig_img is not None:
            orig_row = init_row(file_id)
            if ENABLE_METRICS["clip_score"]:
                orig_row["CLIP"] = evaluator.calc_clip_score(orig_img, orig_prompt)
            if ENABLE_METRICS["image_reward"]:
                orig_row["ImageReward"] = evaluator.calc_image_reward(orig_img, orig_prompt)
            if ENABLE_METRICS["viescore"]:
                orig_row["VIEScore"] = evaluator.calc_viescore(orig_img, orig_prompt)
            per_image_rows["ORIGINAL"].append(orig_row)
        
        # 定義要測試的目標 (Target) 與 對照組 (Reference)
        # 我們要測試 FAST 和 RL
        targets = [
            ("FAST", fast_img, fast_prompt),
            ("RL", rl_img, rl_prompt)
        ]

        for name, img, prompt in targets:
            if img is None: continue
            
            img_np = np.array(img)
            row = init_row(file_id)

            # --- Similarity Metrics (Requires Original) ---
            if orig_img is not None:
                if ENABLE_METRICS["psnr"] and psnr_metric:
                    row["PSNR"] = evaluator.calc_psnr(orig_np, img_np)
                
                if ENABLE_METRICS["ssim"] and ssim_metric:
                    row["SSIM"] = evaluator.calc_ssim(orig_np, img_np)
                
                if ENABLE_METRICS["lpips"]:
                    row["LPIPS"] = evaluator.calc_lpips(orig_img, img)

            # --- Quality Metrics (No Original Needed) ---
            if ENABLE_METRICS["clip_score"]:
                row["CLIP"] = evaluator.calc_clip_score(img, prompt)
            
            if ENABLE_METRICS["image_reward"]:
                row["ImageReward"] = evaluator.calc_image_reward(img, prompt)
            
            if ENABLE_METRICS["viescore"]:
                row["VIEScore"] = evaluator.calc_viescore(img, prompt)

            per_image_rows[name].append(row)

    # ==========================================
    # Summary & Output
    # ==========================================
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)

    final_data = {}
    for model_name in ["FAST", "RL", "ORIGINAL"]:
        rows = per_image_rows[model_name]
        if not rows:
            continue
        df_model = pd.DataFrame(rows)
        averages = {}
        for col in metric_columns:
            if col not in df_model.columns:
                continue
            series = pd.to_numeric(df_model[col], errors="coerce")
            series = series[series.notna()]
            averages[col] = series.mean() if not series.empty else 0.0
        final_data[model_name] = averages

    # Convert to DataFrame for pretty printing
    summary_df = pd.DataFrame(final_data).T
    cols = [c for c in metric_columns if c in summary_df.columns]
    summary_df = summary_df[cols]
    
    if not summary_df.empty:
        print(summary_df.to_string(float_format="{:.4f}".format))
    else:
        print("No metrics computed.")
    print("="*50)
    
    # Save summary to CSV
    summary_df.to_csv("evaluation_results.csv")
    print("Results saved to evaluation_results.csv")

    # 存成三個 CSV 檔 (Original / FAST / RL)
    def save_csv(rows, filename):
        if not rows:
            print(f"No data for {filename}, skip saving.")
            return
        df = pd.DataFrame(rows)
        for col in csv_columns:
            if col not in df.columns:
                df[col] = None
        df = df[csv_columns]
        df.to_csv(filename, index=False)
        print(f"Saved per-image metrics to {filename}")

    save_csv(per_image_rows["ORIGINAL"], "orignal.csv")
    save_csv(per_image_rows["FAST"], "FastVar.csv")
    save_csv(per_image_rows["RL"], "RL.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAR Model Results (Original vs FAST vs RL)")
    
    parser.add_argument("--orig", type=str, required=True, help="Folder path for Original (Unpruned) images")
    parser.add_argument("--fast", type=str, required=True, help="Folder path for FAST (Pruned) images")
    parser.add_argument("--rl", type=str, required=True, help="Folder path for RL images")
    parser.add_argument("--reward-model", type=str, default=None, help="Path or HF repo for ImageReward model (default: zai-org/ImageReward)")
    parser.add_argument("--viescore-model", type=str, default=None, help="Path or HF repo for VIEScore model (default: TIGER-Lab/VIEScore)")
    
    args = parser.parse_args()
    
    # Check folders
    if not os.path.exists(args.orig) or not os.path.exists(args.fast) or not os.path.exists(args.rl):
        print("Error: One or more input directories do not exist.")
    else:
        evaluate(args)
