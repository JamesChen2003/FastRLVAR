import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image  # Changed to standard PIL for robust local loading
import numpy as np
import os

# Numerical stability for log-reward
EPS_LIST = [1e-1, 1e-5]

# Linear similarity normalization window
SIM_MIN = 0.88
SIM_MAX = 0.98

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"

print(f"Loading model: {model_name} on {device}...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def extract_features(image):
    """
    Extracts both CLS (pooled) and Patch features for an image.
    """
    # Force convert to RGB to handle PNG alpha channels (RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    
    # CLS token is at index 0
    cls_token = last_hidden_state[:, 0, :]
    
    # Patch tokens are from index 1 onwards
    patch_tokens = last_hidden_state[:, 1:, :]
    
    # L2 Normalize features
    cls_token = F.normalize(cls_token, p=2, dim=-1)
    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
    
    return cls_token, patch_tokens

def quality_func(x, a=0.88, b=0.98):
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

def get_dinov3_similarity(image1, image2, mode="cls"):
    # Extract features
    cls1, patches1 = extract_features(image1)
    cls2, patches2 = extract_features(image2)
    
    if mode == "cls":
        sim = torch.mm(cls1, cls2.T).item()
        return float(np.clip(sim, -1.0, 1.0))
    
    elif mode == "patch":
        # Handle shape mismatch if needed
        if patches1.shape[1] != patches2.shape[1]:
            min_patches = min(patches1.shape[1], patches2.shape[1])
            patches1 = patches1[:, :min_patches, :]
            patches2 = patches2[:, :min_patches, :]
            
        patch_sims = F.cosine_similarity(patches1, patches2, dim=-1)
        return float(patch_sims.mean().item())

    elif mode == "gram":
        n_patches = patches1.shape[1]
        gram1 = torch.bmm(patches1, patches1.transpose(1, 2)) / (n_patches ** 0.5)
        gram2 = torch.bmm(patches2, patches2.transpose(1, 2)) / (n_patches ** 0.5)
        
        diff = gram1 - gram2
        dist = torch.norm(diff, p="fro").item()
        return dist
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    # === FIXED PATHS BELOW ===
    # Using absolute path structure based on your provided path
    base_dir = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v3/training_tmp_results"
    
    # Updated extension to .png
    image1_path = os.path.join(base_dir, "golden_img_12.png")
    image2_path = os.path.join(base_dir, "pruned_img_12.png")
    
    print(f"Reference: {image1_path}")
    print(f"Candidate: {image2_path}")

    # Use PIL.Image.open directly for local files (safer than transformers load_image)
    try:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find image file. Check path:\n{e}")
        exit(1)

    # --- 1. CLS Score ---
    cls_sim = get_dinov3_similarity(image1, image2, mode="cls")
    print("\n--- [1] CLS Similarity (Global Semantic) ---")
    print(f"Score: {cls_sim:.6f}")
    
    for eps in EPS_LIST:
        reward = float(-np.log(1.0 - cls_sim + float(eps)))
        print(f" -> Log-Reward (eps={eps}): {reward:.6f}")

    # --- 2. Patch Score ---
    patch_sim = get_dinov3_similarity(image1, image2, mode="patch")
    print("\n--- [2] Patch Similarity (Local Structural) ---")
    print(f"Score: {patch_sim:.6f}")

    # --- 3. Gram Distance ---
    gram_dist = get_dinov3_similarity(image1, image2, mode="gram")
    print("\n--- [3] Gram Matrix Distance (Structural Integrity) ---")
    print(f"Distance: {gram_dist:.6f}")

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} | {'Value':<10} | {'Meaning for Pruning'}")
    print("-" * 60)
    print(f"{'CLS Similarity':<20} | {quality_func(cls_sim, 0.9, 0.98):.4f}     | Did the model lose the 'concept'?")
    print(f"{'Patch Similarity':<20} | {quality_func(patch_sim, 0.92, 0.98):.4f}     | Did the faces/details melt?")
    print(f"{'1-Gram Distance':<20} | {quality_func(1-gram_dist, 0.3, 1.0):.4f}     | Is the spatial structure broken?")
    print("-" * 60)