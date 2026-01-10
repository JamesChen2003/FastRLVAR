import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Numerical stability for log-reward
# We report multiple eps settings for comparison.
EPS_LIST = [1e-1, 1e-5]

# Linear similarity normalization window:
# Map SIM_MIN~SIM_MAX -> 0~1, clamp below/above.
SIM_MIN = 0.88
SIM_MAX = 0.98

# Use a smaller DINO model for faster demos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
# Load the processor and model
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval() # Set the model to evaluation mode

def extract_pooled_features(images):
    """Extracts global (CLS token) features for a batch of images."""
    inputs = processor(images=images, return_tensors="pt")
    # Move tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # DINO/ViT models may not expose pooler_output; use CLS token instead.
    cls = outputs.last_hidden_state[:, 0, :]  # [B, D]
    return cls.detach().cpu().numpy()

def extract_patch_features(image):
    """Extracts the dense patch features for a single image."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # The first token is the class token, so we skip it
    patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()
    return patch_features

def get_dinov3_similarity(image1, image2):
    image1_features = extract_pooled_features([image1])
    image2_features = extract_pooled_features([image2])
    similarity = float(cosine_similarity(image1_features, image2_features).item())
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return similarity

if __name__ == "__main__":
    # Resolve paths relative to this script so it works from any CWD
    here = os.path.dirname(os.path.abspath(__file__))
    image1_path = os.path.join(here, "..", "training_tmp_results", "golden_img_12.png")
    image2_path = os.path.join(here, "..", "training_tmp_results", "pruned_img_12.png")

    # Use the cat image as the query
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    image1_features = extract_pooled_features([image1])
    image2_features = extract_pooled_features([image2])

    # Cosine similarity in [-1, 1]
    similarity = float(cosine_similarity(image1_features, image2_features).item())
    similarity = float(np.clip(similarity, -1.0, 1.0))

    # Log-scale reward:
    #   Reward = -log(1 - cos(phi(G), phi(R)) + eps)
    # Also print a normalized version in [0, 1] by dividing over -log(eps).
    rewards_by_eps = []
    for eps in EPS_LIST:
        reward = float(-np.log(1.0 - similarity + float(eps)))
        max_reward = float(-np.log(float(eps)))
        reward_norm = float(np.clip(reward / max_reward, 0.0, 1.0))
        rewards_by_eps.append((float(eps), reward, reward_norm))

    # Alternative normalization: linearly map similarity in [SIM_MIN, SIM_MAX] -> [0, 1],
    # and clamp outside that window.
    sim_linear = float(np.clip((similarity - SIM_MIN) / (SIM_MAX - SIM_MIN), 0.0, 1.0))

    print(f"Similarity between image1 and image2: {similarity:.6f}")
    for eps, reward, reward_norm in rewards_by_eps:
        print(f"Log reward (-log(1 - sim + eps), eps={eps:g}): {reward:.6f}")
        print(f"Normalized log reward (divide by -log(eps)): {reward_norm:.6f}")
    print(f"Linear similarity norm (map {SIM_MIN:.2f}~{SIM_MAX:.2f} -> 0~1): {sim_linear:.6f}")