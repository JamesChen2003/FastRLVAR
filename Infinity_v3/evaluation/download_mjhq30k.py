from huggingface_hub import hf_hub_download

cache_dir = "/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v2/evaluation/MJHQ30K/"

hf_hub_download(
  repo_id="playgroundai/MJHQ-30K", 
  filename="mjhq30k_imgs.zip", 
  local_dir=cache_dir,
  repo_type="dataset"
)

import zipfile
from pathlib import Path

zip_path = Path("/home/remote/LDAP/r14_jameschen-1000043/FastVAR/Infinity_v2/evaluation/MJHQ30K/mjhq30k_imgs.zip")
extract_dir = zip_path.parent / "mjhq30k_imgs"

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

print(f"Extracted to {extract_dir}")