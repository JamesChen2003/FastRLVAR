import os
import requests

cache_dir = "/nfs/home/tensore/RL/FastRLVAR/Infinity_v3/evaluation/MJHQ30K/"
os.makedirs(cache_dir, exist_ok=True)

zip_url = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/mjhq30k_imgs.zip"
zip_path = os.path.join(cache_dir, "mjhq30k_imgs.zip")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

print(f"Starting download: {zip_url}")
with requests.get(zip_url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    if tqdm is None:
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    else:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
print(f"Download finished: {zip_path}")

import zipfile
from pathlib import Path

zip_path = Path(zip_path)
extract_dir = zip_path.parent / "mjhq30k_imgs"

with zipfile.ZipFile(zip_path, "r") as zf:
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    members = zf.infolist()
    if tqdm is None:
        zf.extractall(extract_dir)
    else:
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_dir)

print(f"Extracted to {extract_dir}")
