#!/usr/bin/env python
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_fourier_scales(
    npy_paths,
    labels=None,
    save_path=None,
    title="Fourier Analysis of Large-Scale Steps",
    skip: int = 1,
    last_n: int = None,
):
    """
    Plot stacked Δ log-amplitude spectra.

    Args:
        skip: keep every <skip> curves.
              Example:
                skip=1 → keep all
                skip=2 → keep every 2 curves
                skip=4 → keep every 4 curves
        last_n: keep only the last <last_n> curves
    """
    if len(npy_paths) == 0:
        print("[plot_fourier_scales] No npy_paths provided, skip.")
        return

    # Load all curves
    curves = [np.load(p) for p in npy_paths]

    # Keep only last N curves if requested
    if last_n is not None:
        curves = curves[-last_n:]
        npy_paths = npy_paths[-last_n:]

    # Apply skip
    if skip <= 0:
        skip = 1

    keep_indices = list(range(0, len(curves), skip))
    curves = [curves[i] for i in keep_indices]
    npy_paths = [npy_paths[i] for i in keep_indices]

    if len(curves) == 0:
        print("[plot_fourier_scales] After applying skip, no curves remain. Skip.")
        return

    # Make all curves same length
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    x = np.linspace(0.0, 1.0, min_len)

    # ----- Label logic fix -----
    # If labels are not provided, derive step index from filename AFTER last_n & skip
    if labels is None:
        labels = []
        for path in npy_paths:
            fname = os.path.basename(path)
            m = re.search(r"dft_scale_(\d+)_", fname)
            if m is not None:
                step = int(m.group(1))
                labels.append(f"Step {step}")
            else:
                labels.append("Step ?")
    else:
        # If labels were provided for all original paths, subselect them
        labels = [labels[i] for i in keep_indices]
    # ---------------------------

    # Plot
    cmap = plt.cm.Blues
    num = len(curves)

    plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.gca()

    for i, (curve, label) in enumerate(zip(curves, labels)):
        color = cmap(0.3 + 0.6 * i / max(num - 1, 1))
        lw = 1.5 + 0.5 * i / max(num - 1, 1)
        ax.plot(x * np.pi, curve, color=color, linewidth=lw, label=label)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Δ Log Amplitude")
    ax.set_title(title)

    xticks_pi = np.array([0.00, 0.17, 0.33, 0.50, 0.67, 0.83, 1.00])
    xticks = xticks_pi * np.pi
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.2f}π" for v in xticks_pi])

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[FOURIER PLOT] Saved figure to: {save_path}")
        plt.close()
    else:
        plt.show()


def collect_and_plot_for_dir(dir_path: str, tag: str, skip: int = 1, last_n: int = None):
    """
    Collect dft_scale_XX_*.npy in a directory and plot.
    The output filename and plot title include BOTH:
        (a) the tag   → e.g. "latent_summed"
        (b) the root folder name → e.g. "city"
    """
    if not os.path.isdir(dir_path):
        print(f"[WARN] Directory not found, skip: {dir_path}")
        return

    # Extract parent directory name (e.g. gen_images_city → city)
    parent = os.path.basename(os.path.dirname(dir_path))
    # Often user dirs start with gen_images_xxx → strip prefix:
    if parent.startswith("gen_images_"):
        name = parent[len("gen_images_"):]   # "city"
    else:
        name = parent                       # fallback

    npy_with_scale = []
    for fname in os.listdir(dir_path):
        if not fname.endswith(".npy"):
            continue
        m = re.search(r"dft_scale_(\d+)_", fname)
        if m is None:
            continue
        scale_idx = int(m.group(1))
        full_path = os.path.join(dir_path, fname)
        npy_with_scale.append((scale_idx, full_path))

    if len(npy_with_scale) == 0:
        print(f"[INFO] No matching .npy files in: {dir_path}")
        return

    npy_with_scale.sort(key=lambda t: t[0])
    npy_paths = [p for (_, p) in npy_with_scale]

    # New customized filename
    save_fig_path = os.path.join(
        dir_path,
        f"fourier_plot_{tag}_{name}.png"   # <---- changed here
    )

    # New customized title
    pretty_tag = tag.replace("_", " ")
    plot_title = f"Fourier Spectrum for {pretty_tag} ({name})"

    plot_fourier_scales(
        npy_paths=npy_paths,
        save_path=save_fig_path,
        title=plot_title,
        skip=skip,
        last_n=last_n,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Analyze Fourier Δ log-amplitude spectra saved as .npy"
    )
    # allow multiple --root
    parser.add_argument(
        "--root",
        type=str,
        action="append",
        required=True,
        help=(
            "Root directory containing fourier_* folders. "
            "Can be specified multiple times, e.g. "
            "--root results/gen_images_city --root results/gen_images_man ..."
        ),
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Keep every N curves. skip=1 keeps all, skip=4 keeps every 4 curves.",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=None,
        help="Keep only the last N curves.",
    )
    args = parser.parse_args()

    root_dirs = args.root
    skip = args.skip
    last_n = args.last_n

    fourier_dirs = {
        "fourier_latent_unscaled": "fourier_latent_unscaled",
        "fourier_pixel_summed": "fourier_pixel_summed",
        "fourier_pixel_residual": "fourier_pixel_residual",
        "fourier_latent_summed": "fourier_latent_summed",
        "fourier_latent_residual": "fourier_latent_residual",
    }

    print(f"[INFO] Roots: {root_dirs}")
    print(f"[INFO] skip = {skip} (keep every {skip} curves)")
    print(f"[INFO] last_n = {last_n}")

    # process each root independently
    for root_dir in root_dirs:
        print(f"\n================ Root: {root_dir} ================")
        for tag, rel in fourier_dirs.items():
            dir_path = os.path.join(root_dir, rel)
            print(f"\n[PROCESS] {tag} @ {dir_path}")
            collect_and_plot_for_dir(dir_path, tag, skip=skip, last_n=last_n)


if __name__ == "__main__":
    main()


'''
python plot_fourier.py \
  --root results/gen_images_city \
  --root results/gen_images_man \
  --root results/gen_images_woman \
  --skip 4 --last_n 20
'''