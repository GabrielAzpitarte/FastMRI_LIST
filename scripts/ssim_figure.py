"""Figure: ground truth, slightly blurred ground truth, U-Net and EDM, with fastMRI SSIM maps.

.venv/bin/python scripts/ssim_figure.py --pd runs/diag_ssim/edm_s8/file1000107 --pdfs runs/diag_ssim/edm_s8/file1000114
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def panels(stem: str) -> tuple[list[tuple[str, np.ndarray]], np.ndarray, float]:
    """Central-slice images to compare against the ground truth."""
    z4, z2 = np.load(f"{stem}_4x.npz"), np.load(f"{stem}_2x.npz")
    mid = len(z4["target"]) // 2
    gt = z4["target"][mid]
    return [("ground truth", gt), ("GT blurred 0.7 px", ndimage.gaussian_filter(gt, 0.7)),
            ("U-Net 4×", z4["unet"][mid]), ("EDM 4×", z4["edm"][mid]), ("EDM 2×", z2["edm"][mid])], gt, float(z4["vmax"])


def draw(stems: dict, out: str) -> None:
    """Two rows per contrast: images and SSIM maps."""
    fig, axes = plt.subplots(2 * len(stems), 5, figsize=(15, 6.4 * len(stems)))
    for row, (label, stem) in enumerate(stems.items()):
        items, gt, vmax = panels(stem)
        hi = np.percentile(gt, 99.5)
        for col, (name, img) in enumerate(items):
            score, smap = structural_similarity(gt, img, data_range=vmax, full=True)
            axes[2 * row, col].imshow(img, cmap="gray", vmin=0, vmax=hi)
            axes[2 * row, col].set_title(f"{label}: {name}\nSSIM {score:.3f}", fontsize=10)
            axes[2 * row + 1, col].imshow(smap, cmap="viridis", vmin=0, vmax=1)
            axes[2 * row + 1, col].set_title("SSIM map", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pd", required=True, help="npz stem of a PD volume")
    p.add_argument("--pdfs", required=True, help="npz stem of a PDFS volume")
    p.add_argument("--out", default="docs/figures/ssim_diagnosis.png")
    a = p.parse_args()
    draw({"PD": a.pd, "PDFS": a.pdfs}, a.out)


if __name__ == "__main__":
    main()
