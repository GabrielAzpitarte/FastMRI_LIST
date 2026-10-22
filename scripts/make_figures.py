"""Side-by-side figures on test volumes: ground truth, zero-filled, pretrained U-Net and EDM, plus EDM uncertainty.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/make_figures.py --ckpt runs/main/ema-00004108.pt --out docs/figures
Volumes are fixed by rule (first PD and first PDFS volume after the 10 tuning volumes), not picked by eye.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thesis_exp.baselines.unet import load_pretrained, reconstruct as unet_reconstruct  # noqa: E402
from thesis_exp.data.h5io import list_volumes, read_attrs, read_kspace, read_target  # noqa: E402
from thesis_exp.eval.metrics import psnr, ssim  # noqa: E402
from thesis_exp.eval.pipeline import posterior_samples, undersample  # noqa: E402
from thesis_exp.eval.snapshot import load_ema  # noqa: E402
from thesis_exp.eval.unet_pipeline import cropped_zero_filled  # noqa: E402
from thesis_exp.mri.image import kspace_to_image  # noqa: E402
from thesis_exp.train.build import configure_backends  # noqa: E402

UNET_WEIGHTS = Path("data/pretrained/fastmri_unet/knee_sc_leaderboard_state_dict.pt")
CENTER = {4: 0.08, 8: 0.04}


def pick_volumes(raw: str, skip: int = 10) -> list[Path]:
    """First PD and first PDFS volume of the test split."""
    out = {}
    for path in list_volumes(raw)[skip:]:
        out.setdefault(read_attrs(path)["acquisition"], path)
    return [out["CORPD_FBK"], out["CORPDFS_FBK"]]


def central_slice(path: Path, edm, unet, acc: int, device) -> dict:
    """Images of the central slice for every method, plus the EDM posterior samples."""
    net, cfg, gain = edm
    target = read_target(path)
    mid = len(target) // 2
    kspace, mask, scale = undersample(path, cfg.data.size, acc, CENTER[acc])
    samples = posterior_samples(net, kspace[mid:mid + 1], mask, scale, gain, cfg.recon, device, seed=0)[:, 0].cpu().numpy()
    image = kspace_to_image(torch.from_numpy(read_kspace(path))[mid:mid + 1], cfg.data.size)
    zf = cropped_zero_filled(image, path.name, cfg.data.size, acc, CENTER[acc])
    return dict(target=target[mid], zero_filled=zf[0].numpy(), unet=unet_reconstruct(unet, zf.to(device))[0].cpu().numpy(),
                edm=samples.mean(0), sample=samples[0], std=samples.std(0), mask=np.tile(mask.numpy(), (cfg.data.size, 1)),
                maxval=float(read_attrs(path)["max"]))


def scores(im: dict, key: str) -> str:
    """Slice PSNR / SSIM against the ground truth (volume maximum as data range)."""
    gt, pred = im["target"][None], im[key][None]
    return f"{psnr(gt, pred, im['maxval']):.2f} dB / {ssim(gt, pred, im['maxval']):.3f}"


def show(ax, img, hi: float, title: str, cmap: str = "gray") -> None:
    """One panel without axes."""
    ax.imshow(img, cmap=cmap, vmin=0, vmax=hi)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def comparison_figure(images: list[dict], names: list[str], acc: int, path: Path) -> None:
    """Per volume: images (GT, ZF, U-Net, EDM) and 5x absolute errors with the sampling mask."""
    fig, axes = plt.subplots(2 * len(images), 4, figsize=(12, 6.2 * len(images)))
    for v, (im, name) in enumerate(zip(images, names)):
        hi = float(np.percentile(im["target"], 99.5))
        top, bottom = axes[2 * v], axes[2 * v + 1]
        show(top[0], im["target"], hi, f"{name}: ground truth")
        show(bottom[0], im["mask"], 1, f"{acc}x sampling mask")
        for col, (key, label) in enumerate((("zero_filled", "Zero-filled"), ("unet", "U-Net (pretrained)"),
                                            ("edm", "EDM, mean of 8 (ours)")), start=1):
            show(top[col], im[key], hi, f"{label}\n{scores(im, key)}")
            show(bottom[col], 5 * np.abs(im[key] - im["target"]), hi, f"|error| x5, {label.split(',')[0]}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def uncertainty_figure(images: list[dict], names: list[str], acc: int, path: Path) -> None:
    """Per volume: GT, one posterior sample, the posterior mean, the pixel-wise std and the mean's error."""
    fig, axes = plt.subplots(len(images), 5, figsize=(15, 3.8 * len(images)))
    for row, im, name in zip(axes, images, names):
        hi = float(np.percentile(im["target"], 99.5))
        show(row[0], im["target"], hi, f"{name}: ground truth")
        show(row[1], im["sample"], hi, f"one EDM sample\n{scores(im, 'sample')}")
        show(row[2], im["edm"], hi, f"mean of 8 samples\n{scores(im, 'edm')}")
        show(row[3], 5 * im["std"], hi, "pixel std over 8 samples x5", cmap="magma")
        show(row[4], 5 * np.abs(im["edm"] - im["target"]), hi, "|error| of the mean x5", cmap="magma")
    fig.suptitle(f"EDM posterior at {acc}x: samples, mean and uncertainty", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ema-00004108.pt")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--raw", default="data/raw/singlecoil_val")
    a = p.parse_args()
    configure_backends()
    device = torch.device("cuda", 0)
    net, cfg, gain, _ = load_ema(a.ckpt, ["train.channels_last=false", "recon.eta=1.0", "recon.num_samples=8"], device)
    unet = load_pretrained(UNET_WEIGHTS, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = pick_volumes(a.raw)
    names = [f"{p.stem} ({'PD' if read_attrs(p)['acquisition'] == 'CORPD_FBK' else 'PDFS'})" for p in paths]
    for acc in (4, 8):
        images = [central_slice(path, (net, cfg, gain), unet, acc, device) for path in paths]
        comparison_figure(images, names, acc, out / f"comparison_{acc}x.png")
        uncertainty_figure(images, names, acc, out / f"uncertainty_{acc}x.png")
        print(f"{acc}x done")


if __name__ == "__main__":
    main()
