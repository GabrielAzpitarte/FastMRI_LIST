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
from thesis_exp.eval.recon import consistent_denoise, renoise  # noqa: E402
from thesis_exp.eval.unet_pipeline import cropped_zero_filled  # noqa: E402
from thesis_exp.edm.sampler import heun_sample, sigma_steps  # noqa: E402
from thesis_exp.mri.fft import fft2c, ifft2c  # noqa: E402
from thesis_exp.mri.image import kspace_to_image, to_complex  # noqa: E402
from thesis_exp.mri.masks import file_seed, random_mask  # noqa: E402
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


def zoom_figure(images: list[dict], names: list[str], acc: int, path: Path, rows=slice(80, 208), cols=slice(96, 224)) -> None:
    """Central 128x128 crop (the joint) for every method: smooth supervised output vs sharp diffusion output."""
    keys = (("target", "ground truth"), ("zero_filled", "zero-filled"), ("unet", "U-Net (pretrained)"),
            ("edm", "EDM mean of 8"), ("sample", "EDM one sample"))
    fig, axes = plt.subplots(len(images), len(keys), figsize=(15, 3.75 * len(images)))
    for row, im, name in zip(axes, images, names):
        hi = float(np.percentile(im["target"], 99.5))
        for ax, (key, label) in zip(row, keys):
            ax.imshow(im[key][rows, cols], cmap="gray", vmin=0, vmax=hi, interpolation="nearest")
            ax.set_title(label if key == "target" else f"{label}\n{scores(im, key)}", fontsize=9)
            ax.axis("off")
        row[0].set_title(f"{name}\nground truth", fontsize=9)
    fig.suptitle(f"Zoom on the joint at {acc}x (slice PSNR / SSIM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def kspace_figure(path: Path, size: int, out: Path) -> None:
    """Fully sampled k-space, the 4x and 8x masks applied to it, and the resulting zero-filled images."""
    image = kspace_to_image(torch.from_numpy(read_kspace(path))[len(read_target(path)) // 2], size)
    k = fft2c(image)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    logk = torch.log1p(k.abs() / k.abs().median()).numpy()
    hi = float(np.percentile(image.abs().numpy(), 99.5))
    show(axes[0, 0], logk, logk.max(), "k-space, fully sampled (log magnitude)", cmap="magma")
    show(axes[1, 0], image.abs().numpy(), hi, "image, fully sampled")
    for col, acc in enumerate((4, 8), start=1):
        mask = torch.from_numpy(random_mask(size, acc, CENTER[acc], file_seed(path.name)))
        show(axes[0, col], logk * mask.numpy(), logk.max(), f"{acc}x: {int(mask.sum())} of {size} lines measured", cmap="magma")
        show(axes[1, col], ifft2c(k * mask).abs().numpy(), hi, f"{acc}x zero-filled: aliasing and blur")
    fig.suptitle("Accelerated MRI: skip k-space lines, then reconstruct the missing information", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "kspace_and_masks.png", dpi=120)
    plt.close(fig)


def generated_vs_real(net, cfg, raw: str, out: Path, device, count: int = 8) -> None:
    """Unconditional EDM samples (Heun, 40 steps) next to the central slices of random validation volumes."""
    g = torch.Generator(device=device).manual_seed(1)
    noise = torch.randn((count, 2, cfg.data.size, cfg.data.size), generator=g, device=device)
    fake = to_complex(heun_sample(net, noise, 40)).abs().cpu().numpy()
    vols = list_volumes(raw)
    picks = sorted(np.random.default_rng(1).choice(len(vols), count, replace=False))
    real = [read_target(vols[i])[len(read_target(vols[i])) // 2] for i in picks]
    fig, axes = plt.subplots(2, count, figsize=(2.1 * count, 5.0), gridspec_kw=dict(hspace=0.12, wspace=0.05))
    for row, imgs in ((axes[0], real), (axes[1], fake)):
        for ax, img in zip(row, imgs):
            show(ax, img, float(np.percentile(img, 99.5)), "")
    fig.text(0.01, 0.70, "real", rotation=90, va="center", fontsize=12)
    fig.text(0.01, 0.29, "generated", rotation=90, va="center", fontsize=12)
    fig.suptitle("What the model learned: central slices of real knees (top) vs samples generated from pure noise (bottom)",
                 fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.9, bottom=0.02)
    fig.savefig(out / "generated_vs_real.png", dpi=120)
    plt.close(fig)


@torch.no_grad()
def trajectory_figure(net, cfg, gain, path: Path, acc: int, out: Path, device, shown=(0, 30, 50, 65, 80, 99)) -> None:
    """One 8x reconstruction run: the noisy state and its data-consistent denoised estimate at several steps."""
    target = read_target(path)
    mid = len(target) // 2
    kspace, mask, scale = undersample(path, cfg.data.size, acc, CENTER[acc])
    y, m = kspace[mid:mid + 1].to(device) * (gain / scale), mask.to(device)
    g = torch.Generator(device=device).manual_seed(0)
    t = sigma_steps(cfg.recon.num_steps, device=device).float()
    x = torch.randn((1, 2, *y.shape[-2:]), generator=g, device=device) * t[0]
    frames = []
    for i, (t_cur, t_next) in enumerate(zip(t[:-1], t[1:])):
        x0 = consistent_denoise(net, x, t_cur, y, m)
        if i in shown:
            frames.append((float(t_cur), to_complex(x)[0].abs().cpu().numpy(), to_complex(x0)[0].abs().cpu().numpy()))
        x = renoise(x, x0, t_cur, t_next, 1.0, g)
    hi = float(np.percentile(frames[-1][2], 99.5))
    fig, axes = plt.subplots(2, len(frames), figsize=(2.5 * len(frames), 6.0))
    for col, (sigma, noisy, clean) in enumerate(frames):
        show(axes[0, col], noisy, max(hi, float(np.percentile(noisy, 99.5))), f"step {shown[col] + 1}, σ = {sigma:.3g}")
        show(axes[1, col], clean, hi, "denoised + data consistency")
    axes[0, 0].set_ylabel("current state")
    fig.suptitle(f"One {acc}x reconstruction emerging from noise ({cfg.recon.num_steps} steps, η=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"trajectory_{acc}x.png", dpi=120)
    plt.close(fig)


def training_samples_figure(out: Path) -> None:
    """The probe's unconditional sample grids at 1, 2, 3 and 4 Mimg side by side."""
    grids = sorted((out / "samples_during_training").glob("samples-*.png"))
    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 4.6))
    for ax, grid in zip(np.atleast_1d(axes), grids):
        ax.imshow(plt.imread(grid), cmap="gray")
        ax.set_title(f"after {int(grid.stem.split('-')[1]) / 1000:.0f} Mimg", fontsize=11)
        ax.axis("off")
    fig.suptitle("Unconditional samples during training (same noise, 40 Heun steps)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "samples_during_training.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ema-00004108.pt")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--raw", default="data/raw/singlecoil_val")
    p.add_argument("--only", default="", help="comma-separated subset: kspace,training,generated,trajectory,recon")
    a = p.parse_args()
    want = set(a.only.split(",")) if a.only else {"kspace", "training", "generated", "trajectory", "recon"}
    configure_backends()
    device = torch.device("cuda", 0)
    net, cfg, gain, _ = load_ema(a.ckpt, ["train.channels_last=false", "recon.eta=1.0", "recon.num_samples=8"], device)
    unet = load_pretrained(UNET_WEIGHTS, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = pick_volumes(a.raw)
    names = [f"{p.stem} ({'PD' if read_attrs(p)['acquisition'] == 'CORPD_FBK' else 'PDFS'})" for p in paths]
    if "kspace" in want:
        kspace_figure(paths[0], cfg.data.size, out)
    if "training" in want:
        training_samples_figure(out)
    if "generated" in want:
        generated_vs_real(net, cfg, a.raw, out, device)
    if "trajectory" in want:
        trajectory_figure(net, cfg, gain, paths[0], 8, out, device)
    for acc in (4, 8) if "recon" in want else ():
        images = [central_slice(path, (net, cfg, gain), unet, acc, device) for path in paths]
        comparison_figure(images, names, acc, out / f"comparison_{acc}x.png")
        uncertainty_figure(images, names, acc, out / f"uncertainty_{acc}x.png")
        zoom_figure(images, names, acc, out / f"zoom_{acc}x.png")
        print(f"{acc}x done")


if __name__ == "__main__":
    main()
