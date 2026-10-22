"""Retrospective undersampling and reconstruction of validation volumes."""
from pathlib import Path

import numpy as np
import torch

from ..data.h5io import read_attrs, read_kspace, read_target
from ..mri.image import kspace_to_image, to_complex
from ..mri.masks import file_seed, random_mask
from ..mri.operator import measure, zero_filled
from ..mri.scaling import volume_scale
from .metrics import volume_metrics
from .recon import dc_sample


def central_slices(n: int, count: int | None) -> slice:
    """The `count` central slices, or all slices when count is falsy."""
    if not count or count >= n:
        return slice(0, n)
    start = (n - count) // 2
    return slice(start, start + count)


def undersample(path: Path, size: int, acceleration: int, center_fraction: float):
    """Undersampled k-space of the cropped images, its mask and the measurement-based scale."""
    image = kspace_to_image(torch.from_numpy(read_kspace(path)), size)
    mask = torch.from_numpy(random_mask(size, acceleration, center_fraction, file_seed(path.name)))
    kspace = measure(image, mask)
    return kspace, mask, volume_scale(kspace)


def posterior_samples(net, kspace, mask, scale: float, gain: float, recon_cfg, device, seed: int) -> torch.Tensor:
    """Magnitude posterior samples (num_samples, slices, H, W) in the original intensity units."""
    g = torch.Generator(device=device).manual_seed(seed)
    y, m = kspace.to(device) * (gain / scale), mask.to(device)
    sigma_min = getattr(recon_cfg, "sigma_min", 0.002)
    mags = [to_complex(dc_sample(net, y, m, recon_cfg.num_steps, recon_cfg.eta, g, sigma_min)).abs()
            for _ in range(getattr(recon_cfg, "num_samples", 1))]
    return torch.stack(mags) * (scale / gain)


def reconstruct(net, kspace, mask, scale: float, gain: float, recon_cfg, device, seed: int) -> np.ndarray:
    """Magnitude reconstructions in the original intensity units, averaged over posterior samples."""
    return posterior_samples(net, kspace, mask, scale, gain, recon_cfg, device, seed).mean(0).cpu().numpy()


def evaluate_volume(net, path: Path, cfg, acceleration: int, center_fraction: float,
                    gain: float, device, seed: int = 0) -> tuple[dict, dict]:
    """Metrics of the EDM and zero-filled reconstructions, plus the images."""
    kspace, mask, scale = undersample(path, cfg.data.size, acceleration, center_fraction)
    sl = central_slices(kspace.shape[0], cfg.recon.slices)
    recon = reconstruct(net, kspace[sl], mask, scale, gain, cfg.recon, device, seed)
    zf = zero_filled(kspace[sl]).abs().numpy()
    target = read_target(path)[sl]
    maxval = float(read_attrs(path)["max"])
    metrics = dict(name=path.name, slices=[sl.start, sl.stop], edm=volume_metrics(target, recon, maxval),
                   zero_filled=volume_metrics(target, zf, maxval))
    return metrics, dict(target=target, zero_filled=zf, edm=recon)


def summarize(rows: list[dict]) -> dict:
    """Mean and standard error over volumes for every method and metric."""
    out = {}
    for method in ("edm", "zero_filled"):
        for key in ("nmse", "psnr", "ssim"):
            v = np.array([r[method][key] for r in rows])
            out[f"{method}/{key}"] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    return out
