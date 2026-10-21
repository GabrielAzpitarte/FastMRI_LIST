"""Held-out denoising error at fixed noise levels with fixed noise."""
from pathlib import Path

import numpy as np
import torch


def load_heldout(val_cache, gain: float, count: int, device) -> torch.Tensor:
    """Evenly spaced validation slices, normalized like the training data."""
    images = np.load(Path(val_cache) / "images.npy", mmap_mode="r")
    idx = np.linspace(0, len(images) - 1, count).round().astype(int)
    return torch.from_numpy(np.ascontiguousarray(images[idx])).to(device) * gain


def denoise_error(net, clean: torch.Tensor, sigma: float, generator) -> torch.Tensor:
    """Per-image MSE of D(clean + sigma * n; sigma)."""
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    return ((net(clean + sigma * noise, sigma) - clean) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def heldout_mse(net, images: torch.Tensor, sigmas, seed: int = 0, batch: int = 16) -> dict:
    """Mean denoising MSE per noise level, with identical noise across calls."""
    g = torch.Generator(device=images.device)
    out = {}
    for sigma in sigmas:
        g.manual_seed(seed)
        errs = [denoise_error(net, chunk, sigma, g) for chunk in images.split(batch)]
        out[f"{sigma:g}"] = torch.cat(errs).mean().item()
    return out
