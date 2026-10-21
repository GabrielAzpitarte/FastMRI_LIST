"""Cheap progress checks run on rank 0 during training."""
from pathlib import Path

import torch

from ..edm.sampler import heun_sample
from ..mri.image import to_complex
from ..train.stats import JsonlLog
from .heldout import heldout_mse, load_heldout
from .images import save_grid


@torch.no_grad()
def sample_png(net, count: int, size: int, steps: int, path, seed: int = 0) -> None:
    """Save magnitudes of unconditional samples as a PNG grid."""
    device = next(net.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((count, 2, size, size), generator=g, device=device)
    save_grid(to_complex(heun_sample(net, noise, steps)).abs().cpu().numpy(), path)


class Probe:
    """Held-out denoising error and sample grids for the EMA network."""

    def __init__(self, cfg, gain: float, device, run_dir):
        self.cfg = cfg
        self.images = load_heldout(cfg.data.val_cache, gain, cfg.probe.heldout_slices, device)
        self.log = JsonlLog(Path(run_dir) / "probe.jsonl")
        self.sample_dir = Path(run_dir) / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def run(self, net, nimg: int) -> None:
        p = self.cfg.probe
        mse = heldout_mse(net, self.images, p.sigmas)
        path = self.sample_dir / f"samples-{nimg // 1000:08d}.png"
        sample_png(net, p.num_samples, self.cfg.data.size, p.sample_steps, path)
        self.log.write(kimg=round(nimg / 1000, 1), heldout_mse=mse)
