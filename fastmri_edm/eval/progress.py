"""Cheap progress checks run on rank 0 during training."""
from pathlib import Path

import torch

from ..data.h5io import list_volumes
from ..edm.sampler import heun_sample
from ..mri.image import to_complex
from ..train.stats import JsonlLog
from .heldout import heldout_mse, load_heldout
from .images import save_grid
from .pipeline import evaluate_volume, summarize


@torch.no_grad()
def sample_png(net, count: int, size: int, steps: int, path, seed: int = 0) -> None:
    """Save magnitudes of unconditional samples as a PNG grid."""
    device = next(net.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((count, 2, size, size), generator=g, device=device)
    save_grid(to_complex(heun_sample(net, noise, steps)).abs().cpu().numpy(), path)


def recon_summary(net, paths, cfg, gain: float, device) -> dict:
    """Mean PSNR/SSIM of EDM and zero-filled reconstructions per acceleration."""
    out = {}
    for acc, cf in zip(cfg.recon.accelerations, cfg.recon.center_fractions):
        s = summarize([evaluate_volume(net, p, cfg, acc, cf, gain, device)[0] for p in paths])
        keys = ("edm/psnr", "edm/ssim", "zero_filled/psnr", "zero_filled/ssim")
        out[f"{acc}x"] = {k: round(s[k]["mean"], 4) for k in keys}
    return out


class Probe:
    """Held-out denoising error, reconstruction metrics and sample grids for the EMA network."""

    def __init__(self, cfg, gain: float, device, run_dir):
        self.cfg, self.gain, self.device = cfg, gain, device
        self.images = load_heldout(cfg.data.val_cache, gain, cfg.probe.heldout_slices, device)
        self.paths = list_volumes(cfg.data.val_raw)[:cfg.probe.recon_volumes]
        self.log = JsonlLog(Path(run_dir) / "probe.jsonl")
        self.sample_dir = Path(run_dir) / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def run(self, net, nimg: int) -> None:
        p = self.cfg.probe
        mse = heldout_mse(net, self.images, p.sigmas)
        recon = recon_summary(net, self.paths, self.cfg, self.gain, self.device) if self.paths else {}
        path = self.sample_dir / f"samples-{nimg // 1000:08d}.png"
        sample_png(net, p.num_samples, self.cfg.data.size, p.sample_steps, path)
        self.log.write(kimg=round(nimg / 1000, 1), heldout_mse=mse, recon=recon)

    def try_run(self, net, nimg: int) -> None:
        """run(), but log failures instead of raising so other ranks never wait on a dead probe."""
        try:
            self.run(net, nimg)
        except Exception as e:
            torch.cuda.empty_cache()
            self.log.write(kimg=round(nimg / 1000, 1), probe_error=repr(e))
