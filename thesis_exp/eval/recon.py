"""Accelerated MRI reconstruction with the EDM prior and hard k-space data consistency."""
import torch

from ..edm.sampler import sigma_steps
from ..mri.image import to_channels, to_complex
from ..mri.operator import project


def consistent_denoise(net, x: torch.Tensor, sigma, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Denoised estimate with its sampled k-space entries replaced by the measurements."""
    return to_channels(project(to_complex(net(x, sigma)), kspace, mask))


def renoise(x, x0, t_cur, t_next, eta: float, generator) -> torch.Tensor:
    """Step from noise level t_cur to t_next around x0 (eta=0: DDIM, eta=1: fresh noise)."""
    eps = (x - x0) / t_cur
    z = torch.randn(x.shape, generator=generator, device=x.device)
    return x0 + t_next * ((1 - eta ** 2) ** 0.5 * eps + eta * z)


@torch.no_grad()
def dc_sample(net, kspace: torch.Tensor, mask: torch.Tensor, num_steps: int = 100,
              eta: float = 0.5, generator=None) -> torch.Tensor:
    """Reconstruction (B, 2, H, W) consistent with undersampled k-space (B, H, W)."""
    t = sigma_steps(num_steps, device=kspace.device).float()
    shape = (kspace.shape[0], 2, *kspace.shape[-2:])
    x = torch.randn(shape, generator=generator, device=kspace.device) * t[0]
    for t_cur, t_next in zip(t[:-1], t[1:]):
        x0 = consistent_denoise(net, x, t_cur, kspace, mask)
        x = renoise(x, x0, t_cur, t_next, eta, generator)
    return x0
