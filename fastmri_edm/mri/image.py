"""Image-domain helpers for complex MRI slices."""
import torch

from .fft import ifft2c


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Square crop of the last two dimensions around the center (fastMRI convention)."""
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def kspace_to_image(kspace: torch.Tensor, size: int) -> torch.Tensor:
    """Fully sampled k-space to the cropped complex image."""
    return center_crop(ifft2c(kspace), size)


def to_channels(x: torch.Tensor) -> torch.Tensor:
    """Complex (..., H, W) to real (..., 2, H, W)."""
    return torch.stack([x.real, x.imag], dim=-3)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Real (..., 2, H, W) to complex (..., H, W)."""
    return torch.complex(x[..., 0, :, :].float(), x[..., 1, :, :].float())
