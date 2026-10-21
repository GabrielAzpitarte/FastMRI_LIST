"""Single-coil Cartesian measurement operator on cropped complex images."""
import torch

from .fft import fft2c, ifft2c


def measure(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Undersampled k-space of a complex image (mask broadcasts over rows)."""
    return fft2c(image) * mask


def zero_filled(kspace: torch.Tensor) -> torch.Tensor:
    """Zero-filled complex reconstruction."""
    return ifft2c(kspace)


def project(image: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the sampled k-space entries of an image with the measurements."""
    return ifft2c(torch.where(mask, kspace, fft2c(image)))
