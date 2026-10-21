"""Centered orthonormal 2D FFTs over the last two dimensions."""
import torch

DIMS = (-2, -1)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT."""
    x = torch.fft.ifftshift(x, dim=DIMS)
    x = torch.fft.fft2(x, dim=DIMS, norm="ortho")
    return torch.fft.fftshift(x, dim=DIMS)


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT."""
    k = torch.fft.ifftshift(k, dim=DIMS)
    k = torch.fft.ifft2(k, dim=DIMS, norm="ortho")
    return torch.fft.fftshift(k, dim=DIMS)
