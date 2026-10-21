"""Per-volume intensity scale computed only from always-sampled k-space columns."""
import torch

from .fft import ifft2c
from .masks import center_columns

ACS_FRACTION = 0.04


def lowres_magnitude(kspace: torch.Tensor, fraction: float = ACS_FRACTION) -> torch.Tensor:
    """Magnitude image reconstructed from the central k-space columns only."""
    cols = center_columns(kspace.shape[-1], fraction)
    low = torch.zeros_like(kspace)
    low[..., cols] = kspace[..., cols]
    return ifft2c(low).abs()


def volume_scale(kspace: torch.Tensor) -> float:
    """Max low-resolution magnitude over all slices of a volume."""
    return lowres_magnitude(kspace).amax().item()
