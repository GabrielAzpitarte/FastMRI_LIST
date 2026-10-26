"""EDM preconditioning (Karras et al. 2022, Eq. 7, Table 1): D(x; sigma) = c_skip x + c_out F(c_in x; c_noise)."""
import torch
from torch import nn


class EDMPrecond(nn.Module):
    """Wraps a raw network into the EDM denoiser D(x; sigma)."""

    def __init__(self, model: nn.Module, sigma_data: float = 0.5,
                 dtype: torch.dtype = torch.float32, channels_last: bool = False):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.dtype = dtype
        self.memory_format = torch.channels_last if channels_last else torch.contiguous_format

    def coefficients(self, sigma: torch.Tensor):
        """c_skip, c_out, c_in, c_noise for noise levels shaped (B, 1, 1, 1)."""
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1 / (sd2 + sigma ** 2).sqrt()
        return c_skip, c_out, c_in, sigma.log().flatten() / 4

    def forward(self, x: torch.Tensor, sigma) -> torch.Tensor:
        x = x.float()
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1, 1, 1, 1)
        sigma = sigma.expand(x.shape[0], 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.coefficients(sigma)
        inp = (c_in * x).to(dtype=self.dtype, memory_format=self.memory_format)
        return c_skip * x + c_out * self.model(inp, c_noise).float()
