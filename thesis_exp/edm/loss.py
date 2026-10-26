"""EDM training loss (Karras et al. 2022, Eq. 8, Table 1)."""
import torch


def sample_sigma(n: int, p_mean: float, p_std: float, device) -> torch.Tensor:
    """Log-normal training noise levels shaped (n, 1, 1, 1)."""
    return (torch.randn(n, 1, 1, 1, device=device) * p_std + p_mean).exp()


def loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM loss weighting lambda(sigma)."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def edm_loss(net, images: torch.Tensor, p_mean: float = -1.2, p_std: float = 1.2,
             sigma_data: float = 0.5) -> torch.Tensor:
    """Weighted denoising error per sample, summed over pixels."""
    sigma = sample_sigma(images.shape[0], p_mean, p_std, images.device)
    noisy = images + torch.randn_like(images) * sigma
    err = loss_weight(sigma, sigma_data) * (net(noisy, sigma) - images) ** 2
    return err.flatten(1).sum(1)
