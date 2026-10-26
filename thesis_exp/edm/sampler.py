"""EDM noise schedule (Karras et al. 2022, Eq. 5) and deterministic Heun sampler (Algorithm 1)."""
import torch


def sigma_steps(num_steps: int, sigma_min: float = 0.002, sigma_max: float = 80.0,
                rho: float = 7.0, device=None) -> torch.Tensor:
    """Decreasing EDM noise levels followed by a final zero."""
    i = torch.arange(num_steps, dtype=torch.float64, device=device)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    t = (hi + i / (num_steps - 1) * (lo - hi)) ** rho
    return torch.cat([t, torch.zeros_like(t[:1])])


@torch.no_grad()
def heun_sample(net, noise: torch.Tensor, num_steps: int = 40, **schedule) -> torch.Tensor:
    """EDM second-order deterministic sampler started from unit Gaussian noise."""
    t = sigma_steps(num_steps, device=noise.device, **schedule)
    x = noise.double() * t[0]
    for i, (t_cur, t_next) in enumerate(zip(t[:-1], t[1:])):
        d_cur = (x - net(x, t_cur).double()) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if i < num_steps - 1:
            d_next = (x_next - net(x_next, t_next).double()) / t_next
            x_next = x + (t_next - t_cur) * (d_cur + d_next) / 2
        x = x_next
    return x.float()
