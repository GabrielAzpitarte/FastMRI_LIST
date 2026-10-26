"""Exponential moving average of weights with the EDM half-life schedule (NVlabs/edm training_loop.py)."""
import torch


def ema_beta(batch: int, nimg: int, halflife_kimg: float, rampup_ratio: float | None) -> float:
    """Per-step EMA decay for a half-life measured in thousands of images."""
    ema_nimg = halflife_kimg * 1000
    if rampup_ratio is not None:
        ema_nimg = min(ema_nimg, nimg * rampup_ratio)
    return 0.5 ** (batch / max(ema_nimg, 1e-8))


@torch.no_grad()
def update_ema(ema: torch.nn.Module, net: torch.nn.Module, beta: float) -> None:
    """ema <- beta * ema + (1 - beta) * net."""
    ema_params = [p for p in ema.parameters()]
    net_params = [p.detach() for p in net.parameters()]
    torch._foreach_lerp_(ema_params, net_params, 1 - beta)
