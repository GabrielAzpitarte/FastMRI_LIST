"""Construction of the denoiser and optimizer from a config."""
import torch

from ..edm.precond import EDMPrecond
from ..edm.unet import SongUNet

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def configure_backends() -> None:
    """Fast cuDNN/TF32 defaults."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_net(cfg) -> EDMPrecond:
    """EDM-preconditioned DDPM++ for 2-channel complex slices."""
    m = cfg.model
    unet = SongUNet(cfg.data.size, 2, 2, m.channels, m.channel_mult, m.num_blocks,
                    m.attn_resolutions, m.dropout)
    net = EDMPrecond(unet, cfg.edm.sigma_data, DTYPES[cfg.train.precision], cfg.train.channels_last)
    if cfg.train.channels_last:
        net = net.to(memory_format=torch.channels_last)
    return net


def build_optimizer(net: torch.nn.Module, cfg) -> torch.optim.Adam:
    """Adam with EDM betas."""
    return torch.optim.Adam(net.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999), eps=1e-8, fused=True)


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate of every parameter group."""
    for group in opt.param_groups:
        group["lr"] = lr


def learning_rate(base: float, nimg: int, rampup_kimg: float) -> float:
    """Linear learning-rate warm-up (EDM)."""
    return base * min(nimg / max(rampup_kimg * 1000, 1e-8), 1.0)


def sanitize_grads(net: torch.nn.Module) -> None:
    """Replace non-finite gradients as EDM does."""
    for p in net.parameters():
        if p.grad is not None:
            torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)


def num_params(net: torch.nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
