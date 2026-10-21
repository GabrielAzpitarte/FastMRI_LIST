"""DDPM++ building blocks as configured in EDM (Karras et al. 2022)."""
import math

import torch
import torch.nn.functional as F
from torch import nn

SKIP_SCALE = math.sqrt(0.5)


def xavier_uniform(shape, fan_in: int, fan_out: int) -> torch.Tensor:
    """EDM xavier-uniform initialisation."""
    return math.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)


class Linear(nn.Module):
    """Fully connected layer that runs in the input dtype."""

    def __init__(self, in_features: int, out_features: int, init_weight: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(xavier_uniform([out_features, in_features], in_features, out_features) * init_weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


class Conv2d(nn.Module):
    """Convolution with optional nearest 2x upsampling or 2x2 average downsampling first."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 up: bool = False, down: bool = False, init_weight: float = 1.0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.up, self.down = up, down
        fan_in, fan_out = in_channels * kernel ** 2, out_channels * kernel ** 2
        shape = [out_channels, in_channels, kernel, kernel]
        self.weight = nn.Parameter(xavier_uniform(shape, fan_in, fan_out) * init_weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            x = F.avg_pool2d(x, 2)
        w = self.weight.to(x.dtype)
        return F.conv2d(x, w, self.bias.to(x.dtype), padding=w.shape[-1] // 2)


class GroupNorm(nn.Module):
    """Group normalisation with EDM group sizing."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = min(32, num_channels // 4)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)


class Attention(nn.Module):
    """Single-head self-attention over spatial positions (EDM qkv layout)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.qkv = Conv2d(channels, channels * 3, 1, init_weight=math.sqrt(0.2))
        self.proj = Conv2d(channels, channels, 1, init_weight=1e-5)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, c, 3, h * w).unbind(2)
        q, k, v = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        return self.proj(a.squeeze(1).transpose(1, 2).reshape(b, c, h, w))


class UNetBlock(nn.Module):
    """DDPM++ residual block with optional resampling and attention."""

    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 up: bool = False, down: bool = False, attention: bool = False, dropout: float = 0.0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.dropout = dropout
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = Conv2d(in_channels, out_channels, 3, up=up, down=down)
        self.affine = Linear(emb_channels, out_channels)
        self.norm1 = GroupNorm(out_channels)
        self.conv1 = Conv2d(out_channels, out_channels, 3, init_weight=1e-5)
        resample = in_channels != out_channels or up or down
        self.skip = Conv2d(in_channels, out_channels, 1, up=up, down=down) if resample else None
        self.attn = Attention(out_channels) if attention else None

    def forward(self, x, emb):
        h = self.conv0(F.silu(self.norm0(x)))
        h = F.silu(self.norm1(h + self.affine(emb)[:, :, None, None].to(h.dtype)))
        h = self.conv1(F.dropout(h, p=self.dropout, training=self.training))
        x = (h + (x if self.skip is None else self.skip(x))) * SKIP_SCALE
        if self.attn is not None:
            x = (x + self.attn(x)) * SKIP_SCALE
        return x


def positional_embedding(sigma_labels: torch.Tensor, channels: int, max_positions: int = 10000) -> torch.Tensor:
    """EDM positional noise embedding (endpoint=True), ordered [sin, cos]."""
    freqs = torch.arange(channels // 2, dtype=torch.float32, device=sigma_labels.device)
    freqs = (1 / max_positions) ** (freqs / (channels // 2 - 1))
    x = sigma_labels.float().outer(freqs)
    return torch.cat([x.sin(), x.cos()], dim=1)
