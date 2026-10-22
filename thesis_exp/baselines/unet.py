"""fastMRI U-Net (facebookresearch/fastMRI, MIT license), with state-dict keys matching the official weights."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(cin: int, cout: int) -> nn.Module:
    """Two 3x3 convolutions, each followed by instance norm and LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, True), nn.Dropout2d(0.0),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, True), nn.Dropout2d(0.0))


def up_block(cin: int, cout: int) -> nn.Module:
    """2x transposed convolution, instance norm and LeakyReLU."""
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, True))


class Block(nn.Module):
    """Wrapper that reproduces the `layers` attribute of the official blocks."""

    def __init__(self, layers: nn.Module):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Unet(nn.Module):
    """Magnitude-image U-Net with average pooling and skip concatenation."""

    def __init__(self, chans: int = 256, num_pool_layers: int = 4, in_chans: int = 1, out_chans: int = 1):
        super().__init__()
        self.down_sample_layers = nn.ModuleList([Block(conv_block(in_chans, chans))])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(Block(conv_block(ch, ch * 2)))
            ch *= 2
        self.conv = Block(conv_block(ch, ch * 2))
        self.up_conv, self.up_transpose_conv = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(Block(up_block(ch * 2, ch)))
            self.up_conv.append(Block(conv_block(ch * 2, ch)))
            ch //= 2
        self.up_transpose_conv.append(Block(up_block(ch * 2, ch)))
        self.up_conv.append(nn.Sequential(Block(conv_block(ch * 2, ch)), nn.Conv2d(ch, out_chans, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack = []
        for layer in self.down_sample_layers:
            x = layer(x)
            stack.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.conv(x)
        for up, conv in zip(self.up_transpose_conv, self.up_conv):
            skip = stack.pop()
            x = up(x)
            pad = [0, int(x.shape[-1] != skip.shape[-1]), 0, int(x.shape[-2] != skip.shape[-2])]
            if any(pad):
                x = F.pad(x, pad, "reflect")
            x = conv(torch.cat([x, skip], dim=1))
        return x


def load_pretrained(path: Path, device) -> Unet:
    """Official single-coil knee leaderboard U-Net (256 channels, 4 pooling layers)."""
    net = Unet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval().requires_grad_(False)


def normalize(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-slice mean/std normalization clamped to [-6, 6], as in fastMRI's UnetDataTransform."""
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True)
    return ((image - mean) / (std + 1e-11)).clamp(-6, 6), mean, std


@torch.inference_mode()
def reconstruct(net: Unet, zf_mag: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """U-Net reconstructions of zero-filled magnitude slices (S, H, W), in the input units."""
    out = []
    for chunk in zf_mag.split(batch):
        x, mean, std = normalize(chunk)
        out.append(net(x.unsqueeze(1)).squeeze(1) * std + mean)
    return torch.cat(out)
