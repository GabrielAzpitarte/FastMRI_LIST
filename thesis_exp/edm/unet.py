"""DDPM++ U-Net (EDM SongUNet with standard encoder and decoder)."""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import Conv2d, GroupNorm, Linear, UNetBlock, positional_embedding


class SongUNet(nn.Module):
    """Unconditional DDPM++ denoiser backbone."""

    def __init__(self, resolution: int, in_channels: int, out_channels: int, channels: int = 128,
                 channel_mult=(1, 2, 2, 2), num_blocks: int = 4, attn_resolutions=(16,), dropout: float = 0.1):
        super().__init__()
        self.noise_channels = channels
        emb = channels * 4
        self.map0 = Linear(channels, emb)
        self.map1 = Linear(emb, emb)
        block = dict(emb_channels=emb, dropout=dropout)
        self.enc, skips, cout = self._encoder(resolution, in_channels, channels, channel_mult,
                                              num_blocks, attn_resolutions, block)
        self.dec, cout = self._decoder(resolution, cout, skips, channels, channel_mult,
                                       num_blocks, attn_resolutions, block)
        self.out_norm = GroupNorm(cout)
        self.out_conv = Conv2d(cout, out_channels, 3, init_weight=1e-5)

    @staticmethod
    def _encoder(resolution, in_channels, channels, channel_mult, num_blocks, attn_resolutions, block):
        enc = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = resolution >> level
            if level == 0:
                enc[f"{res}x{res}_conv"] = Conv2d(cout, channels, 3)
                cout = channels
            else:
                enc[f"{res}x{res}_down"] = UNetBlock(cout, cout, down=True, **block)
            for idx in range(num_blocks):
                cin, cout = cout, channels * mult
                enc[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=res in attn_resolutions, **block)
        skips = [m.out_channels for m in enc.values()]
        return enc, skips, cout

    @staticmethod
    def _decoder(resolution, cout, skips, channels, channel_mult, num_blocks, attn_resolutions, block):
        dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = resolution >> level
            if level == len(channel_mult) - 1:
                dec[f"{res}x{res}_in0"] = UNetBlock(cout, cout, attention=True, **block)
                dec[f"{res}x{res}_in1"] = UNetBlock(cout, cout, **block)
            else:
                dec[f"{res}x{res}_up"] = UNetBlock(cout, cout, up=True, **block)
            for idx in range(num_blocks + 1):
                cin, cout = cout + skips.pop(), channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                dec[f"{res}x{res}_block{idx}"] = UNetBlock(cin, cout, attention=attn, **block)
        return dec, cout

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = positional_embedding(noise_labels, self.noise_channels)
        emb = F.silu(self.map1(F.silu(self.map0(emb))))
        skips = []
        for name, block in self.enc.items():
            x = block(x) if name.endswith("_conv") else block(x, emb)
            skips.append(x)
        for name, block in self.dec.items():
            if "_block" in name:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        return self.out_conv(F.silu(self.out_norm(x)))
