import math

import pytest
import torch
import torch.nn.functional as F

from fastmri_edm.edm.ema import ema_beta, update_ema
from fastmri_edm.edm.layers import Attention
from fastmri_edm.edm.loss import loss_weight
from fastmri_edm.edm.precond import EDMPrecond
from fastmri_edm.edm.sampler import heun_sample, sigma_steps
from fastmri_edm.edm.unet import SongUNet


def tiny_unet(size=32):
    return SongUNet(size, 2, 2, channels=16, channel_mult=(1, 2, 2), num_blocks=1, attn_resolutions=(8,))


class GaussianDenoiser(torch.nn.Module):
    """Exact denoiser for data ~ N(0, s^2 I)."""

    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x, sigma):
        sigma = torch.as_tensor(sigma, dtype=x.dtype)
        return x * self.s ** 2 / (self.s ** 2 + sigma ** 2)


def test_precond_coefficients():
    pre = EDMPrecond(tiny_unet(), sigma_data=0.5)
    c_skip, c_out, c_in, _ = pre.coefficients(torch.tensor([[[[0.5]]]]))
    assert torch.allclose(c_skip, torch.tensor(0.5)) and torch.allclose(c_out, torch.tensor(0.5 / math.sqrt(2)))
    assert torch.allclose(c_in, torch.tensor(1 / math.sqrt(0.5)))


def test_denoiser_is_identity_at_tiny_sigma():
    pre = EDMPrecond(tiny_unet(), sigma_data=0.5)
    x = torch.randn(2, 2, 32, 32)
    assert torch.allclose(pre(x, 1e-6), x, atol=1e-4)


def test_unet_shape_and_near_zero_init():
    net = tiny_unet()
    out = net(torch.randn(3, 2, 32, 32), torch.zeros(3))
    assert out.shape == (3, 2, 32, 32)
    assert out.abs().max() < 1e-2


def test_unet_handles_scalar_sigma_broadcast():
    pre = EDMPrecond(tiny_unet(), sigma_data=0.5)
    x = torch.randn(3, 2, 32, 32)
    assert torch.allclose(pre(x, torch.tensor(0.7)), pre(x, torch.full((3,), 0.7)), atol=1e-6)


def test_attention_matches_edm_reference():
    torch.manual_seed(0)
    attn = Attention(16)
    torch.nn.init.normal_(attn.proj.weight)
    x = torch.randn(2, 16, 4, 4)
    q, k, v = attn.qkv(attn.norm(x)).reshape(2, 16, 3, 16).unbind(2)
    w = torch.einsum("ncq,nck->nqk", q, k / math.sqrt(16)).softmax(dim=2)
    ref = attn.proj(torch.einsum("nqk,nck->ncq", w, v).reshape(2, 16, 4, 4))
    assert torch.allclose(attn(x), ref, atol=1e-5)


def test_loss_weight_formula():
    s = torch.tensor([0.1, 0.5, 2.0])
    assert torch.allclose(loss_weight(s, 0.5), (s ** 2 + 0.25) / (s * 0.5) ** 2)


def test_sigma_steps_endpoints():
    t = sigma_steps(18)
    assert t[0].item() == pytest.approx(80.0) and t[-2].item() == pytest.approx(0.002) and t[-1] == 0
    assert (t[1:] < t[:-1]).all()


def test_heun_sampler_recovers_gaussian_scale():
    torch.manual_seed(0)
    out = heun_sample(GaussianDenoiser(0.3), torch.randn(4, 2, 64, 64), num_steps=32)
    assert abs(out.std().item() - 0.3) < 0.01


def test_ema_beta_and_update():
    assert ema_beta(64, 10_000_000, 500, 0.05) == pytest.approx(0.5 ** (64 / 500_000))
    assert ema_beta(64, 1000, 500, 0.05) == pytest.approx(0.5 ** (64 / 50))
    ema, net = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    expected = 0.9 * ema.weight.detach() + 0.1 * net.weight.detach()
    update_ema(ema, net, 0.9)
    assert torch.allclose(ema.weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bf16_channels_last_matches_fp32():
    torch.manual_seed(0)
    unet = tiny_unet().cuda()
    for p in unet.parameters():
        p.data.normal_(0, 0.05)
    x, sigma = torch.randn(2, 2, 32, 32, device="cuda"), torch.tensor(1.0)
    ref = EDMPrecond(unet, 0.5)(x, sigma)
    fast = EDMPrecond(unet.to(memory_format=torch.channels_last), 0.5, torch.bfloat16, True)(x, sigma)
    assert F.mse_loss(fast, ref) / ref.var() < 1e-3
