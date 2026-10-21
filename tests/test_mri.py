import numpy as np
import torch

from thesis_exp.mri.fft import fft2c, ifft2c
from thesis_exp.mri.image import center_crop, kspace_to_image, to_channels, to_complex
from thesis_exp.mri.operator import measure, project, zero_filled


def rand_complex(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_fft_roundtrip_and_energy():
    x = rand_complex(2, 16, 12)
    k = fft2c(x)
    assert torch.allclose(ifft2c(k), x, atol=1e-5)
    assert torch.allclose(k.abs().square().sum(), x.abs().square().sum(), rtol=1e-4)


def test_fft_matches_numpy_centered():
    x = rand_complex(10, 14)
    ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x.numpy()), norm="ortho"))
    assert np.allclose(fft2c(x).numpy(), ref, atol=1e-5)


def test_center_crop_fastmri_convention():
    x = torch.arange(640 * 368).reshape(640, 368)
    c = center_crop(x, 320)
    assert c.shape == (320, 320)
    assert c[0, 0] == x[160, 24]


def test_kspace_to_image_inverts_full_fft():
    img = rand_complex(3, 40, 36)
    assert torch.allclose(kspace_to_image(fft2c(img), 32), center_crop(img, 32), atol=1e-5)


def test_channels_roundtrip():
    z = rand_complex(2, 8, 8)
    assert torch.equal(to_complex(to_channels(z)), z)


def test_projection_enforces_measurements():
    x, truth = rand_complex(2, 16, 16), rand_complex(2, 16, 16)
    mask = torch.rand(16) < 0.4
    y = measure(truth, mask)
    k = fft2c(project(x, y, mask))
    assert torch.allclose(k[..., mask], y[..., mask], atol=1e-5)
    assert torch.allclose(k[..., ~mask], fft2c(x)[..., ~mask], atol=1e-5)


def test_full_mask_projection_is_zero_filled():
    truth = rand_complex(1, 12, 12)
    mask = torch.ones(12, dtype=torch.bool)
    y = measure(truth, mask)
    assert torch.allclose(project(rand_complex(1, 12, 12), y, mask), zero_filled(y), atol=1e-5)
