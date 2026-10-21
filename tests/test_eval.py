import numpy as np
import torch

from fastmri_edm.config import apply_override, load_config
from fastmri_edm.eval.metrics import volume_metrics
from fastmri_edm.eval.pipeline import central_slices
from fastmri_edm.eval.recon import dc_sample
from fastmri_edm.mri.fft import fft2c
from fastmri_edm.mri.image import to_complex
from fastmri_edm.mri.masks import random_mask
from fastmri_edm.mri.operator import measure, zero_filled
from test_edm import GaussianDenoiser


def test_dc_sample_is_data_consistent():
    truth = torch.randn(2, 32, 32, dtype=torch.complex64) * 0.3
    mask = torch.from_numpy(random_mask(32, 4, 0.08, 0))
    y = measure(truth, mask)
    x = dc_sample(GaussianDenoiser(0.3), y, mask, num_steps=10, eta=0.5)
    k = fft2c(to_complex(x))
    assert torch.allclose(k[..., mask], y[..., mask], atol=1e-4)


def test_dc_sample_with_full_mask_returns_truth():
    truth = torch.randn(1, 16, 16, dtype=torch.complex64)
    mask = torch.ones(16, dtype=torch.bool)
    x = dc_sample(GaussianDenoiser(1.0), measure(truth, mask), mask, num_steps=5)
    assert torch.allclose(to_complex(x), zero_filled(measure(truth, mask)), atol=1e-4)


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    gt = rng.uniform(size=(3, 32, 32)).astype(np.float32)
    perfect = volume_metrics(gt, gt, gt.max())
    assert perfect["nmse"] == 0 and perfect["ssim"] > 0.999
    noisy = volume_metrics(gt, gt + 0.1 * rng.standard_normal(gt.shape).astype(np.float32), gt.max())
    assert noisy["nmse"] > 0 and noisy["ssim"] < 0.99 and np.isfinite(noisy["psnr"])


def test_central_slices():
    assert central_slices(36, 8) == slice(14, 22)
    assert central_slices(5, 8) == slice(0, 5)
    assert central_slices(36, None) == slice(0, 36)


def test_config_overrides(tmp_path):
    cfg = {"train": {"lr": 0.1}}
    apply_override(cfg, "train.lr=1e-4")
    apply_override(cfg, "model.channel_mult=[1,2]")
    assert cfg["train"]["lr"] == 1e-4 and cfg["model"]["channel_mult"] == [1, 2]
    main = load_config("configs/main.yaml", ["train.compile=false"])
    assert main.train.compile is False and main.data.size == 320
