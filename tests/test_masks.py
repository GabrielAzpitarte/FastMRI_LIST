import numpy as np
import torch

from fastmri_edm.data.synthetic import phantom
from fastmri_edm.mri.fft import fft2c
from fastmri_edm.mri.image import center_crop
from fastmri_edm.mri.masks import center_columns, file_seed, random_mask
from fastmri_edm.mri.operator import measure
from fastmri_edm.mri.scaling import volume_scale


def test_center_columns_match_fastmri():
    assert center_columns(368, 0.08) == slice(170, 199)
    assert center_columns(368, 0.04) == slice(177, 192)
    assert center_columns(320, 0.04) == slice(154, 167)


def test_acs_region_nested_in_both_masks():
    for cols in (320, 368, 372):
        small, large = center_columns(cols, 0.04), center_columns(cols, 0.08)
        assert large.start <= small.start and small.stop <= large.stop


def test_random_mask_rate_and_determinism():
    for acc, cf in ((4, 0.08), (8, 0.04)):
        rates = [random_mask(320, acc, cf, seed).mean() for seed in range(400)]
        assert abs(np.mean(rates) - 1 / acc) < 0.01
    seed = file_seed("file1000001.h5")
    assert np.array_equal(random_mask(320, 4, 0.08, seed), random_mask(320, 4, 0.08, seed))


def test_masks_always_contain_center():
    for acc, cf in ((4, 0.08), (8, 0.04)):
        mask = random_mask(320, acc, cf, 3)
        assert mask[center_columns(320, cf)].all()


def test_scale_identical_from_undersampled_data():
    rng = np.random.default_rng(0)
    img = center_crop(torch.from_numpy(np.stack([phantom(rng, 360, 340) for _ in range(3)])), 320)
    full = volume_scale(fft2c(img))
    for acc, cf in ((4, 0.08), (8, 0.04)):
        mask = torch.from_numpy(random_mask(320, acc, cf, 5))
        assert np.isclose(volume_scale(measure(img, mask)), full, rtol=1e-6)
