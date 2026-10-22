"""Pretrained U-Net on retrospectively undersampled validation volumes, under two undersampling protocols.

native:  fastMRI's protocol, mask the full acquired k-space and crop the zero-filled image (what the U-Net saw in training).
cropped: our protocol, mask the k-space of the cropped 320x320 image (what the EDM is evaluated on).
"""
import numpy as np
import torch

from ..baselines.unet import reconstruct
from ..data.h5io import read_attrs, read_kspace, read_target
from ..mri.image import kspace_to_image
from ..mri.masks import file_seed, random_mask
from ..mri.operator import measure, zero_filled
from .metrics import volume_metrics
from .pipeline import central_slices


def native_zero_filled(kspace: torch.Tensor, name: str, size: int, acceleration: int, center_fraction: float) -> torch.Tensor:
    """Zero-filled magnitude from the masked full k-space."""
    mask = torch.from_numpy(random_mask(kspace.shape[-1], acceleration, center_fraction, file_seed(name)))
    return kspace_to_image(kspace * mask, size).abs()


def cropped_zero_filled(image: torch.Tensor, name: str, size: int, acceleration: int, center_fraction: float) -> torch.Tensor:
    """Zero-filled magnitude from the masked k-space of the cropped image."""
    mask = torch.from_numpy(random_mask(size, acceleration, center_fraction, file_seed(name)))
    return zero_filled(measure(image, mask)).abs()


def pair_metrics(target: np.ndarray, unet: np.ndarray, zf: np.ndarray, maxval: float) -> dict:
    """U-Net and zero-filled metrics against the same target."""
    return dict(unet=volume_metrics(target, unet, maxval), zero_filled=volume_metrics(target, zf, maxval))


def evaluate_volume(net, path, size: int, accelerations, center_fractions, slices: int, device) -> tuple[dict, dict]:
    """Metrics per acceleration and protocol, plus the central-slice images of the cropped protocol."""
    kspace = torch.from_numpy(read_kspace(path))
    image = kspace_to_image(kspace, size)
    target, attrs = read_target(path), read_attrs(path)
    sl = central_slices(len(target), slices)
    maxval, t = float(attrs["max"]), target[sl]
    row, images = dict(name=path.name, acquisition=attrs["acquisition"]), {}
    for acc, cf in zip(accelerations, center_fractions):
        nzf = native_zero_filled(kspace, path.name, size, acc, cf)
        nrec = reconstruct(net, nzf.to(device)).cpu().numpy()
        czf = cropped_zero_filled(image[sl], path.name, size, acc, cf)
        crec = reconstruct(net, czf.to(device)).cpu().numpy()
        nzf, czf = nzf.numpy(), czf.numpy()
        row[f"{acc}x"] = dict(cropped=pair_metrics(t, crec, czf, maxval),
                              native=pair_metrics(t, nrec[sl], nzf[sl], maxval),
                              native_all=pair_metrics(target, nrec, nzf, float(target.max())))
        images[f"{acc}x"] = dict(target=t, zero_filled=czf, unet=crec)
    return row, images
