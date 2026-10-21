"""fastMRI evaluation metrics on magnitude volumes (slices, H, W)."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Normalized mean squared error."""
    return float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: float) -> float:
    """Peak signal-to-noise ratio over the volume."""
    return float(peak_signal_noise_ratio(gt, pred, data_range=maxval))


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: float) -> float:
    """Mean per-slice structural similarity."""
    return float(np.mean([structural_similarity(g, p, data_range=maxval) for g, p in zip(gt, pred)]))


def volume_metrics(gt: np.ndarray, pred: np.ndarray, maxval: float) -> dict:
    """NMSE, PSNR and SSIM as in the fastMRI evaluation script."""
    return dict(nmse=nmse(gt, pred), psnr=psnr(gt, pred, maxval), ssim=ssim(gt, pred, maxval))
