"""SSIM under the conventions found across the MRI literature, plus noise-ceiling references."""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_nl_means

GAUSS = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def foreground(gt: np.ndarray) -> np.ndarray:
    """Knee mask of one slice: Otsu (1979) threshold of the smoothed image, holes filled, slightly dilated."""
    smooth = ndimage.gaussian_filter(gt, 3)
    mask = ndimage.binary_fill_holes(smooth > threshold_otsu(smooth))
    return ndimage.binary_dilation(mask, iterations=3)


def masked_mean(smap: np.ndarray, mask: np.ndarray, pad: int = 3) -> float:
    """Mean of an SSIM map inside a mask, ignoring the border skimage also ignores."""
    inner = np.zeros_like(mask)
    inner[pad:-pad, pad:-pad] = True
    sel = mask & inner
    return float(smap[sel].mean()) if sel.any() else float("nan")


def minmax(x: np.ndarray) -> np.ndarray:
    """Image scaled to [0, 1] by its own range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def to_uint8(x: np.ndarray, hi: float) -> np.ndarray:
    """8-bit image with `hi` as white, as when SSIM is computed on saved PNGs."""
    return np.round(np.clip(x / hi, 0, 1) * 255).astype(np.uint8)


def slice_stats(g: np.ndarray, p: np.ndarray, vmax: float) -> dict:
    """Every SSIM convention plus the sums needed for PSNR, NMSE and intensity-scale checks."""
    g, p = g.astype(np.float32), p.astype(np.float32)
    fg = foreground(g)
    _, smap = structural_similarity(g, p, data_range=vmax, full=True)
    hi = np.percentile(g, 99.5)
    g, p = g.astype(np.float64), p.astype(np.float64)
    return dict(
        fastmri=float(smap[3:-3, 3:-3].mean(dtype=np.float64)),
        fg=masked_mean(smap, fg),
        bg=masked_mean(smap, ~fg),
        slice_max=float(structural_similarity(g, p, data_range=g.max())),
        minmax=float(structural_similarity(minmax(g), minmax(p), data_range=1.0)),
        gauss=float(structural_similarity(g, p, data_range=vmax, **GAUSS)),
        gauss_slice=float(structural_similarity(g, p, data_range=g.max(), **GAUSS)),
        uint8=float(structural_similarity(to_uint8(g, hi), to_uint8(p, hi), data_range=255)),
        sse=float(((g - p) ** 2).sum()), gg=float((g * g).sum()), gp=float((g * p).sum()), pp=float((p * p).sum()),
        fg_ratio=float(p[fg].mean() / g[fg].mean()), fg_frac=float(fg.mean()))


def volume_stats(gt: np.ndarray, pred: np.ndarray, vmax: float) -> dict:
    """Per-slice statistics of a volume, as lists keyed by statistic."""
    rows = [slice_stats(g, p, vmax) for g, p in zip(gt, pred)]
    return {k: [r[k] for r in rows] for k in rows[0]}


def mad_std(x: np.ndarray) -> float:
    """Robust standard deviation (median absolute deviation)."""
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def noise_sigma(image: np.ndarray) -> float:
    """Complex noise std of a complex volume (Immerkaer 1996 Laplacian estimator, made robust with the MAD)."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    mid = len(image) // 2
    parts = [x for s in image[max(mid - 4, 0):mid + 4] for x in (s.real, s.imag)]
    return float(np.median([mad_std(ndimage.convolve(x.astype(np.float64), kernel)[1:-1, 1:-1]) / 6 for x in parts]))


def nlm(x: np.ndarray, sigma: float) -> np.ndarray:
    """Non-local-means (Buades et al. 2005) denoised slices (real or complex): a smooth image with the anatomy kept."""
    if np.iscomplexobj(x):
        return nlm(x.real, sigma) + 1j * nlm(x.imag, sigma)
    return np.stack([denoise_nl_means(s, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6,
                                      fast_mode=True) for s in x.astype(np.float64)]).astype(np.float32)


def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c(k: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2D inverse FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def oracle_world(clean: np.ndarray, sigma: float, masks: dict, seed: int) -> dict:
    """A simulated scan with known anatomy: noisy ground truth, zero-filled and the oracle per acceleration.

    The oracle keeps the measured lines (with their noise) and puts the exact noise-free anatomy in the
    unmeasured lines. No method can predict the noise of lines it never measured, so this is a ceiling.
    """
    rng = np.random.default_rng(seed)
    noise = (rng.normal(0, sigma, clean.shape) + 1j * rng.normal(0, sigma, clean.shape)).astype(np.complex64)
    out = dict(target=np.abs(clean + noise).astype(np.float32), clean=np.abs(clean).astype(np.float32))
    for name, mask in masks.items():
        measured = ifft2c(fft2c(noise) * mask)
        out[f"oracle_{name}"] = np.abs(clean + measured).astype(np.float32)
        out[f"zf_{name}"] = np.abs(ifft2c(fft2c(clean + noise) * mask)).astype(np.float32)
    return out
