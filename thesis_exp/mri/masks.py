"""fastMRI-style random 1D Cartesian undersampling masks."""
import numpy as np


def center_columns(num_cols: int, center_fraction: float) -> slice:
    """Fully sampled low-frequency columns, placed as in fastMRI."""
    num_low = round(num_cols * center_fraction)
    pad = (num_cols - num_low + 1) // 2
    return slice(pad, pad + num_low)


def random_mask(num_cols: int, acceleration: int, center_fraction: float, seed) -> np.ndarray:
    """Column mask matching fastMRI RandomMaskFunc for one (fraction, acceleration) pair."""
    rng = np.random.RandomState(seed)
    rng.randint(1)
    center = center_columns(num_cols, center_fraction)
    num_low = center.stop - center.start
    prob = (num_cols / acceleration - num_low) / (num_cols - num_low)
    mask = rng.uniform(size=num_cols) < prob
    mask[center] = True
    return mask


def file_seed(name: str) -> tuple:
    """Per-volume seed used by the fastMRI validation masks."""
    return tuple(map(ord, name))
