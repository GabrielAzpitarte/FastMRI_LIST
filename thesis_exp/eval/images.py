"""PNG grids of magnitude images."""
import numpy as np
from PIL import Image


def to_uint8(mag: np.ndarray, hi: float | None = None) -> np.ndarray:
    """Scale a magnitude image to 0-255 (default: its 99.5th percentile is white)."""
    hi = np.percentile(mag, 99.5) if hi is None else hi
    return (np.clip(mag / (hi + 1e-12), 0, 1) * 255).astype(np.uint8)


def save_grid(mags, path, cols: int = 4, hi: float | None = None) -> None:
    """Tile magnitude images (N, H, W) into one grayscale PNG."""
    tiles = [to_uint8(np.asarray(m), hi) for m in mags]
    rows = -(-len(tiles) // cols)
    h, w = tiles[0].shape
    grid = np.zeros((rows * h, cols * w), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    Image.fromarray(grid).save(path)
