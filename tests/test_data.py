import numpy as np
import torch

from fastmri_edm.data.dataset import InfiniteSampler, SliceCache, data_gain
from fastmri_edm.data.h5io import list_volumes, read_target
from fastmri_edm.data.preprocess import load_meta
from fastmri_edm.mri.image import to_complex


def test_cache_layout(synthetic):
    meta = load_meta(synthetic / "cache" / "train")
    images = np.load(synthetic / "cache" / "train" / "images.npy")
    assert images.shape == (12, 2, 64, 64) and images.dtype == np.float32
    assert meta["slices"] == 12 and len(meta["volumes"]) == 3
    assert [v["offset"] for v in meta["volumes"]] == [0, 4, 8]


def test_cache_reproduces_esc_target(synthetic):
    meta = load_meta(synthetic / "cache" / "train")
    images = torch.from_numpy(np.load(synthetic / "cache" / "train" / "images.npy"))
    assert meta["max_esc_error"] < 1e-5
    for path, vol in zip(list_volumes(synthetic / "raw" / "train"), meta["volumes"]):
        rows = images[vol["offset"]:vol["offset"] + vol["slices"]]
        recon = to_complex(rows).abs().numpy() * vol["scale"]
        assert np.allclose(recon, read_target(path), atol=1e-5 * read_target(path).max())


def test_dataset_rms_is_sigma_data(synthetic):
    gain = data_gain(synthetic / "cache" / "train", 0.5)
    data = SliceCache(synthetic / "cache" / "train", gain)
    stacked = torch.stack([data[i] for i in range(len(data))])
    assert stacked.shape == (12, 2, 64, 64)
    assert abs(stacked.square().mean().sqrt().item() - 0.5) < 1e-4


def test_hflip_only_mirrors(synthetic):
    data = SliceCache(synthetic / "cache" / "train", 1.0, hflip=True)
    base = SliceCache(synthetic / "cache" / "train", 1.0)
    torch.manual_seed(0)
    for i in range(len(data)):
        x, ref = data[i], base[i]
        assert torch.equal(x, ref) or torch.equal(x, ref.flip(-1))


def test_infinite_sampler_shards_cover_each_epoch():
    n, world = 10, 3
    epochs = []
    for rank in range(world):
        it = iter(InfiniteSampler(n, rank, world, seed=1))
        epochs.append([next(it) for _ in range(len(range(rank, n, world)))])
    assert sorted(sum(epochs, [])) == list(range(n))


def test_infinite_sampler_skip_resumes_mid_epoch():
    full = iter(InfiniteSampler(12, 1, 2, seed=3, start_epoch=2))
    resumed = iter(InfiniteSampler(12, 1, 2, seed=3, start_epoch=2, skip=4))
    head = [next(full) for _ in range(10)]
    assert [next(resumed) for _ in range(6)] == head[4:10]
