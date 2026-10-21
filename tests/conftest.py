import pytest

from fastmri_edm.data.preprocess import build_cache
from fastmri_edm.data.synthetic import write_split

SMALL = dict(slices=4, rows=96, cols=80, size=64)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Small synthetic train/val splits with their caches."""
    root = tmp_path_factory.mktemp("synthetic")
    for split, seed in (("train", 0), ("val", 1)):
        write_split(root / "raw" / split, volumes=3, seed=seed, **SMALL)
        build_cache(root / "raw" / split, root / "cache" / split, size=SMALL["size"], workers=2)
    return root
