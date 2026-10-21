import lzma
import shutil
import subprocess
import sys

import numpy as np
import pytest

from thesis_exp.data.xz_parallel import block_jobs, decompress_block, ordered_map, read_varint, write_varint


def test_varint_roundtrip():
    for value in (0, 1, 127, 128, 300, 2 ** 40 + 5):
        assert read_varint(write_varint(value), 0) == (value, len(write_varint(value)))


@pytest.mark.skipif(shutil.which("xz") is None, reason="needs the xz tool")
def test_parallel_matches_serial(tmp_path):
    rng = np.random.default_rng(0)
    payload = rng.standard_normal(400_000).astype(np.float32).tobytes() + b"text" * 50_000
    raw = tmp_path / "data.bin"
    raw.write_bytes(payload)
    subprocess.run(["xz", "-T4", "--block-size=65536", "-k", str(raw)], check=True)
    jobs = block_jobs(str(raw) + ".xz")
    assert len(jobs) > 10
    assert b"".join(ordered_map(decompress_block, jobs, workers=4, window=3)) == payload
    out = subprocess.run([sys.executable, "-m", "thesis_exp.data.xz_parallel", str(raw) + ".xz", "--workers", "3"],
                         check=True, capture_output=True).stdout
    assert out == lzma.decompress((tmp_path / "data.bin.xz").read_bytes())
