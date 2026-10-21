"""Build the normalized slice cache of one split.

python scripts/preprocess.py data/raw/singlecoil_train data/cache/train
"""
import argparse
import json

from fastmri_edm.data.preprocess import build_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("out")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    meta = build_cache(a.raw, a.out, a.size, a.workers)
    keys = ("size", "slices", "rms", "max_esc_error")
    print(json.dumps({k: meta[k] for k in keys} | {"volumes": len(meta["volumes"])}))


if __name__ == "__main__":
    main()
