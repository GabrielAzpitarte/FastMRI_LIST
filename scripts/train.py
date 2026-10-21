"""Train FastMRI-EDM.

torchrun --nproc_per_node=4 scripts/train.py --config configs/main.yaml --run runs/main [key=value ...]
"""
import argparse

from fastmri_edm.config import load_config
from fastmri_edm.train.loop import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    train(load_config(a.config, a.overrides), a.run)


if __name__ == "__main__":
    main()
