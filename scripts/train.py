"""Train thesis_exp.

torchrun --nproc_per_node=4 scripts/train.py --config configs/main.yaml --run-dir runs/main [key=value ...]
"""
import argparse

from thesis_exp.config import load_config
from thesis_exp.train.loop import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    train(load_config(a.config, a.overrides), a.run_dir)


if __name__ == "__main__":
    main()
