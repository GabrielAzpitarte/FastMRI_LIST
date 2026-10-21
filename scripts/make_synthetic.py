"""Write synthetic volumes in the fastMRI single-coil layout.

python scripts/make_synthetic.py data/synthetic/raw/singlecoil_train --volumes 16
"""
import argparse

from thesis_exp.data.synthetic import write_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--volumes", type=int, default=8)
    p.add_argument("--slices", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    write_split(a.out, a.volumes, a.seed, slices=a.slices)


if __name__ == "__main__":
    main()
