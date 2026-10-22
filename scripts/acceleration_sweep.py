"""Zero-filled, pretrained U-Net and EDM across accelerations on a few fixed test volumes.

GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/acceleration_sweep.py --out results/acceleration_sweep.json
Volumes: the first 3 PD and first 3 PDFS volumes after the 10 tuning volumes; central 4 slices; EDM mean of 4.
Center fraction 0.32/R (fastMRI's 0.08 at 4x and 0.04 at 8x), never below the 4% the intensity scale uses.
"""
import argparse
import json
from pathlib import Path

import torch

from thesis_exp.baselines.unet import load_pretrained, reconstruct as unet_reconstruct
from thesis_exp.data.h5io import list_volumes, read_attrs, read_target
from thesis_exp.eval.metrics import volume_metrics
from thesis_exp.eval.pipeline import central_slices, reconstruct, undersample
from thesis_exp.eval.snapshot import load_ema
from thesis_exp.mri.operator import zero_filled
from thesis_exp.train import dist
from thesis_exp.train.build import configure_backends

ACCELERATIONS = (2, 4, 6, 8, 10, 12, 16)
UNET_WEIGHTS = Path("data/pretrained/fastmri_unet/knee_sc_leaderboard_state_dict.pt")


def center_fraction(acc: int) -> float:
    """fastMRI-style center fraction for any acceleration."""
    return max(0.04, 0.32 / acc)


def pick_volumes(raw: str, per_contrast: int = 3, skip: int = 10) -> list[Path]:
    """First volumes of each contrast in the test split."""
    out = {"CORPD_FBK": [], "CORPDFS_FBK": []}
    for path in list_volumes(raw)[skip:]:
        group = out[read_attrs(path)["acquisition"]]
        if len(group) < per_contrast:
            group.append(path)
    return out["CORPD_FBK"] + out["CORPDFS_FBK"]


def run_job(path: Path, acc: int, edm, unet, slices: int, device) -> dict:
    """Metrics of the three methods for one volume at one acceleration."""
    net, cfg, gain = edm
    target = read_target(path)
    sl = central_slices(len(target), slices)
    maxval, t = float(read_attrs(path)["max"]), target[sl]
    kspace, mask, scale = undersample(path, cfg.data.size, acc, center_fraction(acc))
    zf = zero_filled(kspace[sl]).abs()
    return dict(name=path.name, acquisition=read_attrs(path)["acquisition"], acceleration=acc,
                center_fraction=center_fraction(acc),
                zero_filled=volume_metrics(t, zf.numpy(), maxval),
                unet=volume_metrics(t, unet_reconstruct(unet, zf.to(device)).cpu().numpy(), maxval),
                edm=volume_metrics(t, reconstruct(net, kspace[sl], mask, scale, gain, cfg.recon, device, seed=0), maxval))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ema-00004108.pt")
    p.add_argument("--out", default="results/acceleration_sweep.json")
    p.add_argument("--raw", default="data/raw/singlecoil_val")
    p.add_argument("--slices", type=int, default=4)
    p.add_argument("--samples", type=int, default=4)
    a = p.parse_args()
    rank, world, device = dist.init()
    configure_backends()
    net, cfg, gain, _ = load_ema(a.ckpt, ["train.channels_last=false", "recon.eta=1.0", f"recon.num_samples={a.samples}"], device)
    unet = load_pretrained(UNET_WEIGHTS, device)
    jobs = [(path, acc) for path in pick_volumes(a.raw) for acc in ACCELERATIONS]
    rows = [run_job(path, acc, (net, cfg, gain), unet, a.slices, device) for path, acc in jobs[rank::world]]
    rows = sum(dist.gather(rows), [])
    if dist.is_main():
        rows.sort(key=lambda r: (r["acceleration"], r["name"]))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(ckpt=a.ckpt, slices=a.slices, edm_samples=a.samples, eta=1.0, rows=rows), indent=1))
        print(f"{len(rows)} jobs written to {a.out}")
    dist.shutdown()


if __name__ == "__main__":
    main()
