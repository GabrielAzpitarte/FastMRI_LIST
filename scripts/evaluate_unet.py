"""Evaluate the official pretrained fastMRI single-coil U-Net on the validation volumes.

GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/evaluate_unet.py --out runs/baselines/unet
"""
import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.baselines.unet import load_pretrained
from thesis_exp.data.h5io import list_volumes
from thesis_exp.eval.images import save_grid
from thesis_exp.eval.unet_pipeline import evaluate_volume
from thesis_exp.train import dist

WEIGHTS = "data/pretrained/fastmri_unet/knee_sc_leaderboard_state_dict.pt"
SUBSETS = dict(all=slice(0, None), tuning_0_7=slice(0, 8), test_10_198=slice(10, None))


def summarize(rows: list[dict], key: str, protocol: str) -> dict:
    """Mean metric per method over volumes."""
    return {f"{m}/{k}": float(np.mean([r[key][protocol][m][k] for r in rows]))
            for m in ("zero_filled", "unet") for k in ("nmse", "psnr", "ssim")}


def table(rows: list[dict], keys) -> str:
    """Markdown table of mean metrics per subset, acceleration and protocol."""
    lines = ["| subset | acc | protocol | ZF PSNR | ZF SSIM | U-Net PSNR | U-Net SSIM | U-Net NMSE |", "|---|---|---|---|---|---|---|---|"]
    for subset, sl in SUBSETS.items():
        for key in keys:
            for protocol in ("cropped", "native", "native_all"):
                s = summarize(rows[sl], key, protocol)
                lines.append(f"| {subset} ({len(rows[sl])}) | {key} | {protocol} | {s['zero_filled/psnr']:.2f} | "
                             f"{s['zero_filled/ssim']:.4f} | {s['unet/psnr']:.2f} | {s['unet/ssim']:.4f} | {s['unet/nmse']:.4f} |")
    return "\n".join(lines)


def save_examples(images: dict, out: Path, stem: str) -> None:
    """Central slice per acceleration: target, zero-filled, U-Net and 5x errors; arrays as float16 npz."""
    for acc, im in images.items():
        mid = len(im["target"]) // 2
        t, zf, un = (im[k][mid] for k in ("target", "zero_filled", "unet"))
        save_grid([t, zf, un, 5 * np.abs(zf - t), 5 * np.abs(un - t)], out / f"example_{acc}_{stem}.png", cols=5,
                  hi=float(np.percentile(t, 99.5)))
        np.savez_compressed(out / "recons" / f"{stem}_{acc}.npz", **{k: v.astype(np.float16) for k, v in im.items()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--raw", default="data/raw/singlecoil_val")
    p.add_argument("--slices", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    rank, world, device = dist.init()
    net = load_pretrained(Path(a.weights), device)
    paths = list_volumes(a.raw)
    paths = paths[:a.limit] if a.limit else paths
    out = Path(a.out)
    (out / "recons").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(rank, len(paths), world):
        row, images = evaluate_volume(net, paths[i], 320, (4, 8), (0.08, 0.04), a.slices, device)
        rows.append((i, row))
        if i < 10:
            save_examples(images, out, paths[i].stem)
    rows = [r for _, r in sorted(sum(dist.gather(rows), []), key=lambda x: x[0])]
    if dist.is_main():
        (out / "metrics.json").write_text(json.dumps(dict(weights=a.weights, slices=a.slices, volumes=rows), indent=1))
        (out / "metrics.md").write_text(table(rows, ("4x", "8x")) + "\n")
        print(table(rows, ("4x", "8x")))
    dist.shutdown()


if __name__ == "__main__":
    main()
