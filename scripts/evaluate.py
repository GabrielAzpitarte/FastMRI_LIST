"""Evaluate an EMA snapshot on retrospectively undersampled validation volumes.

torchrun --nproc_per_node=4 scripts/evaluate.py --ckpt runs/main/ema-00010000.pt --out runs/main/eval [key=value ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.data.h5io import list_volumes
from thesis_exp.eval.images import save_grid
from thesis_exp.eval.pipeline import evaluate_volume, summarize
from thesis_exp.eval.snapshot import load_ema
from thesis_exp.train import dist
from thesis_exp.train.build import configure_backends


def save_example(images: dict, path) -> None:
    """Central slice: target, zero-filled, EDM, and 5x absolute errors."""
    mid = len(images["target"]) // 2
    t, zf, edm = (images[k][mid] for k in ("target", "zero_filled", "edm"))
    hi = float(np.percentile(t, 99.5))
    save_grid([t, zf, edm, 5 * np.abs(zf - t), 5 * np.abs(edm - t)], path, cols=5, hi=hi)


def table(results: dict) -> str:
    """Markdown summary table."""
    lines = ["| acc | method | NMSE | PSNR | SSIM |", "|---|---|---|---|---|"]
    for acc, res in results.items():
        s = res["summary"]
        for m in ("zero_filled", "edm"):
            cells = [f"{s[f'{m}/{k}']['mean']:.4f} ± {s[f'{m}/{k}']['sem']:.4f}" for k in ("nmse", "psnr", "ssim")]
            lines.append(f"| {acc} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip", type=int, default=0, help="skip the first N volumes (tuning set)")
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N volumes")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    rank, world, device = dist.init()
    configure_backends()
    net, cfg, gain, nimg = load_ema(a.ckpt, a.overrides, device)
    paths = list_volumes(cfg.data.val_raw)[a.skip:]
    paths = paths[:a.limit] if a.limit else paths
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for acc, cf in zip(cfg.recon.accelerations, cfg.recon.center_fractions):
        rows = []
        for i, path in enumerate(paths[rank::world]):
            metrics, images = evaluate_volume(net, path, cfg, acc, cf, gain, device, seed=i)
            rows.append(metrics)
            if rank == 0 and i == 0:
                save_example(images, out / f"example_{acc}x_{path.stem}.png")
        rows = sum(dist.gather(rows), [])
        results[f"{acc}x"] = dict(summary=summarize(rows), volumes=rows)
    if dist.is_main():
        info = dict(ckpt=str(a.ckpt), kimg=nimg / 1000, volumes=len(paths), recon=vars(cfg.recon))
        (out / "metrics.json").write_text(json.dumps(dict(info=info, results=results), indent=1))
        (out / "metrics.md").write_text(table(results) + "\n")
        print(json.dumps(info))
        print(table(results))
    dist.shutdown()


if __name__ == "__main__":
    main()
