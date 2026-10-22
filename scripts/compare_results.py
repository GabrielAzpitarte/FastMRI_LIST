"""Final table: zero-filling, the pretrained U-Net and the EDM on the same test volumes, masks and metrics.

python scripts/compare_results.py --edm runs/main/eval_final --unet runs/baselines/unet --out docs/results_table.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONTRASTS = {"all": None, "PD": "CORPD_FBK", "PDFS": "CORPDFS_FBK"}
METHODS = (("zero_filled", "Zero-filled"), ("unet", "U-Net (pretrained)"), ("edm", "EDM (ours)"))


def load_rows(edm_dir: Path, unet_dir: Path) -> dict:
    """Per acceleration, one row per EDM test volume with all three methods' metrics."""
    edm = json.loads((edm_dir / "metrics.json").read_text())["results"]
    unet = {r["name"]: r for r in json.loads((unet_dir / "metrics.json").read_text())["volumes"]}
    out = {}
    for acc, res in edm.items():
        rows = []
        for r in res["volumes"]:
            u = unet[r["name"]][acc]["cropped"]
            if abs(u["zero_filled"]["psnr"] - r["zero_filled"]["psnr"]) > 1e-4:
                raise ValueError(f"zero-filled mismatch for {r['name']} {acc}: different masks or targets")
            rows.append(dict(name=r["name"], acquisition=unet[r["name"]]["acquisition"],
                             zero_filled=r["zero_filled"], unet=u["unet"], edm=r["edm"]))
        out[acc] = rows
    return out


def mean_sem(v) -> str:
    """Mean ± standard error."""
    v = np.asarray(v)
    return f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def table(rows: dict) -> str:
    """Markdown tables of mean metrics and the paired EDM - U-Net difference, per contrast."""
    lines = []
    for acc, all_rows in rows.items():
        lines += [f"### {acc}", "", "| contrast | method | PSNR (dB) | SSIM | NMSE |", "|---|---|---|---|---|"]
        diffs = ["", "| contrast | EDM − U-Net PSNR | EDM − U-Net SSIM | EDM better (PSNR) |", "|---|---|---|---|"]
        for label, acq in CONTRASTS.items():
            sel = [r for r in all_rows if acq is None or r["acquisition"] == acq]
            for key, name in METHODS:
                cells = [mean_sem([r[key][m] for r in sel]) for m in ("psnr", "ssim", "nmse")]
                lines.append(f"| {label} ({len(sel)}) | {name} | " + " | ".join(cells) + " |")
            dp = [r["edm"]["psnr"] - r["unet"]["psnr"] for r in sel]
            ds = [r["edm"]["ssim"] - r["unet"]["ssim"] for r in sel]
            diffs.append(f"| {label} | {mean_sem(dp)} | {mean_sem(ds)} | {sum(d > 0 for d in dp)}/{len(dp)} |")
        lines += diffs + [""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edm", default="runs/main/eval_final")
    p.add_argument("--unet", default="runs/baselines/unet")
    p.add_argument("--out")
    a = p.parse_args()
    text = table(load_rows(Path(a.edm), Path(a.unet)))
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
