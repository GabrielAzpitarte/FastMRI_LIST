"""Rank sampler settings by mean SSIM over accelerations.

python scripts/summarize_tuning.py runs/main/tune
"""
import argparse
import json
from pathlib import Path


def load(run: Path) -> dict:
    """Mean EDM metrics per acceleration of one tuning run."""
    results = json.loads((run / "metrics.json").read_text())["results"]
    return {acc: {k: r["summary"][f"edm/{k}"]["mean"] for k in ("psnr", "ssim")} for acc, r in results.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    a = p.parse_args()
    rows = {run.name: load(run) for run in sorted(Path(a.root).iterdir()) if (run / "metrics.json").exists()}
    score = {name: sum(v["ssim"] for v in accs.values()) / len(accs) for name, accs in rows.items()}
    for name in sorted(rows, key=score.get, reverse=True):
        cells = "  ".join(f"{acc}: PSNR {v['psnr']:.2f} SSIM {v['ssim']:.4f}" for acc, v in rows[name].items())
        print(f"{name:22s} mean SSIM {score[name]:.4f}  {cells}")


if __name__ == "__main__":
    main()
