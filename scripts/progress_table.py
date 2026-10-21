"""Markdown tables of training progress from stats.jsonl and probe.jsonl.

python scripts/progress_table.py runs/main
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def loss_table(stats: list[dict], every_kimg: float) -> str:
    """Training loss and speed, one row per `every_kimg`."""
    rows = [r for r in stats if "loss" in r]
    lines = ["| kimg | loss | img/s |", "|---|---|---|"]
    mark = 0.0
    for r in rows:
        if r["kimg"] >= mark:
            lines.append(f"| {r['kimg']:.0f} | {r['loss']:.4f} | {r['img_per_sec']:.1f} |")
            mark = r["kimg"] + every_kimg
    return "\n".join(lines)


def probe_table(probes: list[dict]) -> str:
    """Held-out denoising MSE and probe reconstruction metrics per check."""
    probes = [p for p in probes if "heldout_mse" in p]
    if not probes:
        return "(no progress checks yet)"
    sigmas = list(probes[0]["heldout_mse"])
    head = ["kimg"] + [f"MSE σ={s}" for s in sigmas]
    for acc in probes[0].get("recon", {}):
        head += [f"{acc} EDM PSNR", f"{acc} EDM SSIM", f"{acc} ZF PSNR", f"{acc} ZF SSIM"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in probes:
        cells = [f"{p['kimg']:.0f}"] + [f"{p['heldout_mse'][s]:.5f}" for s in sigmas]
        for r in p.get("recon", {}).values():
            cells += [f"{r['edm/psnr']:.2f}", f"{r['edm/ssim']:.4f}", f"{r['zero_filled/psnr']:.2f}", f"{r['zero_filled/ssim']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--every-kimg", type=float, default=500)
    a = p.parse_args()
    run = Path(a.run)
    print("## Progress checks (EMA)\n\n" + probe_table(read_jsonl(run / "probe.jsonl")))
    print("\n## Training loss\n\n" + loss_table(read_jsonl(run / "stats.jsonl"), a.every_kimg))


if __name__ == "__main__":
    main()
