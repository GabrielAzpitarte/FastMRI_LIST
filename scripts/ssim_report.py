"""Markdown tables from the SSIM diagnostics (scripts/ssim_diagnostics.py).

.venv/bin/python scripts/ssim_report.py --dir runs/diag_ssim --out docs/SSIM_DIAGNOSIS_TABLES.md
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from thesis_exp.eval.pipeline import central_slices
from thesis_exp.eval.ssim_variants import volume_stats

PIXELS = 320 * 320
CONTRASTS = {"all": None, "PD": "CORPD_FBK", "PDFS": "CORPDFS_FBK"}
CONVENTIONS = ["fastmri", "gauss", "slice_max", "minmax", "uint8", "fg", "bg"]
SAMPLE_KEYS = [f"edm_n{k}" for k in (1, 2, 4, 8, 16, 32)]
SELECTIONS = {"all slices": lambda n: slice(0, n), "drop first+last": lambda n: slice(1, n - 1),
              "central 8": lambda n: central_slices(n, 8)}


def load_classic(folder: Path) -> list[dict]:
    """Per-volume statistics from every rank."""
    return [json.loads(l) for f in sorted(folder.glob("classic_rank*.jsonl")) for l in f.read_text().splitlines()]


def volume_metric(st: dict, key: str, sl: slice, vmax: float) -> float:
    """Volume-level metric over selected slices (SSIM: mean over slices; PSNR/NMSE from summed errors)."""
    if key == "psnr":
        return float(10 * np.log10(vmax ** 2 / (np.sum(st["sse"][sl]) / (len(st["sse"][sl]) * PIXELS))))
    if key == "nmse":
        return float(np.sum(st["sse"][sl]) / np.sum(st["gg"][sl]))
    if key == "scale":
        return float(np.sum(st["gp"][sl]) / np.sum(st["pp"][sl]))
    return float(np.nanmean(st[key][sl]))


def mean_over(rows: list[dict], method: str, key: str, select, contrast: str | None, vmax_key: str = "vmax") -> str:
    """Mean of a volume-level metric over the volumes of one contrast."""
    vals = [volume_metric(r["methods"][method], key, select(r["slices"]), r[vmax_key])
            for r in rows if method in r["methods"] and (contrast is None or r["acquisition"] == contrast)]
    return f"{np.mean(vals):.3f}" if vals else "-"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table."""
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def slice_table(rows: list[dict]) -> str:
    """fastMRI SSIM and PSNR under three slice selections."""
    out = []
    for method in ("zf", "unet"):
        for acc in (2, 4, 8):
            for cname, c in CONTRASTS.items():
                cells = [f"{mean_over(rows, f'{method}_{acc}x', 'fastmri', sel, c)} / {mean_over(rows, f'{method}_{acc}x', 'psnr', sel, c)}"
                         for sel in SELECTIONS.values()]
                out.append([method, f"{acc}x", cname] + cells)
    return table(["method", "acc", "contrast"] + [f"{k}: SSIM / PSNR" for k in SELECTIONS], out)


def convention_table(rows: list[dict], methods: list[str], select) -> str:
    """Mean SSIM under every convention."""
    out = [[m, cname] + [mean_over(rows, m, k, select, c) for k in CONVENTIONS]
           for m in methods for cname, c in CONTRASTS.items()]
    return table(["method", "contrast"] + CONVENTIONS, out)


def ceiling_table(rows: list[dict], select) -> str:
    """Real methods next to the ground-truth references and the simulated oracle, per contrast."""
    names = ["full", "gt_blur07", "gt_blur10", "gt_nlm", "sim_clean"] + \
            [f"{p}_{acc}x" for acc in (2, 4, 8) for p in ("zf", "sim_zf", "unet", "sim_oracle")]
    out = [[m] + [f"{mean_over(rows, m, 'fastmri', select, c)} / {mean_over(rows, m, 'psnr', select, c, 'vmax' if not m.startswith('sim') else 'sim_max')}"
                  for c in CONTRASTS.values()] for m in names]
    return table(["method"] + [f"{k}: SSIM / PSNR" for k in CONTRASTS], out)


def scale_table(rows: list[dict], methods: list[str], select) -> str:
    """Least-squares intensity scale (1 = correct) and foreground mean ratio."""
    out = [[m] + [f"{mean_over(rows, m, 'scale', select, c)} / {mean_over(rows, m, 'fg_ratio', select, c)}"
                  for c in CONTRASTS.values()] for m in methods]
    return table(["method"] + [f"{k}: LS scale / fg mean ratio" for k in CONTRASTS], out)


def snr_table(rows: list[dict]) -> str:
    """Noise level and signal-to-noise ratio per contrast."""
    out = []
    for cname, c in CONTRASTS.items():
        snr = [r["snr"] for r in rows if c is None or r["acquisition"] == c]
        out.append([cname, str(len(snr)), f"{np.median(snr):.1f}", f"{np.percentile(snr, 10):.1f}", f"{np.percentile(snr, 90):.1f}"])
    return table(["contrast", "volumes", "median SNR", "p10", "p90"], out)


def edm_row(path: Path) -> dict:
    """Statistics of one saved EDM reconstruction file."""
    z = np.load(path)
    vmax = float(z["vmax"])
    methods = {m: volume_stats(z["target"], z[m], vmax) for m in ("zf", "unet", "edm", "edm_1", *SAMPLE_KEYS) if m in z}
    name, acc = path.stem.rsplit("_", 1)
    return dict(name=name, acc=acc, acquisition=str(z["acquisition"]), slices=int(len(z["target"])),
                first=int(z["slices"][0]), vmax=vmax, methods=methods)


def edm_rows(folder: Path, workers: int) -> list[dict]:
    """Statistics of every saved EDM file in a folder."""
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(edm_row, sorted(folder.glob("*.npz"))))


def edm_table(rows: list[dict], keys: list[str]) -> str:
    """Zero-filled, U-Net, EDM mean of 8 and EDM single sample per acceleration and contrast."""
    out = []
    for acc in ("2x", "4x", "8x"):
        sub = [r for r in rows if r["acc"] == acc]
        for cname, c in CONTRASTS.items():
            n = sum(c is None or r["acquisition"] == c for r in sub)
            for m in ("zf", "unet", "edm", "edm_1"):
                out.append([acc, f"{cname} ({n})", m] + [mean_over(sub, m, k, lambda s: slice(0, s), c) for k in keys])
    return table(["acc", "contrast", "method"] + keys, out)


def edm_slice_table(rows: list[dict]) -> str:
    """EDM over all slices of a few volumes under three slice selections."""
    out = []
    for acc in ("2x", "4x", "8x"):
        sub = [r for r in rows if r["acc"] == acc]
        for m in ("zf", "unet", "edm"):
            out.append([acc, m] + [f"{mean_over(sub, m, 'fastmri', sel, None)} / {mean_over(sub, m, 'psnr', sel, None)}"
                                   for sel in SELECTIONS.values()])
    return table(["acc", "method"] + [f"{k}: SSIM / PSNR" for k in SELECTIONS], out)


def samples_table(rows: list[dict], classic: list[dict]) -> str:
    """SSIM / PSNR of the mean of the first k posterior samples, next to the U-Net and the oracle."""
    oracle = {r["name"]: r for r in classic}
    out = []
    for acc in ("2x", "4x"):
        sub = [r for r in rows if r["acc"] == acc]
        for cname, c in CONTRASTS.items():
            part = [r for r in sub if c is None or r["acquisition"] == c]
            cells = [f"{mean_over(part, m, 'fastmri', lambda s: slice(0, s), None)} / {mean_over(part, m, 'psnr', lambda s: slice(0, s), None)}"
                     for m in SAMPLE_KEYS + ["unet"]]
            orc = np.mean([np.mean(oracle[r["name"] + ".h5"]["methods"][f"sim_oracle_{acc}"]["fastmri"][central_slices(oracle[r["name"] + ".h5"]["slices"], 8)])
                           for r in part])
            out.append([acc, f"{cname} ({len(part)})"] + cells + [f"{orc:.3f}"])
    return table(["acc", "contrast"] + [k.replace("edm_n", "mean of ") for k in SAMPLE_KEYS] + ["U-Net", "oracle SSIM"], out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs/diag_ssim")
    p.add_argument("--out", default="docs/SSIM_DIAGNOSIS_TABLES.md")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    folder = Path(a.dir)
    rows = load_classic(folder)
    c8 = SELECTIONS["central 8"]
    recon = [f"{m}_{acc}x" for acc in (2, 4, 8) for m in ("zf", "unet")]
    parts = [f"# SSIM diagnosis tables\n\n{len(rows)} test volumes (validation volumes 10-198) unless stated.",
             "## Noise level\n\n" + snr_table(rows),
             "## Slice selection (zero-filled and U-Net, fastMRI SSIM / PSNR)\n\n" + slice_table(rows),
             "## SSIM conventions (central 8 slices)\n\n" + convention_table(rows, ["full"] + recon, c8),
             "## Ceilings (central 8 slices, fastMRI SSIM / PSNR)\n\n" + ceiling_table(rows, c8),
             "## Intensity scale (central 8 slices)\n\n" + scale_table(rows, recon, c8)]
    if (folder / "edm_s8_n32").exists():
        parts.append("## EDM: number of averaged posterior samples (SSIM / PSNR, central 8 slices)\n\n" +
                     samples_table(edm_rows(folder / "edm_s8_n32", a.workers), rows))
    for sub, title in (("edm_s8", "EDM subset, central 8 slices"), ("edm_sall", "EDM subset, all slices")):
        if (folder / sub).exists():
            er = edm_rows(folder / sub, a.workers)
            parts.append(f"## {title} ({len({r['name'] for r in er})} volumes)\n\n" +
                         edm_table(er, ["fastmri", "psnr", "gauss", "minmax", "fg", "scale"]))
            if sub == "edm_sall":
                parts.append(f"## {title}: slice selection\n\n" + edm_slice_table(er))
    Path(a.out).write_text("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
