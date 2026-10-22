"""Charts for the write-up and slides, built only from the JSON results in results/ (no GPU, no data).

python scripts/make_plots.py --out docs/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"edm": "#2a78d6", "unet": "#eb6834", "zero_filled": "#8a8984"}
LABELS = {"edm": "EDM (ours)", "unet": "U-Net (pretrained, supervised)", "zero_filled": "Zero-filled"}
CONTRAST = {"CORPD_FBK": ("PD", "#1baf7a", "o"), "CORPDFS_FBK": ("PDFS", "#6250d6", "s")}
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SIGMA_DATA = 0.5

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 8,
})


def load(path: Path):
    """JSON or JSONL file."""
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(name)


def training_loss(out: Path) -> None:
    rows = [r for r in load(R / "training/stats.jsonl") if "loss" in r]
    kimg, loss = np.array([r["kimg"] for r in rows]), np.array([r["loss"] for r in rows])
    smooth = np.convolve(loss, np.ones(25) / 25, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(kimg / 1000, loss, color=BLUES[0], linewidth=0.8, alpha=0.6)
    ax.plot(kimg[12:-12] / 1000, smooth, color=COLORS["edm"], label="training loss (moving average)")
    ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1.5)
    ax.text(1.05, 0.335, "reconstructions stop\nimproving (~1 Mimg)", color=MUTED, fontsize=10, va="top")
    ax.axvline(kimg[-1] / 1000, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(kimg[-1] / 1000 - 0.05, 0.335, "stopped\n4.1 Mimg", color=MUTED, fontsize=10, va="top", ha="right")
    ax.set(xlabel="training images seen (millions)", ylabel="EDM loss (1.0 = network adds nothing)",
           title="Training loss: fast drop, then a slow tail", ylim=(0.28, 0.36))
    ax.legend(loc="upper right")
    save(fig, out, "training_loss.png")


def loss_per_noise_level(out: Path) -> None:
    probes = load(R / "training/probe.jsonl")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for color, p in zip(BLUES, probes):
        s = np.array([float(k) for k in p["heldout_mse"]])
        mse = np.array(list(p["heldout_mse"].values()))
        weighted = mse * (s ** 2 + SIGMA_DATA ** 2) / (s * SIGMA_DATA) ** 2
        ax.plot(s, weighted, marker="o", color=color, label=f"{p['kimg'] / 1000:.0f} Mimg")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(0.011, 1.02, "network adds nothing", color=MUTED, fontsize=10, va="bottom")
    ax.set(xscale="log", xlabel="noise level σ (data RMS = 0.5)", ylabel="weighted denoising loss (held-out)",
           title="Loss per noise level: the floor at small σ is the scan's own noise", ylim=(0, 1.1))
    ax.legend(title="checkpoint", loc="center right")
    sig = np.logspace(-3, 2, 400)
    density = np.exp(-(np.log(sig) + 1.2) ** 2 / (2 * 1.2 ** 2))
    bx.fill_between(sig, density / density.max(), color=BLUES[0], alpha=0.5, linewidth=0)
    bx.plot(sig, density / density.max(), color=COLORS["edm"])
    bx.axvspan(0.045, 0.24, color=GRID, alpha=0.9, zorder=0)
    bx.text(0.105, 1.05, "scan noise\n(10th–90th pct)", ha="center", color=MUTED, fontsize=9, va="bottom")
    bx.set(xscale="log", xlabel="noise level σ", ylabel="how often σ is drawn in training", yticks=[],
           title="Training noise levels", ylim=(0, 1.35))
    save(fig, out, "loss_per_noise_level.png")


def summary_mean(path: Path, acc: str, key: str) -> float:
    return load(path)["results"][acc]["summary"][key]["mean"]


def unet_subset(acc: str, idx: slice, metric: str = "psnr", protocol: str = "cropped") -> float:
    rows = load(R / "final/unet_eval.json")["volumes"][idx]
    return float(np.mean([r[acc][protocol]["unet"][metric] for r in rows]))


def checkpoint_plateau(out: Path) -> None:
    files = sorted((R / "checkpoint_compare").glob("ema-*.json"))
    kimg = [load(f)["info"]["kimg"] / 1000 for f in files]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, acc in zip(axes, ("4x", "8x")):
        edm = [summary_mean(f, acc, "edm/psnr") for f in files]
        ax.plot(kimg, edm, marker="o", color=COLORS["edm"], label=LABELS["edm"])
        for key, value in (("unet", unet_subset(acc, slice(0, 8))), ("zero_filled", summary_mean(files[0], acc, "zero_filled/psnr"))):
            ax.axhline(value, color=COLORS[key], linestyle="--", linewidth=1.5, label=LABELS[key])
        ax.set(title=f"{acc.replace('x', '×')} acceleration", xlabel="training images seen (millions)", ylabel="PSNR (dB)")
        ax.set_ylim(min(edm + [summary_mean(files[0], acc, "zero_filled/psnr")]) - 0.6, max(edm + [unet_subset(acc, slice(0, 8))]) + 0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("More training did not improve reconstructions (8 tuning volumes, same sampler and seeds)", fontsize=12)
    save(fig, out, "checkpoint_plateau.png")


TUNING_NAMES = {
    "eta0.5": "η=0.5", "eta1.0": "η=1", "eta1.0_steps200": "η=1, 200 steps", "eta1.0_smin0.15": "η=1, stop at σ=0.15",
    "eta0.5_avg4": "η=0.5, mean of 4", "eta0.5_avg8": "η=0.5, mean of 8", "eta1.0_smin0.15_avg4": "η=1, stop early, mean of 4",
    "eta1.0_avg4": "η=1, mean of 4", "eta1.0_avg8": "η=1, mean of 8 (chosen)",
}


def sampler_tuning(out: Path) -> None:
    files = {f.stem: f for f in (R / "tuning").glob("*.json")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    order = sorted(TUNING_NAMES, key=lambda n: summary_mean(files[n], "4x", "edm/psnr") + summary_mean(files[n], "8x", "edm/psnr"))
    y = np.arange(len(order))
    for ax, acc in zip(axes, ("4x", "8x")):
        values = [summary_mean(files[n], acc, "edm/psnr") for n in order]
        colors = [COLORS["edm"] if n == "eta1.0_avg8" else BLUES[0] for n in order]
        ax.scatter(values, y, s=90, color=colors, zorder=3)
        for yi, v in zip(y, values):
            ax.text(v, yi + 0.3, f"{v:.2f}", va="bottom", ha="center", fontsize=9, color=INK)
        for key, value in (("unet", unet_subset(acc, slice(0, 10))), ("zero_filled", summary_mean(files["eta1.0"], acc, "zero_filled/psnr"))):
            ax.axvline(value, color=COLORS[key], linestyle="--", linewidth=1.5, label=LABELS[key])
        lo = summary_mean(files["eta1.0"], acc, "zero_filled/psnr") - 0.8
        ax.set(xlim=(lo, max(values + [unet_subset(acc, slice(0, 10))]) + 0.8), xlabel="PSNR (dB)",
               title=f"{acc.replace('x', '×')} acceleration")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [TUNING_NAMES[n] for n in order])
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.6, -0.02))
    fig.suptitle("Sampler tuning on the 10 tuning volumes: full noise (η=1) and averaging samples matter most", fontsize=12)
    save(fig, out, "sampler_tuning.png")


def final_rows() -> dict:
    edm = load(R / "final/edm_eval.json")["results"]
    unet = {r["name"]: r for r in load(R / "final/unet_eval.json")["volumes"]}
    out = {}
    for acc, res in edm.items():
        out[acc] = [dict(acquisition=unet[r["name"]]["acquisition"], edm=r["edm"], zero_filled=r["zero_filled"],
                         unet=unet[r["name"]][acc]["cropped"]["unet"]) for r in res["volumes"]]
    return out


def mean_sem(v):
    v = np.asarray(v)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def grouped_dots(ax, groups: list[str], data: dict, metric: str, fmt: str) -> None:
    """Dots per method within each group, with standard-error whiskers and value labels (axis need not start at 0)."""
    x, step = np.arange(len(groups)), 0.25
    for i, key in enumerate(("zero_filled", "unet", "edm")):
        stats = [mean_sem(data[g][key]) for g in groups]
        pos = x + (i - 1) * step
        ax.errorbar(pos, [m for m, _ in stats], yerr=[s for _, s in stats], fmt="o", markersize=9, color=COLORS[key],
                    label=LABELS[key], capsize=3, elinewidth=1.5)
        for q, (m, s) in zip(pos, stats):
            ax.annotate(fmt.format(m), (q, m + s), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=INK)
    ax.set_xticks(x, groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="x", visible=False)


def final_results(out: Path) -> None:
    rows = final_rows()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, fmt in ((axes[0], "psnr", "{:.2f}"), (axes[1], "ssim", "{:.3f}")):
        data = {acc.replace("x", "×"): {k: [r[k][metric] for r in rows[acc]] for k in COLORS} for acc in rows}
        grouped_dots(ax, list(data), data, metric, fmt)
        ax.set(ylabel=metric.upper() + (" (dB)" if metric == "psnr" else ""), title=metric.upper())
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Final results on 189 test volumes: a tie at 4×, +2.9 dB at 8×", fontsize=12)
    save(fig, out, "final_results.png")


def final_by_contrast(out: Path) -> None:
    rows = final_rows()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    groups, data = [], {}
    for acc in rows:
        for acq, (name, _, _) in CONTRAST.items():
            g = f"{acc.replace('x', '×')} {name}"
            groups.append(g)
            data[g] = {k: [r[k]["psnr"] for r in rows[acc] if r["acquisition"] == acq] for k in COLORS}
    grouped_dots(ax, groups, data, "psnr", "{:.2f}")
    ax.set(ylabel="PSNR (dB)", title="By contrast: EDM wins on PD scans, the U-Net on noisy PDFS scans at 4×")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, out, "final_by_contrast.png")


def per_volume_scatter(out: Path) -> None:
    rows = final_rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, acc in zip(axes, rows):
        for acq, (name, color, marker) in CONTRAST.items():
            sel = [r for r in rows[acc] if r["acquisition"] == acq]
            ax.scatter([r["unet"]["psnr"] for r in sel], [r["edm"]["psnr"] for r in sel], s=36, color=color, marker=marker,
                       edgecolor=SURFACE, linewidth=0.8, label=name, zorder=3)
        lo = min(min(r["unet"]["psnr"], r["edm"]["psnr"]) for r in rows[acc]) - 0.5
        hi = max(max(r["unet"]["psnr"], r["edm"]["psnr"]) for r in rows[acc]) + 0.5
        ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1.2)
        wins = sum(r["edm"]["psnr"] > r["unet"]["psnr"] for r in rows[acc])
        ax.text(hi - 0.3, lo + 0.3, f"above the line: EDM better\n({wins} of {len(rows[acc])} volumes)", va="bottom", ha="right",
                color=MUTED, fontsize=10)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="U-Net PSNR (dB)", ylabel="EDM PSNR (dB)",
               title=f"{acc.replace('x', '×')}: one dot per test volume")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    save(fig, out, "per_volume_scatter.png")


def paired_difference(out: Path) -> None:
    rows = final_rows()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, acc in zip(axes, rows):
        d = np.array([r["edm"]["psnr"] - r["unet"]["psnr"] for r in rows[acc]])
        ax.hist(d, bins=30, color=COLORS["edm"], edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.5)
        m, s = mean_sem(d)
        ax.set(xlabel="EDM − U-Net PSNR per volume (dB)", ylabel="volumes",
               title=f"{acc.replace('x', '×')}: mean {m:+.2f} ± {s:.2f} dB")
        ax.grid(axis="x", visible=False)
    save(fig, out, "paired_difference.png")


def acceleration_sweep(out: Path) -> None:
    path = R / "acceleration_sweep.json"
    if not path.exists():
        print("acceleration_sweep.json missing, skipped")
        return
    rows = load(path)["rows"]
    accs = sorted({r["acceleration"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("psnr", "ssim")):
        for key in ("zero_filled", "unet", "edm"):
            stats = [mean_sem([r[key][metric] for r in rows if r["acceleration"] == a]) for a in accs]
            ax.errorbar(accs, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o", color=COLORS[key],
                        label=LABELS[key], capsize=3, linewidth=2)
        ax.axvline(4, color=COLORS["unet"], linestyle=":", linewidth=1.2)
        ax.text(4.2, ax.get_ylim()[1], "U-Net trained here", color=MUTED, fontsize=9, va="top")
        ax.set(xscale="log", xticks=accs, xticklabels=[f"{a}×" for a in accs], xlabel="acceleration",
               ylabel=metric.upper() + (" (dB)" if metric == "psnr" else ""), title=metric.upper())
        ax.minorticks_off()
    axes[0].legend(loc="lower left", fontsize=9)
    n = len({r["name"] for r in rows})
    fig.suptitle(f"One diffusion model, any acceleration ({n} test volumes, 3 PD + 3 PDFS)", fontsize=12)
    save(fig, out, "acceleration_sweep.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (training_loss, loss_per_noise_level, checkpoint_plateau, sampler_tuning, final_results,
               final_by_contrast, per_volume_scatter, paired_difference, acceleration_sweep):
        fn(out)


if __name__ == "__main__":
    main()
