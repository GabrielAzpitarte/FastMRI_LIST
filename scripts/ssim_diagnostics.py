"""Why is SSIM low? Metric conventions, slice choice, intensity scale and noise ceilings, at 2x/4x/8x.

classic (all test volumes, all slices, zero-filled + pretrained U-Net + ground-truth references):
  GPUS=4 bash scripts/torchrun.sh scripts/ssim_diagnostics.py classic --out runs/diag_ssim
edm (subset, EDM posterior mean of 8 and a single sample; arrays kept as npz):
  GPUS=4,5 bash scripts/torchrun.sh scripts/ssim_diagnostics.py edm --out runs/diag_ssim --per-contrast 10 --slices 8
"""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from thesis_exp.baselines.unet import load_pretrained, reconstruct as unet_reconstruct
from thesis_exp.data.h5io import list_volumes, read_attrs, read_kspace, read_target
from thesis_exp.eval.pipeline import central_slices, posterior_samples, undersample
from thesis_exp.eval.snapshot import load_ema
from thesis_exp.eval.ssim_variants import nlm, noise_sigma, oracle_world, volume_stats
from thesis_exp.mri.image import kspace_to_image
from thesis_exp.mri.masks import file_seed, random_mask
from thesis_exp.mri.operator import measure, zero_filled
from thesis_exp.train import dist
from thesis_exp.train.build import configure_backends

ACCELERATIONS = {2: 0.16, 4: 0.08, 8: 0.04}
UNET_WEIGHTS = "data/pretrained/fastmri_unet/knee_sc_leaderboard_state_dict.pt"
RAW = "data/raw/singlecoil_val"
SIZE = 320


def test_volumes(skip: int = 10) -> list[Path]:
    """Validation volumes after the tuning set."""
    return list_volumes(RAW)[skip:]


def subset(per_contrast: int) -> list[Path]:
    """First volumes of each contrast in the test split."""
    out = {"CORPD_FBK": [], "CORPDFS_FBK": []}
    for path in test_volumes():
        group = out[read_attrs(path)["acquisition"]]
        if len(group) < per_contrast:
            group.append(path)
    return out["CORPD_FBK"] + out["CORPDFS_FBK"]


def references(target: np.ndarray, image: np.ndarray, masks: dict, seed: int) -> tuple[float, dict, dict]:
    """Ground-truth-derived images (blurred, NLM-denoised) and a simulated scan with known anatomy."""
    sigma = noise_sigma(image)
    refs = dict(gt_blur07=ndimage.gaussian_filter(target, (0, 0.7, 0.7)),
                gt_blur10=ndimage.gaussian_filter(target, (0, 1.0, 1.0)), gt_nlm=nlm(target, sigma))
    return sigma, refs, oracle_world(nlm(image, sigma), sigma, masks, seed)


def cpu_stats(meta: dict, target: np.ndarray, image: np.ndarray, preds: dict, masks: dict) -> dict:
    """All statistics of one volume: the given reconstructions, the references and the simulated scan."""
    vmax = meta["vmax"]
    sigma, refs, world = references(target, image, masks, seed=sum(map(ord, meta["name"])))
    methods = {k: volume_stats(target, p, vmax) for k, p in {**preds, **refs}.items()}
    sim_target, sim_max = world.pop("target"), float(world["clean"].max())
    methods.update({f"sim_{k}": volume_stats(sim_target, p, float(sim_target.max())) for k, p in world.items()})
    return dict(meta, sigma=sigma, snr=float(target[target > 0.2 * target.max()].mean() / sigma),
                sim_max=sim_max, methods=methods)


def classic_volume(path: Path, unet, device, pool):
    """Reconstructions of one volume on the GPU; statistics submitted to the CPU pool."""
    image = kspace_to_image(torch.from_numpy(read_kspace(path)).to(device), SIZE)
    target, attrs = read_target(path), read_attrs(path)
    preds, masks = {"full": image.abs().cpu().numpy()}, {}
    for acc, cf in ACCELERATIONS.items():
        masks[f"{acc}x"] = random_mask(SIZE, acc, cf, file_seed(path.name))
        zf = zero_filled(measure(image, torch.from_numpy(masks[f"{acc}x"]).to(device))).abs()
        preds[f"zf_{acc}x"] = zf.cpu().numpy()
        preds[f"unet_{acc}x"] = unet_reconstruct(unet, zf).cpu().numpy()
    meta = dict(name=path.name, acquisition=attrs["acquisition"], slices=len(target), vmax=float(attrs["max"]),
                gt_max=float(target.max()))
    return pool.submit(cpu_stats, meta, target, image.cpu().numpy(), preds, masks)


def write_done(jobs: list, f) -> list:
    """Write finished volumes; return the jobs still running."""
    for job in [j for j in jobs if j.done()]:
        f.write(json.dumps(job.result()) + "\n")
        f.flush()
    return [j for j in jobs if not j.done()]


def run_classic(a, rank: int, world: int, device) -> None:
    """Zero-filled, U-Net and references on every slice of every test volume."""
    torch.set_num_threads(4)
    unet = load_pretrained(UNET_WEIGHTS, device)
    out = Path(a.out) / f"classic_rank{rank}.jsonl"
    done = {json.loads(l)["name"] for l in out.read_text().splitlines()} if out.exists() else set()
    paths = [p for p in test_volumes()[rank::world][:a.limit or None] if p.name not in done]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool, out.open("a") as f:
        jobs = []
        for path in paths:
            jobs = write_done(jobs + [classic_volume(path, unet, device, pool)], f)
        wait(jobs)
        write_done(jobs, f)


def edm_volume(path: Path, index: int, edm, unet, slices: int, accs, device, out: Path) -> None:
    """EDM (means of the first 1, 2, 4, ... samples), U-Net and zero-filled on the selected slices, saved as npz."""
    net, cfg, gain = edm
    target, attrs = read_target(path), read_attrs(path)
    sl = central_slices(len(target), slices)
    for acc in accs:
        cf = ACCELERATIONS[acc]
        dest = out / f"{path.stem}_{acc}x.npz"
        if dest.exists():
            continue
        kspace, mask, scale = undersample(path, cfg.data.size, acc, cf)
        zf = zero_filled(kspace[sl]).abs()
        samples = posterior_samples(net, kspace[sl], mask, scale, gain, cfg.recon, device, seed=index).cpu().numpy()
        rec = unet_reconstruct(unet, zf.to(device)).cpu().numpy()
        means = {f"edm_n{k}": samples[:k].mean(0) for k in 2 ** np.arange(7) if k <= len(samples)}
        np.savez(dest, target=target[sl], zf=zf.numpy(), unet=rec, edm=samples.mean(0), edm_1=samples[0],
                 edm_std=samples.std(0), vmax=float(attrs["max"]), slices=np.array([sl.start, sl.stop]),
                 acquisition=attrs["acquisition"], **means)


def run_edm(a, rank: int, world: int, device) -> None:
    """EDM reconstructions of a fixed subset at every acceleration."""
    net, cfg, gain, _ = load_ema(a.ckpt, ["train.channels_last=false", "recon.eta=1.0",
                                          f"recon.num_samples={a.samples}"], device)
    unet = load_pretrained(UNET_WEIGHTS, device)
    out = Path(a.out) / (f"edm_s{a.slices or 'all'}" + ("" if a.samples == 8 else f"_n{a.samples}"))
    out.mkdir(parents=True, exist_ok=True)
    paths = subset(a.per_contrast)
    for index in range(rank, len(paths), world):
        edm_volume(paths[index], index, (net, cfg, gain), unet, a.slices, a.accs, device, out)
        print(f"rank {rank}: {paths[index].name} done", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["classic", "edm"])
    p.add_argument("--out", default="runs/diag_ssim")
    p.add_argument("--ckpt", default="runs/main/ema-00004108.pt")
    p.add_argument("--per-contrast", type=int, default=10)
    p.add_argument("--slices", type=int, default=8, help="central slices for the EDM part (0 = all)")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--accs", type=int, nargs="+", default=list(ACCELERATIONS), help="EDM part: accelerations")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="classic: at most N volumes per rank (testing)")
    a = p.parse_args()
    rank, world, device = dist.init()
    configure_backends()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (run_classic if a.part == "classic" else run_edm)(a, rank, world, device)
    dist.shutdown()


if __name__ == "__main__":
    main()
