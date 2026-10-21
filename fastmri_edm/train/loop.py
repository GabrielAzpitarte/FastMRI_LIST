"""Distributed EDM training loop with resumable checkpoints."""
import copy
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import save_config, to_dict
from ..data.dataset import SliceCache, data_gain, infinite_batches
from ..edm.ema import ema_beta, update_ema
from ..edm.loss import edm_loss
from ..eval.progress import Probe
from . import checkpoint, dist
from .build import (build_net, build_optimizer, configure_backends, learning_rate, num_params,
                    sanitize_grads, set_lr)
from .stats import JsonlLog


def accumulation_rounds(batch: int, micro_batch: int, world: int) -> int:
    """Gradient-accumulation rounds per optimizer step."""
    rounds, rest = divmod(batch, micro_batch * world)
    if rest or not rounds:
        raise ValueError("batch must be a positive multiple of micro_batch * world")
    return rounds


def crossed(nimg: int, batch: int, every_kimg: float) -> bool:
    """True when nimg just passed a multiple of every_kimg."""
    every = int(every_kimg * 1000)
    return nimg // every != (nimg - batch) // every


def restore(run_dir, net, ema, opt, device) -> int:
    """Resume from the newest training state; return images seen so far."""
    path = checkpoint.latest(run_dir)
    if path is None:
        return 0
    state = checkpoint.load(path, device)
    net.load_state_dict(state["net"])
    ema.load_state_dict(state["ema"])
    opt.load_state_dict(state["opt"])
    return state["nimg"]


def save_state(run_dir, net, ema, opt, nimg: int, gain: float, cfg) -> None:
    """Full training state and an EMA-only snapshot."""
    kimg = nimg // 1000
    meta = dict(nimg=nimg, gain=gain, config=to_dict(cfg))
    full = dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), **meta)
    checkpoint.save(full, checkpoint.state_path(run_dir, kimg))
    checkpoint.save(dict(ema=ema.state_dict(), **meta), checkpoint.ema_path(run_dir, kimg))
    checkpoint.prune(run_dir)


def train_step(ddp, batches, rounds: int, per_gpu: int, cfg, device) -> torch.Tensor:
    """Forward and backward over all accumulation rounds; returns the summed sample loss."""
    total = torch.zeros((), device=device)
    for r in range(rounds):
        images = next(batches).to(device, non_blocking=True)
        defer = isinstance(ddp, DDP) and r < rounds - 1
        with ddp.no_sync() if defer else nullcontext():
            loss = edm_loss(ddp, images, cfg.edm.p_mean, cfg.edm.p_std, cfg.edm.sigma_data)
            loss.sum().div(per_gpu).backward()
        total += loss.detach().sum()
    return total


class Meter:
    """Loss and throughput between two log lines."""

    def __init__(self, nimg: int, device):
        self.device = device
        self.reset(nimg)

    def reset(self, nimg: int) -> None:
        self.loss = torch.zeros((), device=self.device)
        self.samples = 0
        self.nimg, self.time = nimg, time.time()

    def row(self, nimg: int, pixels: int, lr: float) -> dict:
        loss = dist.mean(self.loss.item() / max(self.samples * pixels, 1), self.device)
        dt = time.time() - self.time
        return dict(kimg=round(nimg / 1000, 2), loss=loss, lr=lr,
                    img_per_sec=round((nimg - self.nimg) / dt, 1),
                    mem_gb=round(torch.cuda.max_memory_allocated(self.device) / 1e9, 1))


def setup(cfg, run_dir, rank: int, world: int, device):
    """Network, EMA, optimizer, resume point and data stream."""
    torch.manual_seed(cfg.train.seed + rank)
    net = build_net(cfg).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = build_optimizer(net, cfg)
    nimg = restore(run_dir, net, ema, opt, device)
    gain = data_gain(cfg.data.train_cache, cfg.edm.sigma_data)
    data = SliceCache(cfg.data.train_cache, gain, cfg.data.hflip)
    batches = infinite_batches(data, cfg.train.micro_batch, rank, world, cfg.train.seed,
                               nimg // len(data), cfg.train.workers)
    if cfg.train.compile:
        net.model.compile()
    return net, ema, opt, nimg, gain, batches


def train(cfg, run_dir) -> None:
    """Train until cfg.train.duration_kimg, resuming from the newest state in run_dir."""
    rank, world, device = dist.init()
    configure_backends()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    net, ema, opt, nimg, gain, batches = setup(cfg, run_dir, rank, world, device)
    ddp = DDP(net, device_ids=[device.index], gradient_as_bucket_view=True) if world > 1 else net
    t = cfg.train
    rounds = accumulation_rounds(t.batch, t.micro_batch, world)
    per_gpu, pixels, duration = t.batch // world, 2 * cfg.data.size ** 2, int(t.duration_kimg * 1000)
    log = JsonlLog(run_dir / "stats.jsonl") if dist.is_main() else None
    probe = Probe(cfg, gain, device, run_dir) if dist.is_main() else None
    if log:
        save_config(cfg, run_dir / "config.yaml")
        log.write(event="start", kimg=nimg / 1000, params=num_params(net), world=world, rounds=rounds, gain=gain)
    meter, step, stop = Meter(nimg, device), 0, False
    while nimg < duration:
        lr = learning_rate(t.lr, nimg, t.lr_rampup_kimg)
        set_lr(opt, lr)
        meter.loss += train_step(ddp, batches, rounds, per_gpu, cfg, device)
        meter.samples += per_gpu
        sanitize_grads(net)
        opt.step()
        opt.zero_grad(set_to_none=True)
        nimg += t.batch
        update_ema(ema, net, ema_beta(t.batch, nimg, t.ema_halflife_kimg, t.ema_rampup_ratio))
        step += 1
        if step % t.log_every == 0:
            row = meter.row(nimg, pixels, lr)
            if log:
                log.write(**row)
            meter.reset(nimg)
            stop = dist.mean(float((run_dir / "STOP").exists()), device) > 0
        if crossed(nimg, t.batch, t.snapshot_kimg) or nimg >= duration or stop:
            if dist.is_main():
                save_state(run_dir, net, ema, opt, nimg, gain, cfg)
            dist.barrier()
        if crossed(nimg, t.batch, t.eval_kimg) or nimg >= duration:
            if probe:
                probe.run(ema, nimg)
            dist.barrier()
        if stop:
            break
    if log:
        log.write(event="end", kimg=nimg / 1000)
    dist.shutdown()
