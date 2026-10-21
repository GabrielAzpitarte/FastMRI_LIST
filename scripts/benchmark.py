"""Single-GPU training-step throughput of one config variant on synthetic tensors.

python scripts/benchmark.py --config configs/main.yaml --micro-batch 16 train.compile=true
"""
import argparse
import copy
import json
import time

import torch

from fastmri_edm.config import load_config
from fastmri_edm.edm.ema import update_ema
from fastmri_edm.edm.loss import edm_loss
from fastmri_edm.train.build import build_net, build_optimizer, compile_model, configure_backends, num_params


def run_steps(net, ema, opt, images, cfg, steps: int) -> float:
    """Seconds for `steps` full optimizer steps."""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        edm_loss(net, images, cfg.edm.p_mean, cfg.edm.p_std, cfg.edm.sigma_data).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema, net, 0.999)
    torch.cuda.synchronize()
    return time.time() - start


def measure(cfg, micro_batch: int, steps: int, warmup: int) -> dict:
    """Images per second and peak memory, or an OOM marker."""
    net = build_net(cfg).cuda()
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = build_optimizer(net, cfg)
    compile_model(net, cfg)
    images = torch.randn(micro_batch, 2, cfg.data.size, cfg.data.size, device="cuda") * cfg.edm.sigma_data
    try:
        run_steps(net, ema, opt, images, cfg, warmup)
        torch.cuda.reset_peak_memory_stats()
        dt = run_steps(net, ema, opt, images, cfg, steps)
    except torch.OutOfMemoryError:
        return dict(micro_batch=micro_batch, oom=True)
    return dict(micro_batch=micro_batch, img_per_sec=round(micro_batch * steps / dt, 2),
                mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), params=num_params(net))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    configure_backends()
    cfg = load_config(a.config, a.overrides)
    row = dict(variant=" ".join(a.overrides) or "default", **measure(cfg, a.micro_batch, a.steps, a.warmup))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
