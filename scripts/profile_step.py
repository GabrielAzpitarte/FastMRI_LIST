"""FLOPs per training image and the most expensive CUDA kernels of one variant.

python scripts/profile_step.py --config configs/main.yaml --micro-batch 16 [key=value ...]
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from thesis_exp.config import load_config
from thesis_exp.edm.loss import edm_loss
from thesis_exp.train.build import build_net, build_optimizer, compile_model, configure_backends


def step(net, opt, images, cfg) -> None:
    edm_loss(net, images, cfg.edm.p_mean, cfg.edm.p_std, cfg.edm.sigma_data).mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def flops_per_image(net, opt, images, cfg) -> float:
    """Forward + backward FLOPs per image."""
    with FlopCounterMode(display=False) as counter:
        step(net, opt, images, cfg)
    return counter.get_total_flops() / images.shape[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--rows", type=int, default=25)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    configure_backends()
    cfg = load_config(a.config, a.overrides)
    net = build_net(cfg).cuda()
    opt = build_optimizer(net, cfg)
    images = torch.randn(a.micro_batch, 2, cfg.data.size, cfg.data.size, device="cuda") * 0.5
    print(f"TFLOP per image (fwd+bwd): {flops_per_image(net, opt, images, cfg) / 1e12:.3f}")
    compile_model(net, cfg)
    for _ in range(3):
        step(net, opt, images, cfg)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step(net, opt, images, cfg)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.rows, max_name_column_width=70))


if __name__ == "__main__":
    main()
