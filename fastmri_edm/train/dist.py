"""Process-group helpers for single- and multi-GPU runs."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init() -> tuple[int, int, torch.device]:
    """Initialise NCCL under torchrun; return rank, world size and device."""
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 1, torch.device("cuda", 0)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def is_main() -> bool:
    """True on rank 0 or without a process group."""
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    """Synchronise all ranks."""
    if dist.is_initialized():
        dist.barrier()


def mean(value: float, device) -> float:
    """Average a scalar over ranks."""
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t)
    return t.item() / dist.get_world_size()


def gather(obj) -> list:
    """Collect a picklable object from every rank."""
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def shutdown() -> None:
    """Tear down the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
