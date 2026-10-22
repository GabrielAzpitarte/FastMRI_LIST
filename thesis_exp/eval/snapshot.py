"""Loading EMA snapshots for evaluation."""
from ..config import apply_override, to_namespace
from ..train import checkpoint
from ..train.build import build_net


def load_ema(path, overrides, device):
    """EMA denoiser, its config (with overrides), data gain and training images seen."""
    state = checkpoint.load(path)
    cfg = state["config"]
    for item in overrides:
        apply_override(cfg, item)
    cfg = to_namespace(cfg)
    net = build_net(cfg)
    net.load_state_dict(state["ema"])
    return net.to(device).eval().requires_grad_(False), cfg, state["gain"], state["nimg"]
