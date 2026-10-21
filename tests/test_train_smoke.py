import json

import pytest
import torch

from fastmri_edm.config import load_config
from fastmri_edm.eval.pipeline import evaluate_volume
from fastmri_edm.data.h5io import list_volumes
from fastmri_edm.train import checkpoint
from fastmri_edm.train.build import build_net
from fastmri_edm.train.loop import train

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def smoke_cfg(root, duration_kimg):
    return load_config("configs/smoke.yaml", [
        f"data.train_cache={root / 'cache' / 'train'}", f"data.val_cache={root / 'cache' / 'val'}",
        f"data.val_raw={root / 'raw' / 'val'}", "data.size=64", "model.channels=16",
        "model.channel_mult=[1,2,2]", "model.attn_resolutions=[16]", "train.batch=16",
        "train.micro_batch=8", "train.lr=2e-3", "train.log_every=25", "train.workers=0",
        f"train.duration_kimg={duration_kimg}", "train.snapshot_kimg=1", "train.eval_kimg=1",
        "probe.heldout_slices=4", "recon.slices=2", "recon.num_steps=4",
    ])


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_resume_and_evaluate(synthetic, tmp_path):
    run = tmp_path / "run"
    train(smoke_cfg(synthetic, 2), run)
    rows = [r for r in read_rows(run / "stats.jsonl") if "loss" in r]
    assert len(rows) == 5 and all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert checkpoint.latest(run).name == "state-00000002.pt"
    assert (run / "samples" / "samples-00000002.png").exists()

    train(smoke_cfg(synthetic, 3), run)
    starts = [r for r in read_rows(run / "stats.jsonl") if r.get("event") == "start"]
    assert starts[-1]["kimg"] == 2.0
    assert checkpoint.latest(run).name == "state-00000003.pt"

    state = checkpoint.load(checkpoint.latest(run, "ema"))
    cfg = smoke_cfg(synthetic, 3)
    net = build_net(cfg).cuda().eval()
    net.load_state_dict(state["ema"])
    path = list_volumes(cfg.data.val_raw)[0]
    metrics, images = evaluate_volume(net, path, cfg, 4, 0.08, state["gain"], torch.device("cuda"))
    assert images["edm"].shape == images["target"].shape == (2, 64, 64)
    assert all(map(lambda v: v == v, metrics["edm"].values()))
