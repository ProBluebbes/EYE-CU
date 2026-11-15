"""Train a video classifier on one fold. Examples:
    uv run python scripts/train.py --config configs/r3d_18.yaml --fold 0
    uv run python scripts/train.py --config configs/vivit_b16.yaml --fold 0 --set train.epochs=30
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # robust to macOS hiding the editable-install .pth

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from palsynet.config import apply_overrides, load_config
from palsynet.data import build_loaders
from palsynet.engine import evaluate, get_device, train_one_epoch
from palsynet.models import build_model
from palsynet.splits import load_splits, split_for_fold


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r3d_18.yaml")
    p.add_argument("--fold", type=int)
    p.add_argument("--set", nargs="*", action="extend", default=[], help="config overrides, e.g. train.epochs=2 (repeatable)")
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    args = p.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.fold is not None:
        cfg["fold"] = args.fold
    set_seed(cfg["seed"])

    meta, videos = load_splits(Path(cfg["splits"]))
    train_v, val_v, test_v = split_for_fold(videos, cfg["fold"], meta["n_folds"])
    print(f"fold {cfg['fold']}: train {len(train_v)} / val {len(val_v)} / test {len(test_v)} videos")
    train_dl, val_dl, _ = build_loaders(cfg, train_v, val_v, test_v, Path(cfg["raw_root"]))
    if args.max_steps:
        from itertools import islice
        class _Cut:  # wrap loader to cap steps for smoke tests
            def __init__(self, dl, n): self.dl, self.n, self.dataset = dl, n, dl.dataset
            def __iter__(self): return islice(iter(self.dl), self.n)
            def __len__(self): return min(self.n, len(self.dl))
        train_dl, val_dl = _Cut(train_dl, args.max_steps), _Cut(val_dl, args.max_steps)

    device = get_device(cfg["train"]["device"])
    m = cfg["model"]
    model = build_model(m, cfg["data"]).to(device)
    print(f"model {m['name']}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    t = cfg["train"]
    from palsynet.optim import param_groups
    opt = torch.optim.AdamW(param_groups(model, t["lr"], t["weight_decay"], t.get("lr_mult")), lr=t["lr"], weight_decay=t["weight_decay"])
    accum = t.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(train_dl) / accum)  # optimizer steps
    total = t["epochs"] * steps_per_epoch
    warm = t["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    )
    scaler = torch.amp.GradScaler("cuda") if (t["amp"] and device.type == "cuda") else None

    out = Path(cfg["output_dir"]) / f"{m['name']}_fold{cfg['fold']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    print(f"device={device} -> {out}")

    # Best checkpoint: highest val video metric; ties (common: with ~10 val videos AUC saturates at
    # 1.0 within a few epochs) go to the lower val loss, i.e. the more confident, better-calibrated epoch.
    best, best_epoch, history = (-float("inf"), -float("inf")), -1, []
    for epoch in range(t["epochs"]):
        train_dl.dataset.set_epoch(epoch)
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, opt, sched, device, scaler, grad_accum=accum)
        val = evaluate(model, val_dl, device)
        score = val["video"][t["select_metric"]]
        score = (-1.0 if np.isnan(score) else score, -val["loss"])
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val["loss"], "val_segment": val["segment"], "val_video": val["video"], "time_s": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch}: train {tr_loss:.4f} | val loss {val['loss']:.4f} "
              f"segment acc {val["segment"]["accuracy"]:.3f} | video acc {val['video']['accuracy']:.3f} auc {val['video']['auc']:.3f}")
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": rec}
        torch.save(ckpt, out / "last.pt")
        if score > best:
            best, best_epoch = score, epoch
            torch.save(ckpt, out / "best.pt")
        json.dump(history, open(out / "history.json", "w"), indent=2)
    print(f"best val video {t['select_metric']}: {best[0]:.3f} (val loss {-best[1]:.4f}, epoch {best_epoch})  checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
