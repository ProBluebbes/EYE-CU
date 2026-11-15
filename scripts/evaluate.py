"""Evaluate a checkpoint on its fold's held-out test videos (or --split val). Example:
    uv run python scripts/evaluate.py runs/<run>/best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # robust to macOS hiding the editable-install .pth

import argparse
import csv
import json
from pathlib import Path

import torch

from palsynet.data import build_loaders
from palsynet.engine import evaluate, get_device
from palsynet.models import build_model
from palsynet.splits import CLASSES, load_splits, split_for_fold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    meta, videos = load_splits(Path(cfg["splits"]))
    train_v, val_v, test_v = split_for_fold(videos, cfg["fold"], meta["n_folds"])
    _, val_dl, test_dl = build_loaders(cfg, train_v, val_v, test_v, Path(cfg["raw_root"]))
    loader, vids = (test_dl, test_v) if args.split == "test" else (val_dl, val_v)

    device = get_device(args.device)
    m = cfg["model"]
    model = build_model({**m, "pretrained": False}, cfg["data"])  # weights come from the checkpoint
    model.load_state_dict(ckpt["model"])
    model.to(device)

    res = evaluate(model, loader, device)
    out_dir = args.checkpoint.parent
    with open(out_dir / f"{args.split}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "prob_affected", "pred", "n_segments"])
        for r in res["per_video"]:
            w.writerow([vids[r["video_idx"]].video_id, CLASSES[r["label"]], f"{r['prob_affected']:.4f}",
                        CLASSES[int(r["prob_affected"] >= 0.5)], r["n_segments"]])
    summary = {k: res[k] for k in ("loss", "segment", "video")}
    json.dump(summary, open(out_dir / f"{args.split}_metrics.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
