"""Webcam false-alarm test: how often does a trained landmark model call ordinary webcam users "affected"?

Uses the VCD clips from scripts/fetch_vcd.py (data/vcd/videos.json; crowd-sourced volunteers on consumer
webcams, treated as unaffected). Every fold's model is scored only on the VCD clips of its own test fold,
so models trained with those clips (data.extra_unaffected) are still tested on clips they never saw.

    uv run python scripts/eval_webcam.py runs/cv_landmark_tcn_3d_*        # one or more cv_* folders
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import numpy as np
import torch

from palsynet.landmarks import LandmarkSegmentDataset
from palsynet.models import build_model
from palsynet.splits import VideoInfo


@torch.no_grad()
def score(ckpt_path: Path, videos: list[VideoInfo], root: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, d = ck["config"], ck["config"]["data"]
    m = build_model({**cfg["model"], "pretrained": False}, d)
    m.load_state_dict(ck["model"]); m.eval()
    ds = LandmarkSegmentDataset(videos, root, False, segments_per_video=d["eval_segments_per_video"],
                                segment_seconds=d["segment_seconds"], sample_fps=d["sample_fps"],
                                mode=d.get("landmark_mode", "2d"), smooth=d.get("landmark_smooth", 5),
                                seq_len=d.get("seq_len", 1), seq_hop=d.get("seq_hop", 1.0))
    seg_p, vid = [], []
    for i in range(len(ds)):
        b = ds[i]
        seg_p.append(float(m(b["video"][None].float()).softmax(-1)[0, 1]))
        vid.append(int(b["video_idx"]))
    seg_p, vid = np.array(seg_p), np.array(vid)
    return seg_p, np.array([seg_p[vid == i].mean() for i in range(len(videos))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cv_dirs", nargs="+", type=Path)
    p.add_argument("--videos", type=Path, default=Path("data/vcd/videos.json"))
    p.add_argument("--root", type=Path, default=Path("data/aligned/landmarks"))
    args = p.parse_args()
    allv = [VideoInfo(**r) for r in json.load(open(args.videos))]
    print(f"{len(allv)} extra healthy clips from {args.videos} (treated as unaffected)\n")
    print(f"{'run':<44} {'clips':>5} {'video FA':>9} {'segment FA':>11} {'mean p':>7} {'median p':>9}")
    for cv in args.cv_dirs:
        segs, vids = [], []
        for ck in sorted(cv.glob("*_fold*/best.pt")):
            k = int(ck.parent.name.split("_fold")[1].split("_")[0])
            vs = [v for v in allv if v.fold == k]
            if vs:
                s, v = score(ck, vs, args.root)
                segs.append(s); vids.append(v)
        s, v = np.concatenate(segs), np.concatenate(vids)
        print(f"{cv.name:<44} {len(v):>5} {np.mean(v > 0.5):>9.1%} {np.mean(s > 0.5):>11.1%} {v.mean():>7.3f} {np.median(v):>9.3f}")
    print("\nFA = false alarm: share of these healthy clips (video level) / 2 s windows (segment level) scored > 0.5")


if __name__ == "__main__":
    main()
