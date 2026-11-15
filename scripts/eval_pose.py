"""Head-angle robustness test for 3D landmark models: are 2 s windows with a turned head scored differently?

Every fold's model scores only its own test fold: the palsynet test videos plus the VCD webcam clips of that
fold (data/vcd/videos.json, healthy). Windows (2 s, 1 s apart) are grouped by the largest head yaw inside them.
Reported per group: healthy windows called "affected" (false alarms) and affected windows caught.

    uv run python scripts/eval_pose.py runs/cv_landmark_tcn_3d_vcd_expr_still_* runs/cv_..._turn_*
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import numpy as np
import torch

from palsynet.landmarks import load_video_landmarks_3d
from palsynet.models import build_model
from palsynet.splits import VideoInfo, load_splits

BINS = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 90)]


@torch.no_grad()
def windows(cv: Path, videos: list[VideoInfo], root: Path, folds=None):
    """-> list of (label, max |yaw| in window, p(affected)) over every fold's test videos."""
    out = []
    for ck in sorted(cv.glob("*_fold*/best.pt")):
        k = int(ck.parent.name.split("_fold")[1].split("_")[0])
        if folds is not None and k not in folds:
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        d = c["config"]["data"]
        m = build_model({**c["config"]["model"], "pretrained": False}, d)
        m.load_state_dict(c["model"]); m.eval()
        for v in (v for v in videos if v.fold == k):
            x, yaw = load_video_landmarks_3d(root / Path(v.path).with_suffix(".npz"), d.get("landmark_smooth", 5), return_yaw=True)
            span = round(d["segment_seconds"] * v.fps)
            starts = np.arange(0, max(len(x) - span, 0) + 1, max(1, round(v.fps)))
            idx = np.stack([np.linspace(s, s + span - 1, d["segment_seconds"] * d["sample_fps"]).round().astype(int) for s in starts])
            idx = np.clip(idx, 0, len(x) - 1)
            p = m(torch.from_numpy(x[idx]).float()).softmax(-1)[:, 1].numpy()
            out += [(v.label, float(yaw[i[0]:i[-1] + 1].max()), float(q)) for i, q in zip(idx, p)]
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cv_dirs", nargs="+", type=Path)
    ap.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--extra", type=Path, nargs="*", default=[Path("data/vcd/videos.json")])
    ap.add_argument("--root", type=Path, default=Path("data/aligned/landmarks"))
    ap.add_argument("--folds", type=int, nargs="*", help="only these folds (e.g. while a CV run is unfinished)")
    args = ap.parse_args()
    _, videos = load_splits(args.splits)
    videos = list(videos) + [VideoInfo(**r) for e in args.extra for r in json.load(open(e))]
    head = "  ".join(f"{f'{a}-{b}°':>13}" for a, b in BINS)
    for cv in args.cv_dirs:
        w = windows(cv, videos, args.root, args.folds)
        print(f"\n{cv.name}\n  {'max |yaw| in window':28s}{head}")
        for lab, name in [(0, "healthy: flagged > 0.5"), (1, "affected: caught > 0.5")]:
            cells = []
            for a, b in BINS:
                s = w[(w[:, 0] == lab) & (w[:, 1] >= a) & (w[:, 1] < b)]
                cells.append(f"{np.mean(s[:, 2] > 0.5):6.0%} (n={len(s):4d})" if len(s) else f"{'-':>13}")
            print(f"  {name:28s}" + "  ".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":
    main()
