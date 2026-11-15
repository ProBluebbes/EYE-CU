"""Synthetic-droop benchmark: StrokeWatch's droop detector vs the learned palsy models, on real clips.

Step 1 (this script, `extract`): for every clip, build a timeline at StrokeWatch's analysis rate (15 fps,
longest side 640 px): the clip forward, then backward (so the head motion continues without a jump). The
first half is where StrokeWatch builds its baseline. In the second half a one-sided mouth droop is warped into
the image (the same Gaussian warp as scripts/synth_clip.py, centred on that frame's mouth corner, ramping up over
3 s), for each side x strength. The untouched timeline is the control (false alarms under identical motion).
MediaPipe runs on every version; per frame we keep StrokeWatch's FrameFeatures and the raw (478, 3) landmarks.

    uv run python scripts/droop_benchmark.py extract strokewatch/*.mov strokewatch/*.MOV --out runs/droop_bench
"""
from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FPS = 15.0
MAX_SIDE = 640
RAMP_S = 3.0
STRENGTHS = {"mild": 0.06, "moderate": 0.12, "strong": 0.20}  # mouth-corner drop, x eye distance
SIDES = ("left", "right")  # patient's side


ROTATE = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def load_timeline(path: Path, rotate: str | None = None) -> list[bytes]:
    """Frames at 15 fps (nearest source frame), longest side 640, forward then backward, as JPEG bytes.
    rotate: "cw" / "ccw" / "180" for a clip recorded sideways (like the phone page's Rotate button)."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, k, next_t = [], 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = k / src_fps
        k += 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / FPS
        if rotate:
            f = cv2.rotate(f, ROTATE[rotate])
        h, w = f.shape[:2]
        s = MAX_SIDE / max(h, w)
        f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
    cap.release()
    return frames + frames[-2::-1]


def droop(img: np.ndarray, cx: float, cy: float, amount_px: float, sigma: float) -> np.ndarray:
    """Pull the image down around (cx, cy) (Gaussian falloff), only inside a +-3 sigma patch."""
    if amount_px <= 0:
        return img
    h, w = img.shape[:2]
    r = int(3 * sigma) + 2
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + int(amount_px) + 2)
    if x1 <= x0 or y1 <= y0:
        return img
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    dy = amount_px * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, xs, ys - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def run_pass(frames, cfg, onset: int = 0, centres=None, side=None, frac=0.0, reuse=None):
    """MediaPipe + StrokeWatch features on the timeline. With side/frac: droop from frame `onset` on, at the
    control pass's mouth corner (centres[i] = (cx, cy, eye_dist) or None). Frames before `onset` reuse `reuse`."""
    from strokewatch.vision import FaceAnalyzer

    an = FaceAnalyzer(cfg)
    out = list(reuse[:onset]) if reuse is not None else []
    for i in range(len(out), len(frames)):
        img = cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR)
        t = i / FPS
        if side is not None and centres[i] is not None:
            cx, cy, iod = centres[i]
            ramp = min(1.0, (i - onset) / (RAMP_S * FPS))
            img = droop(img, cx, cy, frac * iod * ramp, 0.22 * iod)
        ff, px = an.process(img, t)
        out.append((ff, None if px is None else an.last_lms.copy(), None if px is None else px.copy(), img.shape[:2]))
    an.close()
    return out


def extract_clip(job) -> str:
    spec, out_dir = job
    path, _, rotate = spec.partition(":")  # "clip.mov:cw" = rotate 90 deg clockwise first
    from strokewatch import load_config
    from strokewatch.vision import EYE_OUT_L, EYE_OUT_R, MOUTH_L, MOUTH_R

    cfg = load_config()
    frames = load_timeline(Path(path), rotate or None)
    onset = len(frames) // 2
    control = run_pass(frames, cfg)
    corners = {"left": MOUTH_L, "right": MOUTH_R}
    runs = {"control": control}
    for side in SIDES:
        centres = []
        for ff, lms, px, _ in control:
            if px is None:
                centres.append(None)
                continue
            iod = float(np.linalg.norm(px[EYE_OUT_L] - px[EYE_OUT_R]))
            cx, cy = px[corners[side]]
            centres.append((float(cx), float(cy), iod))
        for name, frac in STRENGTHS.items():
            runs[f"{side}_{name}"] = run_pass(frames, cfg, onset, centres, side, frac, reuse=control)
    rec = {"clip": Path(path).name, "fps": FPS, "onset": onset, "n": len(frames),
           "runs": {k: [(ff, lms, shape) for ff, lms, px, shape in v] for k, v in runs.items()}}
    dst = Path(out_dir) / f"{Path(path).stem}.pkl"  # a rotated clip replaces its unrotated data
    with open(dst, "wb") as f:
        pickle.dump(rec, f)
    face = np.mean([lms is not None for _, lms, _, _ in control])
    return f"{Path(path).name}: {len(frames)} frames ({len(frames) / FPS:.0f} s), onset {onset / FPS:.0f} s, face {face:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("clips", nargs="+", help='video files; "clip.mov:cw" rotates a sideways clip first')
    ex.add_argument("--out", default="runs/droop_bench")
    ex.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(min(a.workers, len(a.clips))) as pool:
        for msg in pool.imap_unordered(extract_clip, [(c, a.out) for c in a.clips]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
