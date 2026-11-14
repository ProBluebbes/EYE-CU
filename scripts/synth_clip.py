"""Make a synthetic test clip from a single face photo: head-roll wobble, then a
progressive droop of the patient's LEFT mouth corner. For testing without a camera.

    python scripts/synth_clip.py face.jpg clips/synth_droop.mp4 [--pre-droop 6] [--droop 14]
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strokewatch import load_config  # noqa: E402
from strokewatch.vision import EYE_OUT_L, EYE_OUT_R, MOUTH_L, FaceAnalyzer  # noqa: E402


def droop(img, cx, cy, amount_px, sigma):
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dy = amount_px * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return cv2.remap(img, xs, ys - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=50)
    ap.add_argument("--droop-start", type=float, default=32)
    ap.add_argument("--pre-droop", type=float, default=0, help="px of droop present from the start")
    ap.add_argument("--droop", type=float, default=14, help="extra px of droop added later")
    a = ap.parse_args()

    img = cv2.imread(a.photo)
    h, w = img.shape[:2]
    an = FaceAnalyzer(load_config())
    _, px = an.process(img, 0.0)
    if px is None:
        raise SystemExit("no face found in photo")
    cx, cy = px[MOUTH_L]
    iod = float(np.linalg.norm(px[EYE_OUT_L] - px[EYE_OUT_R]))
    print(f"mouth corner at ({cx:.0f},{cy:.0f}), eye distance {iod:.0f}px")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(int(a.seconds * a.fps)):
        t = i / a.fps
        extra = a.droop * min(1.0, max(0.0, (t - a.droop_start) / 3))
        f = droop(img, cx, cy, a.pre_droop + extra, 0.22 * iod) if a.pre_droop + extra > 0 else img
        ang = 10 * math.sin(t * 0.8)  # head roll wobble, should not alert
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        f = np.clip(f.astype(np.int16) + rng.normal(0, 3, f.shape).astype(np.int16), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
