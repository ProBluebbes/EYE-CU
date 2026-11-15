"""Make a demo video from a real recording: the face stays normal long enough for the
baseline, then one side of the mouth slowly droops (a central facial palsy look: mouth
corner, lip and lower cheek sag on one side, forehead and eyes untouched).

    python scripts/demo_droop.py clips/still.mov clips/demo_droop.mp4
    python scripts/demo_droop.py clips/still.mov clips/demo_right.mp4 --side right --droop-mm 4
    python scripts/demo_droop.py clips/still.mov clips/demo_label.mp4 --label   # mm readout burned in

The output is upright (rotation picked automatically), so run it with camera.rotate 0:
    python -m strokewatch.server --camera "Bed 3=clips/demo_droop.mp4"
    python -m strokewatch.replay clips/demo_droop.mp4

Droop is in mm of real mouth-corner drop, assuming --iod-mm between the outer eye corners.
A CSV next to the output lists the droop per frame (t, droop_mm).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_report import pick_rotation  # noqa: E402
from strokewatch import load_config  # noqa: E402
from strokewatch.vision import EYE_OUT_L, EYE_OUT_R, FaceAnalyzer, rotate_frame  # noqa: E402

# Control points on the patient's LEFT (MediaPipe face mesh) and their RIGHT mirrors:
# (left idx, right idx, share of the droop moved down, share moved toward the midline,
#  Gaussian radius as a fraction of eye-corner distance).
CONTROLS = [
    (291, 61, 1.00, 0.20, 0.17),   # mouth corner (its bump also sags the outer lips)
    (436, 216, 0.40, 0.00, 0.20),  # lower cheek beside the mouth (flattens the smile line)
    (425, 205, 0.20, 0.00, 0.18),  # mid cheek
]


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def track(path, rot, scale, cfg):
    """Pass 1: landmarks for every frame (output coordinates); NaN where no face."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    an = FaceAnalyzer(cfg)
    pts, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = prep(f, rot, scale)
        _, px = an.process(f, i / fps)
        pts.append(px if px is not None else np.full((478, 2), np.nan, np.float32))
        i += 1
        if i % 300 == 0:
            print(f"  tracked {i} frames", flush=True)
    cap.release()
    an.close()
    return np.array(pts), fps


def smooth(pts, win):
    """Centred moving average over time, filling short gaps, so the warp doesn't jitter."""
    out = pts.copy()
    ok = ~np.isnan(pts[:, 0, 0])
    if ok.sum() == 0:
        raise SystemExit("no face found in the clip")
    idx = np.arange(len(pts))
    flat = pts.reshape(len(pts), -1)
    filled = np.empty_like(flat)
    for c in range(flat.shape[1]):
        filled[:, c] = np.interp(idx, idx[ok], flat[ok, c])
    k = np.ones(win) / win
    pad = np.pad(filled, ((win // 2, win - 1 - win // 2), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(flat.shape[1])], 1)
    out = sm.reshape(pts.shape)
    return out, ok


def prep(frame, rot, scale):
    frame = rotate_frame(frame, rot)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def warp(frame, lm, droop_px, side):
    """Pull the control points down (face's own 'down', so head tilt is respected)."""
    if droop_px <= 0:
        return frame
    eye_r, eye_l = lm[EYE_OUT_R], lm[EYE_OUT_L]
    iod = float(np.linalg.norm(eye_l - eye_r))
    u = (eye_l - eye_r) / iod                      # toward the patient's left
    down = np.array([-u[1], u[0]], np.float32)     # perpendicular, toward the chin
    toward_mid = -u if side == "left" else u
    ctr, disp, sig = [], [], []
    for li, ri, dn, mid, s in CONTROLS:
        ctr.append(lm[li if side == "left" else ri])
        disp.append(droop_px * (dn * down + mid * toward_mid))
        sig.append(s * iod)
    ctr, disp, sig = np.array(ctr), np.array(disp), np.array(sig)

    h, w = frame.shape[:2]
    r = 3.5 * sig.max() + droop_px
    x0, y0 = int(max(0, ctr[:, 0].min() - r)), int(max(0, ctr[:, 1].min() - r))
    x1, y1 = int(min(w, ctr[:, 0].max() + r)), int(min(h, ctr[:, 1].max() + r))
    if x1 <= x0 or y1 <= y0:
        return frame
    # Inverse-distance (Shepard) interpolation hits each control point's target exactly and
    # never exceeds the largest one; the Gaussian envelope fades it to zero away from them.
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    num = np.zeros(xs.shape + (2,), np.float32)
    den = np.zeros_like(xs)
    keep = np.ones_like(xs)  # product of (1 - g_i); envelope = 1 - keep
    for c, d, s in zip(ctr, disp, sig):
        r2 = (xs - c[0]) ** 2 + (ys - c[1]) ** 2
        wi = 1.0 / (r2 + 1e-3)
        num += wi[..., None] * d
        den += wi
        keep *= 1.0 - np.exp(-r2 / (2 * s * s))
    field = (1.0 - keep)[..., None] * num / den[..., None]
    dx, dy = field[..., 0], field[..., 1]
    out = frame.copy()
    # Backward map: the pixel at x shows what was at x - d(x).
    out[y0:y1, x0:x1] = cv2.remap(frame, xs - dx, ys - dy, cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT)
    return out


def label(frame, text):
    h, w = frame.shape[:2]
    fs = max(0.5, w / 1400)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.rectangle(frame, (10, h - th - 26), (24 + tw, h - 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (17, h - 18), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2,
                cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="patient's weak side (patient's left = right of an unmirrored image)")
    ap.add_argument("--droop-mm", type=float, default=5.0, help="final mouth-corner droop, real mm")
    ap.add_argument("--normal-s", type=float, default=30.0,
                    help="seconds of normal face before the droop starts (baseline needs ~20 s)")
    ap.add_argument("--ramp-s", type=float, default=60.0, help="seconds from no droop to full droop")
    ap.add_argument("--seconds", type=float, default=None,
                    help="output length (default: the clip, or normal + ramp + 20 s if the clip is shorter: "
                         "then it plays forward/backward)")
    ap.add_argument("--iod-mm", type=float, default=90.0, help="outer-eye-corner distance, mm")
    ap.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--max-side", type=int, default=1280, help="downscale so the longest side is this")
    ap.add_argument("--label", action="store_true", help="burn in 'SIMULATED droop x.x mm'")
    a = ap.parse_args()

    cfg = load_config()
    cfg["camera"]["rotate"] = 0
    rot = int(pick_rotation(a.clip, cfg) if a.rotate == "auto" else a.rotate)
    cap = cv2.VideoCapture(a.clip)
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read {a.clip}")
    scale = min(1.0, a.max_side / max(f0.shape[:2]))
    print(f"rotation {rot} cw, scale {scale:.2f}; tracking...")

    pts, fps = track(a.clip, rot, scale, cfg)
    n_src = len(pts)
    # Output timeline. A clip too short for baseline + ramp + a hold plays forward, then backward, and so
    # on (no jump in the motion at the turns).
    want = a.seconds if a.seconds is not None else max(n_src / fps, a.normal_s + a.ramp_s + 20)
    n = int(round(want * fps))
    period = max(1, 2 * (n_src - 1))
    idx = np.array([(k % period) if (k % period) < n_src else period - (k % period) for k in range(n)])
    pingpong = n > n_src
    lm, has_face = smooth(pts[idx], win=max(3, int(round(fps * 0.2)) | 1))
    iod_px = np.nanmedian(np.linalg.norm(pts[:, EYE_OUT_L] - pts[:, EYE_OUT_R], axis=1))
    print(f"{n} frames at {fps:.1f} fps ({n_src} in the clip{', played forward/backward' if pingpong else ''}), "
          f"face in {has_face.mean():.0%}, eye distance ~{iod_px:.0f} px ({iod_px / a.iod_mm:.2f} px/mm)")

    h, w = prep(f0, rot, scale).shape[:2]
    w, h = w - w % 2, h - h % 2  # x264 needs even sizes
    cache = None
    if pingpong:  # short clip: keep its frames in memory
        if n_src * w * h * 3 > 4e9:
            raise SystemExit("clip too long to loop in memory; pass --seconds up to the clip length")
        cap = cv2.VideoCapture(a.clip)
        cache = []
        while len(cache) < n_src:
            ok, f = cap.read()
            if not ok:
                break
            cache.append(prep(f, rot, scale)[:h, :w])
        cap.release()
        idx = np.minimum(idx, len(cache) - 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-crf", "18",
         "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)
    cap = None if pingpong else cv2.VideoCapture(a.clip)
    log = []
    for i in range(n):
        if pingpong:
            f = cache[idx[i]]
        else:
            ok, f = cap.read()
            if not ok:
                break
            f = prep(f, rot, scale)[:h, :w]
        t = i / fps
        mm = a.droop_mm * smoothstep((t - a.normal_s) / max(a.ramp_s, 1e-6))
        iod = float(np.linalg.norm(lm[i, EYE_OUT_L] - lm[i, EYE_OUT_R]))
        f = warp(f, lm[i], mm / a.iod_mm * iod, a.side)
        if a.label:
            f = label(f, f"SIMULATED {a.side} droop {mm:.1f} mm   t={t:5.1f}s")
        enc.stdin.write(np.ascontiguousarray(f).tobytes())
        log.append((round(t, 3), round(mm, 3)))
    if cap is not None:
        cap.release()
    enc.stdin.close()
    enc.wait()
    with open(Path(a.out).with_suffix(".droop.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "droop_mm"])
        wr.writerows(log)
    print(f"wrote {a.out} ({len(log) / fps:.0f} s: normal until {a.normal_s:.0f} s, "
          f"{a.side} droop reaches {a.droop_mm:g} mm at {a.normal_s + a.ramp_s:.0f} s)")


if __name__ == "__main__":
    main()
