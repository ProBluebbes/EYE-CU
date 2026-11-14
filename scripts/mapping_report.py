"""Mapping-accuracy report for recorded test clips (stillness test, head-sweep test).

    python scripts/mapping_report.py clips/still.mov clips/tilt.mov [--iod-mm 90] [--rotate auto]

For each clip: picks the rotation that makes the face upright (camera mount), runs the
landmarker on every frame, and reports:
  - skip rate and reasons
  - noise of each feature (raw per frame, frame-to-frame jitter, after the detector's
    EWMA smoothing), in feature units and in mm (feature * iod_mm)
  - pose bias: linear fit of each feature against yaw/pitch/roll, in mm per 10 degrees
  - talking-gate inputs at rest (lip_gap, jaw_open percentiles)
  - what the real Detector does on the clip (baseline from the first 20 s, max score, alerts)
Writes <clip>.report.png next to the clip.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from strokewatch import apply_overrides, load_config  # noqa: E402
from strokewatch.detector import Detector  # noqa: E402
from strokewatch.vision import FaceAnalyzer, rotate_frame  # noqa: E402

ROTATIONS = ("0", "90", "180", "270")  # clockwise degrees, same as camera.rotate
FEATS = ("mouth_diff", "eye_ratio", "brow_diff")


def rot(frame, key):
    return rotate_frame(frame, int(key))


def pick_rotation(path, cfg):
    """Try each rotation on a few frames; keep the one with faces found and smallest |roll|."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(n - 1, 0), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    best = None
    for key in ROTATIONS:
        an = FaceAnalyzer(cfg)
        rolls = []
        for j, f in enumerate(frames):
            ff, _ = an.process(rot(f, key), j * 1.0)
            if ff.roll is not None:
                rolls.append(abs(ff.roll))
        an.close()
        score = (len(rolls), -np.median(rolls) if rolls else 0)
        if best is None or score > best[0]:
            best = (score, key)
    return best[1]


def ewma(xs, ts, tau):
    out, s, last = [], None, None
    for x, t in zip(xs, ts):
        if s is None:
            s = x
        else:
            a = 1 - math.exp(-max(t - last, 0) / tau)
            s += a * (x - s)
        last = t
        out.append(s)
    return np.array(out)


def analyze(path, cfg, iod_mm, rotate):
    key = pick_rotation(path, cfg) if rotate == "auto" else rotate
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    an, det = FaceAnalyzer(cfg), Detector(cfg)
    rows, i, alert = [], 0, None
    det_skips = {}
    max_score = 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        ff, _ = an.process(rot(f, key), t)
        st = det.update(ff)
        if ff.valid and not st.valid:
            det_skips[st.invalid_reason] = det_skips.get(st.invalid_reason, 0) + 1
        if st.state != "baselining" and ff.valid:
            max_score = max(max_score, st.score)
        if st.state == "ALERT" and alert is None:
            alert = (t, st.side)
        rows.append(ff)
    cap.release()
    an.close()

    print(f"\n=== {path}  ({len(rows)} frames, {fps:.1f} fps, rotation {key} cw -> use camera.rotate={key})")
    reasons = {}
    for r in rows:
        if not r.valid:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    valid = [r for r in rows if r.valid]
    print(f"valid {len(valid)}/{len(rows)} ({100 * len(valid) / max(len(rows), 1):.0f}%)  "
          f"skipped: {reasons or 'none'}")
    if len(valid) < 30:
        print("too few valid frames to analyze")
        return
    ts = np.array([r.t for r in valid])
    pose = {k: np.array([getattr(r, k) for r in valid]) for k in ("yaw", "pitch", "roll")}
    print("pose range (deg): " + "  ".join(
        f"{k} {np.percentile(v, 2):+.0f}..{np.percentile(v, 98):+.0f}" for k, v in pose.items()))

    tau = cfg["detector"]["ewma_tau_s"]
    print(f"\n{'feature':11s} {'median':>8s} {'SD':>7s} {'robustSD':>8s} {'jitter':>7s} "
          f"{'SD@EWMA':>8s}   (mm = units x {iod_mm:g})")
    series = {}
    for k in FEATS:
        m = np.array([(r.t, getattr(r, k)) for r in valid if getattr(r, k) is not None])
        if len(m) < 30:
            continue
        t_k, x = m[:, 0], m[:, 1]
        series[k] = (t_k, x)
        rsd = 1.4826 * np.median(np.abs(x - np.median(x)))
        jit = np.std(np.diff(x)) / math.sqrt(2)
        sm = ewma(x, t_k, tau)
        unit = "" if k == "eye_ratio" else "mm"
        conv = 1 if k == "eye_ratio" else iod_mm
        print(f"{k:11s} {np.median(x):+8.4f} {np.std(x):7.4f} {rsd:8.4f} {jit:7.4f} {np.std(sm):8.4f}"
              + (f"   = SD {np.std(x) * conv:.2f}{unit}, robust {rsd * conv:.2f}{unit}, "
                 f"smoothed {np.std(sm) * conv:.2f}{unit}" if unit else ""))

    print("\npose bias (multivariate linear fit, change per 10 deg):")
    X = np.column_stack([pose["yaw"], pose["pitch"], pose["roll"], np.ones(len(valid))])
    for k in FEATS:
        x = np.array([getattr(r, k) if getattr(r, k) is not None else np.nan for r in valid])
        ok = ~np.isnan(x)
        if ok.sum() < 30:
            continue
        coef, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
        pred = X[ok] @ coef
        r2 = 1 - np.sum((x[ok] - pred) ** 2) / max(np.sum((x[ok] - x[ok].mean()) ** 2), 1e-12)
        conv, unit = (1, "") if k == "eye_ratio" else (iod_mm, "mm")
        print(f"  {k:11s} yaw {coef[0] * 10 * conv:+.2f}{unit}  pitch {coef[1] * 10 * conv:+.2f}{unit}  "
              f"roll {coef[2] * 10 * conv:+.2f}{unit}   R^2 {r2:.2f}")

    lg = np.array([r.lip_gap for r in rows if r.lip_gap is not None])
    jo = np.array([r.jaw_open for r in rows if r.jaw_open is not None])
    if len(lg):
        print(f"\ntalking gate inputs: lip_gap p50 {np.median(lg):.3f} p95 {np.percentile(lg, 95):.3f} "
              f"(gate {cfg['vision']['lip_gap_max']})   jaw_open p50 {np.median(jo):.3f} "
              f"p95 {np.percentile(jo, 95):.3f} (gate {cfg['vision']['jaw_open_max']})")

    print(f"\ndetector: max score after baseline {max_score:.2f} (threshold "
          f"{cfg['detector']['threshold']}), alert: "
          + (f"YES at {alert[0]:.1f}s side {alert[1]}" if alert else "none"))
    if det.spread:
        thr = cfg["detector"]["threshold"]
        print("  spread used: " + ", ".join(
            f"{f} {sp:.4f}" + ("" if f == "eye_ratio" else
                               f" (alert at ~{thr * sp / cfg['detector']['weights'][f] * iod_mm:.1f} mm)")
            for f, sp in det.spread.items()))
    print(f"  frames skipped by detector: {det_skips or 'none'}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, k in zip(axes[0], FEATS):
        if k in series:
            t_k, x = series[k]
            ax.plot(t_k, x, lw=0.6, color="0.6", label="per frame")
            ax.plot(t_k, ewma(x, t_k, tau), lw=1.5, color="#1f77b4", label=f"EWMA {tau:g}s")
            ax.set_title(k)
            ax.set_xlabel("time (s)")
            ax.legend(fontsize=8)
    for ax, (pk, col) in zip(axes[1], [("yaw", "#d62728"), ("pitch", "#2ca02c"), ("roll", "#9467bd")]):
        ax.scatter(pose[pk], [r.mouth_diff for r in valid], s=3, alpha=0.4, color=col)
        ax.set_xlabel(f"{pk} (deg)")
        ax.set_ylabel("mouth_diff")
        ax.set_title(f"mouth_diff vs {pk}")
    fig.suptitle(f"{Path(path).name}: rotation {key}, valid {len(valid)}/{len(rows)}")
    fig.tight_layout()
    out = str(Path(path).with_suffix("")) + ".report.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"plot: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--iod-mm", type=float, default=90.0,
                    help="outer-eye-corner distance of the subject in mm (adult ~85-95)")
    ap.add_argument("--rotate", default="auto", choices=["auto", *ROTATIONS])
    ap.add_argument("--set", action="append", metavar="KEY=VAL", help="config override")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(), a.set)
    for c in a.clips:
        analyze(c, cfg, a.iod_mm, a.rotate)


if __name__ == "__main__":
    main()
