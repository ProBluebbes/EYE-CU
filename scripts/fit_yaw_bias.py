"""Fit vision.yaw_bias: the fake asymmetry each feature shows when the head is turned.

    python scripts/fit_yaw_bias.py clips/macsidetoside.mov clips/phonesidetoside.MOV \
        clips/mactilt.mov clips/phonetilt.MOV clips/tilt.mov

Record clips with a NEUTRAL face: slow left-right turns past the gate (vision.max_yaw_deg),
on each device / camera position you care about. Still or tilt clips help pin down the
centre. Every clip gets equal weight, so the curve is a middle ground between setups,
and its own intercept (natural asymmetry differs per session).

Prints a yaw_bias block to paste under `vision:` in strokewatch/config.yaml, plus the
fake score a held head turn would still produce: in-sample, and with each clip left out
of the fit (how well the curve carries over to a setup it hasn't seen).
Landmarks are cached next to each clip (<clip>.landmarks.npz) so refits are fast.
"""
import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapping_report import pick_rotation, rot  # noqa: E402

from strokewatch import load_config  # noqa: E402
from strokewatch.vision import YAW_BIAS_FEATURES, FaceAnalyzer, compute_features  # noqa: E402

KNOTS = np.arange(-60, 61, 10.0)
BINS = np.arange(-45, 46, 5)


def landmarks(path: Path, cfg: dict) -> dict:
    cache = path.with_suffix(".landmarks.npz")
    if cache.exists():
        return dict(np.load(cache))
    key = pick_rotation(str(path), cfg)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    an = FaceAnalyzer(cfg)
    lms, mats, ts, bss, wh, i = [], [], [], [], (0, 0), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = rot(f, key)
        wh = (f.shape[1], f.shape[0])
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        res = an.landmarker.detect_for_video(img, int(i / fps * 1000) + 1)
        i += 1
        if not res.face_landmarks:
            continue
        lms.append([(p.x, p.y, p.z) for p in res.face_landmarks[0]])
        mats.append(np.asarray(res.facial_transformation_matrixes[0]))
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        bss.append([bs.get(k, 0.0) for k in ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight")])
        ts.append(i / fps)
    an.close()
    d = dict(lm=np.array(lms), mat=np.array(mats), t=np.array(ts), bs=np.array(bss), wh=np.array(wh))
    np.savez(cache, **d)
    return d


def features(d: dict, vc: dict) -> dict:
    """Run the real compute_features, with the bias table off and the angle gates open."""
    vc = copy.deepcopy(vc)
    vc.update(yaw_bias=None, max_yaw_deg=90, max_pitch_deg=90, max_roll_deg=90)
    w, h = (int(x) for x in d["wh"])
    out = {k: [] for k in ("yaw", *YAW_BIAS_FEATURES)}
    for lm, m, t, bs in zip(d["lm"], d["mat"], d["t"], d["bs"]):
        pts = [NS(x=a, y=b, z=c) for a, b, c in lm]
        bsl = [NS(category_name=n, score=s) for n, s in zip(("jawOpen", "eyeBlinkLeft", "eyeBlinkRight"), bs)]
        ff = compute_features(pts, bsl, m, w, h, float(t), vc)
        for k in out:
            v = getattr(ff, k)
            out[k].append(np.nan if v is None else v)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def hat(x):
    """Linear-interpolation basis on KNOTS (so the fit is exactly what np.interp evaluates)."""
    eye = np.eye(len(KNOTS))
    return np.stack([np.interp(x, KNOTS, eye[j]) for j in range(len(KNOTS))], axis=1)


def fit(clips: list[dict], feat: str, unit: float, smooth: float) -> np.ndarray:
    xs, ys, ws, ids = [], [], [], []
    for i, F in enumerate(clips):
        ok = ~np.isnan(F[feat]) & (np.abs(F["yaw"]) <= KNOTS[-1])
        xs.append(hat(F["yaw"][ok]))
        ys.append(F[feat][ok] / unit)
        ws.append(np.full(ok.sum(), 1.0 / max(ok.sum(), 1)))  # equal weight per clip
        ids.append(np.full(ok.sum(), i))
    X = np.hstack([np.vstack(xs), np.eye(len(clips))[np.concatenate(ids)]])
    sw = np.sqrt(np.concatenate(ws) * len(clips))
    y = np.concatenate(ys)
    d2 = np.diff(np.eye(len(KNOTS)), 2, axis=0)  # curvature penalty
    reg = np.zeros((len(d2) + 1, X.shape[1]))
    reg[:len(d2), :len(KNOTS)] = d2 * np.sqrt(smooth)
    reg[-1, list(KNOTS).index(0)] = 1e3  # g(0) = 0
    c, *_ = np.linalg.lstsq(np.vstack([X * sw[:, None], reg]),
                            np.concatenate([y * sw, np.zeros(len(reg))]), rcond=None)
    return c[:len(KNOTS)] * unit


def held_z(F: dict, feat: str, g, unit: float, weight: float) -> np.ndarray:
    """Fake weighted z for a head HELD at each angle in BINS (median per 5 deg band),
    relative to the clip's own frames facing the camera."""
    v = F[feat] - (np.interp(F["yaw"], KNOTS, g) if g is not None else 0.0)
    front = np.abs(F["yaw"]) < 5
    if np.sum(front & ~np.isnan(v)) < 8:
        return np.full(len(BINS), np.nan)
    base = np.nanmedian(v[front])
    out = []
    for a in BINS:
        s = (np.abs(F["yaw"] - a) < 2.5) & ~np.isnan(v)
        out.append(weight * (np.median(v[s]) - base) / unit if s.sum() >= 8 else np.nan)
    return np.array(out)


def worst(z: np.ndarray, gate: float) -> str:
    z = z[np.abs(BINS) <= gate]
    return "   -" if np.all(np.isnan(z)) else f"{np.nanmax(np.abs(z)):4.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--smooth", type=float, default=0.3, help="curvature penalty")
    args = ap.parse_args()
    cfg = load_config()
    vc, dc = cfg["vision"], cfg["detector"]
    gate = vc["max_yaw_deg"]
    names = [Path(c).name for c in args.clips]
    clips = []
    for c in args.clips:
        print(f"loading {c} ...", file=sys.stderr)
        clips.append(features(landmarks(Path(c), cfg), vc))
    for n, F in zip(names, clips):
        print(f"# {n}: {len(F['yaw'])} frames, yaw {np.nanmin(F['yaw']):.0f}..{np.nanmax(F['yaw']):.0f}",
              file=sys.stderr)

    curves = {f: fit(clips, f, dc["min_spread"][f], args.smooth) for f in YAW_BIAS_FEATURES}
    print(f"  # Fitted by scripts/fit_yaw_bias.py on: {', '.join(names)}")
    print("  yaw_bias:")
    print(f"    correction: {vc.get('correction', '2d')}")
    print(f"    knots: [{', '.join(f'{k:.0f}' for k in KNOTS)}]")
    for f, g in curves.items():
        print(f"    {f}: [{', '.join(f'{x:.5f}' for x in g)}]")

    print(f"\nWorst fake score for a head held within ±{gate}° (threshold {dc['threshold']}), "
          "uncorrected / corrected / corrected with this clip left out:", file=sys.stderr)
    print(f"{'clip':28s} " + " ".join(f"{f:>20s}" for f in YAW_BIAS_FEATURES), file=sys.stderr)
    for i, (n, F) in enumerate(zip(names, clips)):
        cells = []
        for f in YAW_BIAS_FEATURES:
            u, w = dc["min_spread"][f], dc["weights"][f]
            loo = fit([c for j, c in enumerate(clips) if j != i], f, u, args.smooth) if len(clips) > 1 else None
            cells.append(" / ".join(worst(held_z(F, f, g, u, w), gate) for g in (None, curves[f], loo)))
        print(f"{n:28s} " + " ".join(f"{c:>20s}" for c in cells), file=sys.stderr)


if __name__ == "__main__":
    main()
