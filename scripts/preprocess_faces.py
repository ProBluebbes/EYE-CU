"""Detect facial landmarks and write face-normalised videos.

For every data/raw/data/<label>/<n>.mp4 this writes
    data/aligned/<label>/<n>.mp4                 aligned face video, same frame count + fps, H.264 (crf 17)
    data/aligned/landmarks/<label>/<n>.npz       landmarks (T,478,3) normalised source coords, float16, NaN = no face
                                                 found (T,) bool, transforms (T,2,3) source px -> aligned px,
                                                 src_size (w,h)
    data/aligned/report.csv                      per-video detection rate
Because relative paths are unchanged, data/splits.json works as-is: point `raw_root` at data/aligned.

    uv run python scripts/preprocess_faces.py                 # all videos, 4 processes
    uv run python scripts/preprocess_faces.py --only affected/3 unaffected/5 --preview runs/previews
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import time
from fractions import Fraction
from multiprocessing import Pool

import av
import cv2
import numpy as np

from palsynet.faces import AlignConfig, detect_landmarks, landmarks_to_aligned, similarity_transforms, warp
from palsynet.splits import CLASSES


def iter_rgb(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()


def process(job) -> dict:
    src, rel, out_root, model, cfg, preview_dir, overwrite = job
    out_vid = out_root / rel
    out_npz = out_root / "landmarks" / rel.with_suffix(".npz")
    vid_id = f"{rel.parent.name}_{rel.stem}"
    if out_vid.exists() and out_npz.exists() and not overwrite:
        found = np.load(out_npz)["found"]
        return {"video_id": vid_id, "frames": len(found), "detected": float(found.mean()), "seconds": 0.0, "status": "skipped"}

    t0 = time.time()
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    lms, found = detect_landmarks(iter_rgb(src), model, fps)  # pass 1
    mats = similarity_transforms(lms, found, w, h, cfg)

    out_vid.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_vid), "w") as container:  # pass 2
        stream = container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        stream.width = stream.height = cfg.size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "17", "preset": "medium"}
        n = 0
        for i, f in enumerate(iter_rgb(src)):
            frame = av.VideoFrame.from_ndarray(warp(f, mats[i], cfg.size), format="rgb24")
            container.mux(stream.encode(frame))
            n += 1
        container.mux(stream.encode())
    if n != len(lms):
        raise RuntimeError(f"{src}: decoded {n} frames in pass 2 but {len(lms)} in pass 1")
    np.savez_compressed(out_npz, landmarks=lms.astype(np.float16), found=found, transforms=mats, src_size=np.array([w, h]))

    if preview_dir is not None:
        write_preview(src, out_vid, lms, found, mats, w, h, Path(preview_dir) / f"{vid_id}.jpg")
    return {"video_id": vid_id, "frames": n, "detected": float(found.mean()), "seconds": round(time.time() - t0, 1), "status": "done"}


def write_preview(src, out_vid, lms, found, mats, w, h, path: Path, k: int = 4) -> None:
    """Top row: source frames with landmarks (red = no detection, interpolated transform).
    Bottom row: aligned frames with landmarks mapped through the transform, and eye-line guide."""
    idx = np.linspace(0, len(lms) - 1, k).round().astype(int)
    aligned_pts = landmarks_to_aligned(lms, mats, w, h)
    top, bottom = [], []
    for i in idx:
        for path_, row in ((src, top), (out_vid, bottom)):
            cap = cv2.VideoCapture(str(path_)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read(); cap.release()
            f = cv2.resize(f, (224, 224))
            if row is top:
                if found[i]:
                    for x, y in lms[i, :, :2] * np.array([224, 224]):
                        cv2.circle(f, (int(x), int(y)), 1, (0, 255, 0), -1)
                else:
                    cv2.rectangle(f, (0, 0), (223, 223), (0, 0, 255), 4)
            else:
                if found[i]:
                    for x, y in aligned_pts[i] * (224 / f.shape[1]):
                        cv2.circle(f, (int(x), int(y)), 1, (0, 255, 0), -1)
                cv2.line(f, (0, int(0.38 * 224)), (223, int(0.38 * 224)), (255, 200, 0), 1)
                cv2.line(f, (112, 0), (112, 223), (255, 200, 0), 1)
            cv2.putText(f, f"frame {i}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            row.append(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(top), np.hstack(bottom)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw/data"))
    p.add_argument("--out", type=Path, default=Path("data/aligned"))
    p.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--eye-y", type=float, default=0.38)
    p.add_argument("--eye-dist", type=float, default=0.27)
    p.add_argument("--smooth-window", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*", help="subset like affected/3 unaffected/5")
    p.add_argument("--preview", type=Path, help="write a raw-vs-aligned preview JPEG per video here")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing {args.model}. Download it:\n  curl -L -o {args.model} "
                         "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    cfg = AlignConfig(args.size, args.eye_y, args.eye_dist, args.smooth_window)
    rels = [Path(c) / v.name for c in CLASSES for v in sorted((args.raw / c).glob("*.mp4"), key=lambda x: int(x.stem))]
    if args.only:
        rels = [r for r in rels if str(r.with_suffix("")) in set(args.only)]
    # longest first so the pool doesn't end on one long straggler
    rels.sort(key=lambda r: -int(cv2.VideoCapture(str(args.raw / r)).get(cv2.CAP_PROP_FRAME_COUNT)))
    jobs = [(args.raw / r, r, args.out, args.model, cfg, args.preview, args.overwrite) for r in rels]

    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}] {r['video_id']:<16} {r['frames']:>6} frames  face found {r['detected']:.1%}  {r['seconds']}s  {r['status']}", flush=True)

    report = args.out / "report.csv"
    existing = {}
    if report.exists():
        existing = {row["video_id"]: row for row in csv.DictReader(open(report))}
    existing.update({r["video_id"]: r for r in rows})
    with open(report, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["video_id", "frames", "detected", "seconds", "status"])
        wr.writeheader()
        wr.writerows(sorted(existing.values(), key=lambda r: r["video_id"]))
    low = [r for r in rows if r["detected"] < 0.9]
    if low:
        print("WARNING: face found in <90% of frames for:", ", ".join(f"{r['video_id']} ({r['detected']:.0%})" for r in low))


if __name__ == "__main__":
    main()
