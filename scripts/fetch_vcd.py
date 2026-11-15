"""Stream a subset of Microsoft's VCD2 webcam dataset (https://github.com/microsoft/VCD, data licence
CDLA-Permissive-2.0) and keep only the face landmarks: extra "unaffected" webcam videos.

Why: in palsynet the affected videos are mostly home webcam / selfie recordings and the unaffected
ones studio videos, so a model can learn "webcam = affected". Ordinary people on consumer webcams
break that shortcut (participants are crowd-sourced volunteers, not clinically screened).

Each clip (~180 MB lossless FFV1 at 720p) is downloaded to a temp file, run through the same
MediaPipe landmark + alignment pass as scripts/preprocess_faces.py, saved as
    data/aligned/landmarks/vcd/<hash>.npz      (same keys as the palsynet landmark files)
and deleted, so only a few clips are ever on disk. Also writes data/vcd/selection.csv and
data/vcd/videos.json (VideoInfo records with a fold 0-4 each). Re-running skips finished clips.

    uv run python scripts/fetch_vcd.py --n 100 --workers 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import json
import os
import random
import tempfile
import time
import urllib.request
from multiprocessing import Pool

import cv2
import numpy as np

from palsynet.faces import AlignConfig, detect_landmarks, similarity_transforms

BASE = "https://vcdpublic.blob.core.windows.net/vcd2/"


def select(meta: Path, n: int, seed: int, resolution: str, scenarios: str | None = None,
           exclude: set[str] = frozenset()) -> list[dict]:
    """Default: talking-head (S01), reading (S03) and mixed/video-call (S04) clips at one resolution,
    roughly 60/25/15 %, spread over as many different cameras as possible.
    scenarios="S02:1.0" etc. picks other proportions; clips in `exclude` (relative paths) are skipped."""
    rows = [r for r in csv.DictReader(open(meta)) if r["video_resolution"] == resolution and r["relative_path"] not in exclude]
    rng = random.Random(seed)
    if scenarios:
        fr = {k: float(v) for k, v in (x.split(":") for x in scenarios.split(","))}
        quota = {k: round(n * v) for k, v in fr.items()}
    else:
        quota = {"S01": round(n * 0.6), "S03": round(n * 0.25)}
        quota["S04"] = n - sum(quota.values())
    picked = []
    for sc, q in quota.items():
        pool = [r for r in rows if r["log_scenario_id"] == sc]
        rng.shuffle(pool)
        seen_cam, first, rest = set(), [], []
        for r in pool:  # one clip per camera model first, then fill up
            (rest if r["log_recording_camera_code"] in seen_cam else first).append(r)
            seen_cam.add(r["log_recording_camera_code"])
        picked += (first + rest)[:q]
    return picked


def iter_rgb(path: str):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()


def process(job) -> dict:
    row, out_dir, model = job
    vid = Path(row["relative_path"]).stem
    out = out_dir / f"{vid}.npz"
    if out.exists():
        z = np.load(out)
        return {"id": vid, "frames": len(z["found"]), "fps": float(z["fps"]), "detected": float(z["found"].mean()), "status": "skipped"}
    t0 = time.time()
    fd, tmp = tempfile.mkstemp(suffix=".mkv")
    os.close(fd)
    try:
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(BASE + row["relative_path"], tmp)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        cap = cv2.VideoCapture(tmp)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fps = float(row["video_real_fps"] or 30) or 30.0
        lms, found = detect_landmarks(iter_rgb(tmp), model, fps)
        if not found.any():
            return {"id": vid, "frames": len(found), "fps": fps, "detected": 0.0, "status": "no face"}
        mats = similarity_transforms(lms, found, w, h, AlignConfig())
        np.savez_compressed(out, landmarks=lms.astype(np.float16), found=found, transforms=mats,
                            src_size=np.array([w, h]), fps=np.array(fps))
        return {"id": vid, "frames": len(found), "fps": fps, "detected": float(found.mean()),
                "status": f"done {time.time() - t0:.0f}s"}
    finally:
        os.remove(tmp)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, default=Path("data/vcd/vcd2_metadata.csv"))
    p.add_argument("--out", type=Path, default=Path("data/aligned/landmarks/vcd"))
    p.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--scenarios", help='e.g. "S02:1.0" (default: S01 60%%, S03 25%%, S04 15%%)')
    p.add_argument("--exclude", type=Path, nargs="*", default=[], help="selection.csv files of clips already used")
    p.add_argument("--list-dir", type=Path, help="where selection.csv / videos.json go (default: next to --meta)")
    args = p.parse_args()

    if not args.meta.exists():
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + "metadata/vcd2_metadata.csv", args.meta)
    exclude = {r["relative_path"] for e in args.exclude for r in csv.DictReader(open(e))}
    rows = select(args.meta, args.n, args.seed, args.resolution, args.scenarios, exclude)
    args.out.mkdir(parents=True, exist_ok=True)
    list_dir = args.list_dir or args.meta.parent
    list_dir.mkdir(parents=True, exist_ok=True)
    with open(list_dir / "selection.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"{len(rows)} clips selected ({args.resolution}); ~{len(rows) * 0.18:.0f} GB to stream, deleted after use", flush=True)

    results = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, [(row, args.out, args.model) for row in rows]):
            results.append(r)
            print(f"[{len(results)}/{len(rows)}] {r['id'][:12]} {r['frames']:>4} frames  face {r['detected']:.0%}  {r['status']}", flush=True)

    # VideoInfo-style records, folds assigned round-robin after a seeded shuffle
    ok = sorted((r for r in results if r["detected"] >= 0.9), key=lambda r: r["id"])
    random.Random(args.seed).shuffle(ok)
    videos = [{"video_id": f"vcd_{r['id'][:10]}", "path": f"vcd/{r['id']}.mp4", "label": 0,
               "num_frames": r["frames"], "fps": r["fps"], "fold": i % args.folds} for i, r in enumerate(ok)]
    json.dump(videos, open(list_dir / "videos.json", "w"), indent=1)
    dropped = len(results) - len(ok)
    print(f"{len(videos)} usable clips -> {list_dir / 'videos.json'}" + (f" ({dropped} dropped: face <90%)" if dropped else ""))


if __name__ == "__main__":
    main()
