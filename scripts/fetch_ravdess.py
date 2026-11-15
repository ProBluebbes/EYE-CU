"""Stream the RAVDESS speech videos (https://zenodo.org/records/1188976, 24 actors, licence
CC BY-NC-SA 4.0: non-commercial use only) and keep only the face landmarks as extra "unaffected"
videos of healthy people making facial expressions.

Why: the VCD webcam clips (scripts/fetch_vcd.py) are mostly people talking or reading calmly, so a
model can still read "big expression" as a sign of palsy. RAVDESS actors say short sentences with
neutral, calm, happy, sad, angry, fearful, disgusted and surprised faces, at normal and strong intensity.

One actor zip (~550 MB) at a time is downloaded to a temp dir. Its video-only speech clips
(file names 02-01-*.mp4) go through the same MediaPipe landmark + alignment pass as
scripts/preprocess_faces.py and are saved as
    data/aligned/landmarks/ravdess/<clip>_<actor>.npz
Then the zip is deleted. Also writes data/ravdess/videos.json (VideoInfo records). Folds are assigned
by actor, so an actor's clips are all in one fold. Re-running skips actors that are already done.

    uv run python scripts/fetch_ravdess.py --workers 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

from palsynet.faces import AlignConfig, detect_landmarks, similarity_transforms

URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1"
EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}


def iter_rgb(path: str):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()


def process_actor(job) -> list[dict]:
    actor, out_dir, model = job
    done_flag = out_dir / f".actor_{actor:02d}.json"
    if done_flag.exists():
        return json.load(open(done_flag))
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix=f"ravdess{actor:02d}_"))
    results = []
    try:
        zpath = tmp / "a.zip"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(URL.format(actor), zpath)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10)
        with zipfile.ZipFile(zpath) as z:
            members = [m for m in z.namelist() if Path(m).name.startswith("02-01-") and m.endswith(".mp4")]
            for m in sorted(members):
                clip = tmp / Path(m).name
                with z.open(m) as src, open(clip, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                cap = cv2.VideoCapture(str(clip))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                cap.release()
                lms, found = detect_landmarks(iter_rgb(str(clip)), model, fps)
                clip.unlink()
                parts = [int(x) for x in clip.stem.split("-")]  # modality-channel-emotion-intensity-statement-rep-actor
                rec = {"name": clip.stem, "frames": len(found), "fps": float(fps), "detected": float(found.mean()) if len(found) else 0.0,
                       "emotion": EMOTIONS[parts[2]], "intensity": "strong" if parts[3] == 2 else "normal", "actor": actor}
                if found.any():
                    mats = similarity_transforms(lms, found, w, h, AlignConfig())
                    np.savez_compressed(out_dir / f"{clip.stem}.npz", landmarks=lms.astype(np.float16), found=found,
                                        transforms=mats, src_size=np.array([w, h]), fps=np.array(fps))
                results.append(rec)
        zpath.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(results, open(done_flag, "w"))
    print(f"actor {actor:02d}: {len(results)} clips in {time.time() - t0:.0f}s", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/aligned/landmarks/ravdess"))
    p.add_argument("--meta", type=Path, default=Path("data/ravdess"))
    p.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    p.add_argument("--actors", type=int, default=24)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.meta.mkdir(parents=True, exist_ok=True)
    print(f"{args.actors} actors, ~{args.actors * 0.56:.0f} GB to stream (one zip per worker on disk at a time)", flush=True)

    with Pool(args.workers) as pool:
        per_actor = pool.map(process_actor, [(a, args.out, args.model) for a in range(1, args.actors + 1)], chunksize=1)

    ok = [r for rs in per_actor for r in rs if r["detected"] >= 0.9 and r["frames"] >= 60]
    videos = [{"video_id": f"rav_{r['name']}", "path": f"ravdess/{r['name']}.mp4", "label": 0,
               "num_frames": r["frames"], "fps": r["fps"], "fold": (r["actor"] - 1) % args.folds} for r in ok]
    json.dump(videos, open(args.meta / "videos.json", "w"), indent=1)
    json.dump(ok, open(args.meta / "clips.json", "w"), indent=1)
    n_all = sum(len(rs) for rs in per_actor)
    print(f"{len(videos)} usable clips of {n_all} -> {args.meta / 'videos.json'}")


if __name__ == "__main__":
    main()
