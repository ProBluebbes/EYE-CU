"""Live / streaming inference: camera, stream URL or video file -> rolling p(affected).

Pipeline (same preprocessing as training):
    frame -> MediaPipe landmarks -> StreamingFaceAligner (3-frame / 0.1 s delay, identical smoothing)
          -> sample at the model's sample_fps by timestamp -> rolling window of segment_seconds
          -> eval transforms from training -> model, every --hop seconds
          -> mean of the last --avg windows (as eval averages segments per video)

Whether frames are face-aligned follows the checkpoint: models trained on data/aligned get aligned
input; models trained on raw frames get raw frames (override with --align / --no-align).

    uv run python scripts/live.py runs/<run>/best.pt --source 0                 # webcam 0
    uv run python scripts/live.py runs/cv_<run>/ --source 0 --show              # ensemble: all folds' best.pt
    uv run python scripts/live.py runs/<run>/best.pt --source clip.mp4 --save out.mp4 --log out.csv
    uv run python scripts/live.py runs/<run>/best.pt --source rtsp://host/stream --show

Research prototype: p(affected) is a model score, not a diagnosis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import queue
import threading
import time
from collections import deque
from fractions import Fraction

import cv2
import numpy as np
import torch

from palsynet.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from palsynet.engine import get_device
from palsynet.faces import AlignConfig, LandmarkDetector, StreamingFaceAligner
from palsynet.landmarks import StreamingLandmarkSmoother
from palsynet.landmarks import frame_features as landmark_features
from palsynet.models import build_model


# ----------------------------------------------------------------------------- sources
class FileSource:
    """Video file: timestamps from the frame index, read as fast as processing allows (or real time)."""

    live = False

    def __init__(self, path: str, realtime: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime, self.i, self.t0 = realtime, 0, None

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None
        ts = self.i / self.fps
        self.i += 1
        if self.realtime:
            self.t0 = self.t0 or time.monotonic()
            time.sleep(max(0.0, self.t0 + ts - time.monotonic()))
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB), ts

    def close(self):
        self.cap.release()


class LiveSource:
    """Camera / network stream read on its own thread; if processing falls behind, the oldest frames are dropped."""

    live = True

    def __init__(self, src, width: int | None = None, height: int | None = None):
        self.src, self.width, self.height = src, width, height
        self.cap = self._open()
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open source {src!r}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self.fps = fps if 5 <= fps <= 120 else 30.0  # macOS cameras can report nonsense (e.g. 1.0)
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.dropped, self.reopened, self._stop, self._t0 = 0, 0, threading.Event(), time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _run(self, *a, **kw):
        try:
            self._read_loop(*a, **kw)
        finally:
            # released here, on the thread that reads: releasing from another thread while cap.read() is in
            # progress crashes OpenCV's macOS camera backend (segfault in grabImageUntilDate)
            self.cap.release()

    def _read_loop(self, reopen_after: float = 2.0, give_up_after: float = 20.0):
        """Cameras can return no frames while warming up, and macOS cameras sometimes stall mid-stream.
        After `reopen_after` s without a frame the camera is reopened; after `give_up_after` s the stream ends."""
        last_ok = last_reopen = time.monotonic()
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok:
                now = time.monotonic()
                if now - last_ok >= give_up_after:
                    print(f"no frames from the source for {give_up_after:.0f}s - stopping", flush=True)
                    self.q.put(None)
                    return
                if now - last_reopen >= reopen_after and now - last_ok >= reopen_after:
                    print(f"no frames for {now - last_ok:.0f}s - reopening the camera", flush=True)
                    self.cap.release()
                    self.cap = self._open()
                    self.reopened += 1
                    last_reopen = time.monotonic()
                time.sleep(0.05)
                continue
            last_ok = time.monotonic()
            item = (cv2.cvtColor(f, cv2.COLOR_BGR2RGB), time.monotonic() - self._t0)
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(item)

    def read(self):
        return self.q.get()

    def close(self):
        """Stop the reader; it releases the camera itself once its current read returns."""
        self._stop.set()
        self._thread.join(timeout=3.0)  # if a read hangs, leave the camera to process exit rather than crash


# ----------------------------------------------------------------------------- model
class SegmentClassifier:
    """One checkpoint, or several (e.g. the 5 cross-validation fold models) averaged as an ensemble.
    All checkpoints must share the same input settings (frames, fps, size, normalisation, alignment)."""

    INPUT_KEYS = ("input", "segment_seconds", "sample_fps", "size", "mean", "std", "landmark_mode", "landmark_smooth",
                  "seq_len", "seq_hop")

    def __init__(self, ckpt_paths: list[Path], device: str = "auto"):
        self.device = get_device(device)
        self.models, cfgs = [], []
        for path in ckpt_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            m = build_model({**cfg["model"], "pretrained": False}, cfg["data"])
            m.load_state_dict(ckpt["model"])
            self.models.append(m.to(self.device).eval())
            cfgs.append(cfg)
        self.cfg = cfg = cfgs[0]
        d = cfg["data"]
        key = lambda c: ([c["data"].get(k) for k in self.INPUT_KEYS], "aligned" in str(c.get("raw_root", "")))
        bad = [str(p) for p, c in zip(ckpt_paths, cfgs) if key(c) != key(cfg)]
        if bad:
            raise SystemExit(f"checkpoints use different input settings than {ckpt_paths[0]}: {bad}")
        self.num_frames = max(1, round(d["segment_seconds"] * d["sample_fps"]))
        self.sample_fps = d["sample_fps"]
        self.landmark_input = d.get("input", "video") == "landmarks"
        self.landmark_3d = self.landmark_input and d.get("landmark_mode", "2d") == "3d"
        self.landmark_smooth = d.get("landmark_smooth", 5)
        # landmark models are trained on aligned landmark coordinates, so they always need the aligner
        self.trained_aligned = self.landmark_input or "aligned" in str(cfg.get("raw_root", ""))
        self.transform = build_transforms(False, d["size"], d.get("mean", IMAGENET_MEAN), d.get("std", IMAGENET_STD))
        # sequence models (landmark_seq_gru): every model keeps the encodings of its recent windows; the
        # verdict uses up to seq_len of them, seq_hop seconds apart, like in training
        self.seq_len, self.seq_hop = d.get("seq_len", 1), d.get("seq_hop", 1.0)
        self.history = [[] for _ in self.models]  # per model: [(timestamp, window encoding)]
        self._reset = False

    def reset(self):
        """Forget the history (face lost / stream restarted); applied before the next prediction."""
        self._reset = True

    def _pick_history(self, hist, ts):
        """Newest-first walk: one stored encoding about every seq_hop seconds before ts (oldest first)."""
        chosen, target, tol = [], ts - self.seq_hop, 0.3 * self.seq_hop
        for t_e, e in reversed(hist):
            if len(chosen) == self.seq_len - 1:
                break
            if t_e <= target + tol:
                chosen.append(e)
                target = t_e - self.seq_hop
        return chosen[::-1]

    @torch.no_grad()
    def probs(self, frames_rgb: list[np.ndarray], ts: float | None = None) -> list[float]:
        """p(affected) of every model in the ensemble for one window (ts = its time, for sequence models)."""
        if self.seq_len > 1:
            if self._reset:
                self.history = [[] for _ in self.models]
                self._reset = False
            x = torch.from_numpy(np.stack(frames_rgb)).float().unsqueeze(0).to(self.device)
            out = []
            for m, hist in zip(self.models, self.history):
                emb = m.encode(x)  # (1, E)
                seq = torch.cat(self._pick_history(hist, ts) + [emb], 0).unsqueeze(0)  # (1, K, E) oldest first
                out.append(float(m.classify(seq, torch.tensor([seq.shape[1]])).float().softmax(-1)[0, 1]))
                hist.append((ts, emb))
                del hist[: max(0, len(hist) - 4 * self.seq_len * 4)]  # keep enough for any --hop >= 0.25 s
            return out
        if self.landmark_input:  # frames are (N, 2|3) landmark arrays
            x = torch.from_numpy(np.stack(frames_rgb)).float().unsqueeze(0).to(self.device)
        else:
            x = torch.from_numpy(np.stack(frames_rgb)).permute(0, 3, 1, 2)  # T, C, H, W uint8
            x = self.transform(x).contiguous().unsqueeze(0).to(self.device)  # contiguous: strided input makes Conv3d on MPS ~6x slower
        return [float(m(x).float().softmax(-1)[0, 1]) for m in self.models]

    def __call__(self, frames_rgb: list[np.ndarray]) -> float:
        return float(np.mean(self.probs(frames_rgb)))


def expand_checkpoints(paths: list[Path]) -> list[Path]:
    """Files are used as-is; a directory (e.g. a cv_* run folder) contributes every best.pt under it."""
    out = []
    for p in paths:
        found = sorted(p.rglob("best.pt")) if p.is_dir() else [p]
        if not found:
            raise SystemExit(f"no best.pt under {p}")
        out += found
    return out


# ----------------------------------------------------------------------------- overlay
def draw(frame_rgb, af, aligned_mode: bool, state: str, p_win, p_avg, buf_len, need, threshold: float = 0.4,
         names: list[str] | None = None, face_on: bool = False):
    """p_win / p_avg: one value per model group (lists), or None before the first prediction."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if af is not None and af.landmarks is not None:
        for x, y in af.landmarks[:, :2] * np.array([w, h]):
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    if aligned_mode and af is not None and af.aligned is not None:
        inset = cv2.cvtColor(af.aligned, cv2.COLOR_RGB2BGR)
        s = min(h, w) // 3
        img[0:s, w - s:w] = cv2.resize(inset, (s, s))
    lines = [state, f"buffer {buf_len}/{need}"]
    if p_win is not None:
        if len(p_win) == 1:
            lines += [f"p(affected) window {p_win[0]:.2f}", f"p(affected) avg    {p_avg[0]:.2f}"]
        else:
            lines += [f"{n:<9s} window {pw:.2f}  avg {pa:.2f}" for n, pw, pa in zip(names, p_win, p_avg)]
    scale = max(0.4, min(h, w) / 700)
    lh = int(22 * scale / 0.5)
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t in lines) + 12
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, lh * len(lines) + 8), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (6, lh * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if face_on:  # space-bar label: the user says they are making a face right now
        t = "MAKING A FACE (space to stop)"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, 2)
        x0 = max(0, w - tw - 16)
        cv2.rectangle(img, (x0, h // 2 - th - 10), (w, h // 2 + 8), (0, 160, 230), -1)
        cv2.putText(img, t, (x0 + 8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    if p_avg is not None:  # decision banners along the bottom (one per model): averaged p(affected) vs --threshold
        multi = len(p_avg) > 1
        texts = [(f"{n}: " if multi else "") + f"{'POSITIVE' if pa >= threshold else 'negative'}  {pa:.2f} / {threshold:.2f}"
                 for n, pa in zip(names or [""] * len(p_avg), p_avg)]
        big = scale * (1.2 if multi else 1.6)
        widest = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, big, 2)[0][0] for t in texts)
        if widest > w - 20:  # shrink to fit narrow frames
            big *= (w - 20) / widest
        bottom = h
        for t, pa in reversed(list(zip(texts, p_avg))):
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, big, 2)
            cv2.rectangle(img, (0, bottom - th - 16), (tw + 20, bottom), (0, 0, 200) if pa >= threshold else (0, 140, 0), -1)
            cv2.putText(img, t, (10, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, big, (255, 255, 255), 2, cv2.LINE_AA)
            bottom -= th + 20
    return img


class VideoSink:
    def __init__(self, path: Path, fps: float):
        import av

        self.av, self.path, self.fps, self.c, self.s = av, path, fps, None, None

    def write(self, bgr):
        if self.c is None:
            self.c = self.av.open(str(self.path), "w")
            self.s = self.c.add_stream("libx264", rate=Fraction(round(self.fps * 1000), 1000))
            self.s.height, self.s.width = bgr.shape[0] // 2 * 2, bgr.shape[1] // 2 * 2
            self.s.pix_fmt = "yuv420p"
        f = self.av.VideoFrame.from_ndarray(bgr[: self.s.height, : self.s.width], format="bgr24")
        self.c.mux(self.s.encode(f))

    def close(self):
        if self.c is not None:
            self.c.mux(self.s.encode())
            self.c.close()


# ----------------------------------------------------------------------------- main loop
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="+", help="best.pt file(s), or a cv_* folder = ensemble of all its folds")
    p.add_argument("--source", default="0", help="camera index, video file, or stream URL")
    p.add_argument("--landmarker", type=Path, default=Path("models/face_landmarker.task"))
    al = p.add_mutually_exclusive_group()
    al.add_argument("--align", dest="align", action="store_true", default=None, help="force face alignment")
    al.add_argument("--no-align", dest="align", action="store_false", help="force raw frames")
    p.add_argument("--hop", type=float, default=1.0, help="seconds between predictions")
    p.add_argument("--avg", type=int, default=5, help="average the last N window predictions")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="averaged p(affected) at or above this is shown as POSITIVE (default 0.4)")
    p.add_argument("--warn-yaw", type=float, default=12.0,
                   help="3D landmark models: flag predictions whose window had the head turned more than this many "
                        "degrees left/right as less reliable (training data is ~90%% within 10 deg; beyond ~15 deg "
                        "the landmarks of the far side are guessed and look asymmetric)")
    p.add_argument("--max-yaw", type=float, default=0.0,
                   help="3D landmark models: pause predictions entirely while the head is turned more than this "
                        "many degrees (0 = never pause, the default)")
    p.add_argument("--max-gap", type=float, default=0.5, help="seconds without a face before the window resets")
    p.add_argument("--device", default="auto")
    p.add_argument("--realtime", action="store_true", help="for files: pace reading to the video's fps")
    p.add_argument("--show", action="store_true", help="display a window (needs opencv-python, not -headless)")
    p.add_argument("--save", type=Path, help="write the annotated video here")
    p.add_argument("--log", type=Path, help="write one CSV row per prediction here")
    p.add_argument("--dump", type=Path, help="save every window's model inputs (landmark models; no video) and each "
                                              "ensemble member's p(affected) to this .npz, for diagnostics")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--compare", action="store_true",
                   help="each checkpoint argument (e.g. several cv_* folders) is a separate model, all run on the same "
                        "feed with their own scores and verdicts, instead of one ensemble")
    p.add_argument("--names", nargs="+", help="labels for the --compare models (default: from the model type)")
    args = p.parse_args()

    groups_paths = [[c] for c in args.checkpoint] if args.compare else [args.checkpoint]
    clfs = [SegmentClassifier(expand_checkpoints(g), args.device) for g in groups_paths]
    clf = clfs[0]  # input settings (window, fps, landmark mode) come from the first; the others must match
    for c, g in zip(clfs[1:], groups_paths[1:]):
        same = (c.num_frames, c.sample_fps, c.landmark_input, c.landmark_3d, c.landmark_smooth, c.trained_aligned) == \
               (clf.num_frames, clf.sample_fps, clf.landmark_input, clf.landmark_3d, clf.landmark_smooth, clf.trained_aligned)
        if not same:
            raise SystemExit(f"--compare: {g[0]} uses different input settings than {groups_paths[0][0]}")
    SHORT = {"landmark_tcn": "TCN", "landmark_geom_cnn": "3D-CNN", "landmark_stgcn": "GraphNet",
             "landmark_seq_gru": "GRU-mem", "landmark_crnn": "CRNN"}
    names = args.names or [SHORT.get(c.cfg["model"]["name"], c.cfg["model"]["name"]) for c in clfs]
    if len(names) != len(clfs):
        raise SystemExit("--names needs one label per model")
    ckpts = [c for g in groups_paths for c in expand_checkpoints(g)]

    def reset_all():
        for c in clfs:
            c.reset()
    aligned_mode = clf.trained_aligned if args.align is None else args.align
    if clf.landmark_input and not aligned_mode:
        raise SystemExit("landmark models need face alignment; drop --no-align")
    is_file = Path(args.source).exists()
    is_camera = args.source.isdigit()
    if is_camera and not (args.width or args.height):
        args.width, args.height = 640, 480  # plenty for 300x300 training data; full HD makes landmarks slow
    src = FileSource(args.source, args.realtime) if is_file else LiveSource(int(args.source) if is_camera else args.source, args.width, args.height)
    detector = LandmarkDetector(args.landmarker) if (aligned_mode or args.show or args.save) else None
    print(("compare " + " | ".join(f"{n} = {c.cfg['model']['name']} x{len(c.models)}" for n, c in zip(names, clfs)) + "\n"
           if len(clfs) > 1 else "") +
          f"model {clf.cfg['model']['name']} x{len(clf.models)} | input {'3D frontalised landmarks' if clf.landmark_3d else 'aligned landmarks' if clf.landmark_input else 'aligned faces' if aligned_mode else 'raw frames'} | "
          f"window {clf.num_frames} frames @ {clf.sample_fps} fps | device {clf.device} | source fps {src.fps:.1f}"
          + (f" | memory: up to {clf.seq_len} windows, {clf.seq_hop:g} s apart (resets when the face is lost)"
             if clf.seq_len > 1 else ""))

    aligner = None
    buf: deque = deque(maxlen=clf.num_frames)
    preds = [deque(maxlen=args.avg) for _ in clfs]  # one rolling average per model group
    next_sample, next_pred, p_win, p_avg = None, None, None, None
    sink = VideoSink(args.save, src.fps) if args.save else None
    log_f = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_f) if log_f else None
    if log:
        log.writerow(["stream_time_s", "first_frame", "last_frame"]
                     + ([f"p_window_{n}" for n in names] + [f"p_avg_{n}" for n in names] if len(clfs) > 1 else ["p_window", "p_avg"])
                     + ["infer_ms"])
    frame_idx_of = deque(maxlen=clf.num_frames)
    yaw_of = deque(maxlen=clf.num_frames)  # |yaw| of each buffered frame (3D landmark models), for --dump
    face_of = deque(maxlen=clf.num_frames)  # space-bar label of each buffered frame (1 = "making a face")
    face_on = [False]                       # current space-bar label
    face_frac = []                          # per prediction: share of the window's frames labelled "making a face"
    n_frames, t_start, infer_ms = 0, time.monotonic(), []
    last_ts = None
    last_pts = None  # landmark models: last frame's landmark features, held through short detection gaps
    # Inference runs on a worker thread so the video never waits for the model. On a live source a
    # window that comes due while the previous one is still running is skipped; on a file we wait.
    from concurrent.futures import ThreadPoolExecutor
    worker = ThreadPoolExecutor(max_workers=1)
    pending = None  # (future, ts, first_frame, last_frame, max |yaw| in window, share labelled "making a face")

    dump = {"t": [], "probs": [], "inputs": [], "max_yaw": [], "group_p": [], "face_frac": []} if args.dump else None

    def infer(frames, ts=None):
        t0 = time.perf_counter()
        ps = [c.probs(frames, ts) for c in clfs]  # per group: every member's p(affected)
        return [float(np.mean(x)) for x in ps], (time.perf_counter() - t0) * 1000, ps, frames

    def collect(wait: bool = False):
        nonlocal pending, p_win, p_avg
        if pending is None or not (wait or pending[0].done()):
            return
        fut, ts, first, last, wyaw, wface = pending
        pending = None
        p_win, ms, ps, frames = fut.result()
        if dump is not None:
            dump["t"].append(ts); dump["probs"].append(ps[0]); dump["max_yaw"].append(wyaw); dump["group_p"].append(p_win)
            dump["face_frac"].append(wface)
            if clf.landmark_input:
                dump["inputs"].append(np.stack(frames))
        infer_ms.append(ms)
        for q, pw in zip(preds, p_win):
            q.append(pw)
        p_avg = [float(np.mean(q)) for q in preds]
        turn = ""
        if wyaw == wyaw:  # not NaN: 3D landmark model
            turn = f"  yaw {wyaw:3.0f}°" + ("  (head turned - less reliable)" if wyaw > args.warn_yaw else "")
        for k, pa in enumerate(p_avg):
            decisions[k].append(pa >= args.threshold)
            win_pos[k].append(p_win[k] >= args.threshold)
        face_frac.append(wface)
        tag = "  [FACE]" if wface >= 0.5 else ("  [face~]" if wface > 0 else "")
        if len(clfs) == 1:
            verdict = "POSITIVE" if decisions[0][-1] else "negative"
            print(f"t={ts:7.2f}s  p(affected) window {p_win[0]:.3f}  avg[{len(preds[0])}] {p_avg[0]:.3f}  {verdict:8s} ({ms:.0f} ms){turn}{tag}", flush=True)
        else:
            cells = "  |  ".join(f"{n} {pw:.2f} avg {pa:.2f} {'POS' if pa >= args.threshold else 'neg'}"
                                 for n, pw, pa in zip(names, p_win, p_avg))
            print(f"t={ts:7.2f}s  {cells}  ({ms:.0f} ms){turn}{tag}", flush=True)
        if log:
            log.writerow([f"{ts:.3f}", first, last] + [f"{v:.4f}" for v in p_win + p_avg] + [f"{ms:.1f}"])

    face_count = {True: 0, False: 0}  # output frames with / without a usable face
    decisions = [[] for _ in clfs]  # per model group: averaged p(affected) >= --threshold, one per prediction
    win_pos = [[] for _ in clfs]    # per model group: single-window p(affected) >= --threshold
    turned_count, turned_state = [0], [False]  # frames skipped because the head was turned too far

    def handle(af_or_frame, ts, af):
        """One output frame (after alignment delay): sample into the window, predict when due."""
        nonlocal next_sample, next_pred, pending, last_pts
        collect()
        turned = False
        if clf.landmark_3d:
            usable = af is not None and af.features is not None
            if usable and args.max_yaw and af.pose is not None and abs(af.pose[0]) > args.max_yaw:
                usable, turned = False, True
        else:
            usable = (af is not None and af.aligned is not None) if aligned_mode else True
        face_count[usable] += 1
        turned_count[0] += turned
        if turned and not turned_state[0]:
            print(f"t={ts:7.2f}s  head turned {abs(af.pose[0]):.0f} deg (> {args.max_yaw:.0f}) - predictions paused, face the camera", flush=True)
        turned_state[0] = turned
        if not usable:
            buf.clear(); frame_idx_of.clear(); yaw_of.clear(); face_of.clear(); next_sample = None; reset_all()
            state = f"head turned {abs(af.pose[0]):.0f} deg - face the camera" if turned else "no face - window reset"
        else:
            if next_sample is None:
                next_sample = ts
            if ts + 1e-6 >= next_sample:
                if clf.landmark_3d:
                    buf.append(af.features)  # smoothed + frontalised; gaps already held by the smoother
                elif clf.landmark_input:
                    if af.landmarks is not None:
                        last_pts = landmark_features(af.landmarks, af.matrix, aligner.width, aligner.height)
                    buf.append(last_pts)  # a held gap frame repeats the last landmarks (offline interpolates)
                else:
                    buf.append(af.aligned if aligned_mode else af_or_frame)
                frame_idx_of.append(af.index if af is not None else None)
                yaw_of.append(abs(float(af.pose[0])) if getattr(af, "pose", None) is not None else np.nan)
                face_of.append(1.0 if face_on[0] else 0.0)
                next_sample += 1.0 / clf.sample_fps
            state = "collecting" if len(buf) < clf.num_frames else "predicting"
            if clf.landmark_3d and af.pose is not None:
                state += f" | yaw {af.pose[0]:+.0f} pitch {af.pose[1]:+.0f}"
                if abs(af.pose[0]) > args.warn_yaw:
                    state += " - turned, less reliable"
            if len(buf) == clf.num_frames and (next_pred is None or ts + 1e-6 >= next_pred) and pending is None:
                pending = (worker.submit(infer, list(buf), ts), ts, frame_idx_of[0], frame_idx_of[-1],
                           float(np.max(yaw_of)) if yaw_of else np.nan, float(np.mean(face_of)) if face_of else 0.0)
                next_pred = ts + args.hop
                if not src.live:
                    collect(wait=True)
        if args.show or sink:
            img = draw(af.frame if af is not None else af_or_frame, af, aligned_mode, state, p_win, p_avg, len(buf), clf.num_frames, args.threshold, names, face_on[0])
            if sink:
                sink.write(img)
            if args.show:
                cv2.imshow("palsynet live", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt
                if key == 32:  # space: toggle the "making a face" label
                    face_on[0] = not face_on[0]
                    print(f"t={ts:7.2f}s  >>> {'MAKING A FACE' if face_on[0] else 'relaxed'} (space)", flush=True)

    try:
        while True:
            item = src.read()
            if item is None:
                break
            frame, ts = item
            n_frames += 1
            if last_ts is not None and ts - last_ts > args.max_gap:  # stream stalled (e.g. camera reopened): start over
                if aligned_mode and aligner is not None:
                    for af in aligner.flush():
                        handle(af.frame, af.timestamp, af)
                buf.clear(); frame_idx_of.clear(); yaw_of.clear(); face_of.clear(); next_sample = None; reset_all()
            last_ts = ts
            lms = detector(frame, ts) if detector is not None else None
            if aligner is None:
                h, w = frame.shape[:2]
                max_gap = max(1, round(args.max_gap * src.fps))
                if clf.landmark_3d:
                    aligner = StreamingLandmarkSmoother(clf.landmark_smooth, w, h, max_gap=max_gap)
                else:
                    aligner = StreamingFaceAligner(AlignConfig(), w, h, max_gap=max_gap)
            if aligned_mode:
                for af in aligner.push(frame, lms, ts):
                    handle(af.frame, af.timestamp, af)
            else:
                from palsynet.faces import AlignedFrame
                handle(frame, ts, AlignedFrame(n_frames - 1, ts, frame, lms, None, None))
        if aligned_mode and aligner is not None:
            for af in aligner.flush():
                handle(af.frame, af.timestamp, af)
        collect(wait=True)
    except KeyboardInterrupt:
        pass
    finally:
        worker.shutdown(wait=True)
        src.close()
        if detector:
            detector.close()
        if sink:
            sink.close()
        if log_f:
            log_f.close()
        if args.show:
            cv2.destroyAllWindows()

    if dump is not None and dump["t"]:
        np.savez_compressed(args.dump, t=np.array(dump["t"]), probs=np.array(dump["probs"]), max_yaw=np.array(dump["max_yaw"]),
                            inputs=np.stack(dump["inputs"]) if dump["inputs"] else np.zeros(0),
                            checkpoints=np.array([str(c) for c in ckpts]),
                            group_p=np.array(dump["group_p"]), group_names=np.array(names),
                            face_frac=np.array(dump["face_frac"]))
        print(f"dumped {len(dump['t'])} windows -> {args.dump}")
    wall = time.monotonic() - t_start
    print(f"\n{n_frames} frames in {wall:.1f}s ({n_frames / max(wall, 1e-9):.1f} fps processed)"
          + (f", {src.dropped} dropped" if getattr(src, "dropped", 0) else "")
          + f", face usable in {face_count[True]}/{face_count[True] + face_count[False]}"
          + (f" ({turned_count[0]} skipped: head turned > {args.max_yaw:.0f} deg)" if turned_count[0] else "")
          + (f" | inference {np.mean(infer_ms):.0f} ms avg over {len(infer_ms)} windows" if infer_ms else "")
          + ("" if p_avg is not None else " | no full window collected"))
    if face_frac and any(face_frac):
        # Space-bar evaluation. Windows: "relaxed" = no labelled frame, "face" = at least half labelled; mixed
        # windows (around a key press) are left out. Verdicts (the --avg average) count only when all the windows
        # they average share one label.
        ff = np.array(face_frac)
        relax_w, face_w = ff == 0, ff >= 0.5
        a = args.avg
        relax_v = np.array([ff[max(0, i - a + 1):i + 1].max() == 0 for i in range(len(ff))])
        face_v = np.array([ff[max(0, i - a + 1):i + 1].min() >= 0.5 for i in range(len(ff))])
        print(f"\nspace-bar evaluation (threshold {args.threshold:.2f}): {relax_w.sum()} relaxed / {face_w.sum()} face windows"
              f" ({len(ff) - relax_w.sum() - face_w.sum()} mixed, left out)")
        print(f"  {'model':<10s}{'relaxed flagged':>17s}{'face flagged':>14s}   {'verdicts: relaxed':>18s}{'face':>7s}")
        for n, wp, dec in zip(names, win_pos, decisions):
            wp, dec = np.array(wp), np.array(dec)
            f = lambda x, m: f"{x[m].mean():.0%}" if m.any() else "-"
            print(f"  {n:<10s}{f(wp, relax_w):>17s}{f(wp, face_w):>14s}   {f(dec, relax_v):>18s}{f(dec, face_v):>7s}")
        print("  relaxed flagged = false alarms (lower is better); face flagged = how often a face you made was "
              "flagged (higher is better only if the face was meant to look asymmetric)")
    if p_avg is not None:
        for n, pa, dec in zip(names, p_avg, decisions):
            print(f"{(n + ': ') if len(clfs) > 1 else ''}final p(affected) avg {pa:.3f} | POSITIVE (avg >= {args.threshold:.2f}) "
                  f"in {np.mean(dec):.0%} of {len(dec)} predictions")


if __name__ == "__main__":
    main()
