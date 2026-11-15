"""Landmark-only input: face geometry over time, no pixels.

Why: in this dataset the two classes were recorded very differently (affected = home selfie /
webcam clips, unaffected = studio talking-head videos), so pixel models can learn recording style
instead of palsy. Landmark coordinates carry no lighting, background, sharpness or compression,
which removes that shortcut; what is left is shape and movement, which is where palsy shows.

Coordinates: the 478 MediaPipe points are mapped through the same smoothed similarity transform
used for the aligned videos (so head position, size and roll are removed), then expressed in
"eye units": origin at the midpoint of the eye centres, x to the right, 1 unit = distance
between the eye centres. Only 152 points in the expressive regions are kept (lips, eyes,
eyebrows, nose, face oval); the iris points are dropped because they follow gaze.

MIRROR[i] is the landmark that point i maps onto when the face is mirrored left<->right
(found by matching the mean face to its reflection; mean mismatch 0.02 eye units). It is used
for horizontal-flip augmentation and for the asymmetry features in the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import VideoSegmentDataset
from .faces import AlignConfig, landmarks_to_aligned
from .splits import VideoInfo

# MediaPipe face-mesh indices: union of FACE_LANDMARKS_{LIPS, LEFT/RIGHT_EYE, LEFT/RIGHT_EYEBROW, FACE_OVAL, NOSE}
LANDMARK_IDX = [
    0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 19, 21, 33, 37, 39, 40, 45, 46, 48, 52, 53, 54, 55, 58, 61, 63, 64, 65, 66,
    67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 98, 103, 105, 107, 109, 115, 127, 132, 133, 136, 144, 145,
    146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191,
    195, 197, 220, 234, 246, 249, 251, 263, 267, 269, 270, 275, 276, 278, 282, 283, 284, 285, 288, 291, 293, 294, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332, 334, 336, 338, 344, 356, 361, 362,
    365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
    415, 440, 454, 466,
]
# positions within LANDMARK_IDX (not mesh indices); MIRROR[MIRROR[i]] == i
MIRROR = [
    0, 1, 2, 3, 4, 5, 84, 7, 8, 9, 10, 11, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 41, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 60, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 71, 142, 143, 144, 145,
    146, 147, 148, 79, 80, 149, 150, 151, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83,
]
NUM_POINTS = len(LANDMARK_IDX)


def to_eye_units(aligned_px: np.ndarray, cfg: AlignConfig = AlignConfig()) -> np.ndarray:
    """(..., N, 2) aligned-frame pixel coords -> eye units (origin between the eyes, 1 = eye distance)."""
    origin = np.array([cfg.size / 2, cfg.eye_y * cfg.size], dtype=np.float32)
    return ((aligned_px - origin) / (cfg.eye_dist * cfg.size)).astype(np.float32)


def frame_features(landmarks: np.ndarray, matrix: np.ndarray, width: int, height: int, cfg: AlignConfig = AlignConfig()) -> np.ndarray:
    """One frame: (478, 3) normalised source landmarks + (2, 3) alignment matrix -> (NUM_POINTS, 2) eye units.
    Used by the live pipeline; identical to what load_video_landmarks does per frame."""
    a = landmarks_to_aligned(landmarks[None, LANDMARK_IDX].astype(np.float32), matrix[None], width, height)[0]
    return to_eye_units(a, cfg)


def load_video_landmarks(npz_path: Path, cfg: AlignConfig = AlignConfig()) -> np.ndarray:
    """preprocess_faces.py output -> (T, NUM_POINTS, 2) float32 eye units; frames without a face are
    linearly interpolated from their neighbours (edge-held at the ends)."""
    z = np.load(npz_path)
    w, h = (int(v) for v in z["src_size"])
    lms = z["landmarks"][:, LANDMARK_IDX].astype(np.float32)
    pts = to_eye_units(landmarks_to_aligned(lms, z["transforms"], w, h), cfg)
    found = z["found"] & np.isfinite(pts).all(axis=(1, 2))
    if not found.any():
        raise ValueError(f"no face in {npz_path}")
    if not found.all():
        t = np.arange(len(pts))
        flat = pts.reshape(len(pts), -1)
        for d in range(flat.shape[1]):
            flat[:, d] = np.interp(t, t[found], flat[found, d])
        pts = flat.reshape(pts.shape)
    return pts


def augment(x: np.ndarray, rng: np.random.Generator, hflip: bool = True, rot_deg: float = 5.0, scale: float = 0.05,
            shift: float = 0.03, noise: float = 0.005) -> np.ndarray:
    """(T, N, 2) eye units. Same random geometry for every frame of the segment (like the video
    transforms), plus small independent per-point jitter (landmark detector noise)."""
    d = x.shape[-1]
    if hflip and rng.random() < 0.5:  # palsy can be on either side
        x = x[:, MIRROR] * np.array([-1.0] + [1.0] * (d - 1), dtype=np.float32)
    s = rng.uniform(1 - scale, 1 + scale)
    if d == 2:
        a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    else:  # 3D: small residual yaw / pitch / roll (frontalisation from estimated depth is not perfect)
        r = np.eye(3, dtype=np.float32)
        for ax in range(3):
            a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
            c, sn = np.cos(a), np.sin(a)
            i, j = [k for k in range(3) if k != ax]
            m = np.eye(3, dtype=np.float32)
            m[i, i], m[i, j], m[j, i], m[j, j] = c, -sn, sn, c
            r = r @ m
    shift_v = np.zeros(d, dtype=np.float32)
    shift_v[:2] = rng.uniform(-shift, shift, size=2)
    x = (x @ (s * r).T) + shift_v
    if noise:
        x = x + rng.normal(0, noise, size=x.shape).astype(np.float32)
    return x.astype(np.float32)


class LandmarkSegmentDataset(VideoSegmentDataset):
    """Same segments (timing, sampling, per-video counts) as VideoSegmentDataset, but returns
    {"video": (T, NUM_POINTS, 2) float32 eye units, ...} read from the landmark files."""

    def __init__(self, videos: list[VideoInfo], landmarks_root: Path, train: bool, aug: dict | None = None,
                 mode: str = "2d", smooth: int = 5, seg_counts: list[int] | None = None, slow_prob: float = 0.0,
                 turn_prob: float = 0.0, turn_deg: float = 12.0, seq_len: int = 1, seq_hop: float = 1.0, **kw):
        super().__init__(videos, landmarks_root, train, **kw)
        self.aug = aug or {}
        # seq_len > 1: each item is the window plus up to seq_len - 1 earlier windows, seq_hop seconds apart
        # (for LandmarkSeqGRU), shape (seq_len, T, N, D), left-padded with NaN where there is no history yet
        self.seq_len, self.seq_hop = seq_len, seq_hop
        self.slow_prob = slow_prob
        # "turned head" oversampling (3D, training only): with probability turn_prob a window is placed over a
        # moment where the head is turned > turn_deg, if the video has one. Done for both classes, so the
        # model sees more turned heads (where MediaPipe guesses the far side) without turning becoming a label cue.
        self.turn_prob = turn_prob if (train and mode == "3d") else 0.0
        self.turned_frames: list[np.ndarray] = []
        # optional per-video number of segments (training only), e.g. to balance classes
        counts = seg_counts or [self.segments_per_video] * len(videos)
        if seq_len > 1 and not train:  # evaluation: no more segments than distinct positions >= 0.5 s apart
            counts = [min(c, max(1, (v.num_frames - round(self.segment_seconds * v.fps)) // max(1, round(0.5 * v.fps)) + 1))
                      for c, v in zip(counts, videos)]
        self._counts = counts
        self._index = [(vi, si) for vi, c in enumerate(counts) for si in range(c)]
        paths = [Path(landmarks_root) / Path(v.path).with_suffix(".npz") for v in videos]
        if mode == "3d" and self.turn_prob:
            loaded = [load_video_landmarks_3d(p, smooth, return_yaw=True) for p in paths]
            self.cache = [a for a, _ in loaded]
            self.turned_frames = [np.flatnonzero(y > turn_deg) for _, y in loaded]
            n_vid = sum(len(t) > 0 for t in self.turned_frames)
            n_fr = sum(len(t) for t in self.turned_frames)
            print(f"turned-head oversampling: {n_vid}/{len(paths)} videos have frames with |yaw| > {turn_deg:.0f} deg "
                  f"({n_fr} frames); affected {sum(len(t) > 0 for t, v in zip(self.turned_frames, videos) if v.label == 1)}, "
                  f"unaffected {sum(len(t) > 0 for t, v in zip(self.turned_frames, videos) if v.label == 0)}")
        elif mode == "3d":
            self.cache = [load_video_landmarks_3d(p, smooth) for p in paths]
        else:
            self.cache = [load_video_landmarks(p) for p in paths]
        for v, arr in zip(videos, self.cache):
            if len(arr) != v.num_frames:  # transforms/landmarks are per decoded frame; keep frame_indices in range
                v.num_frames = min(v.num_frames, len(arr))
        if seq_len > 1 and train:
            # History length must not give the label away: short clips (e.g. 3-4 s RAVDESS, all healthy) can
            # only supply 1-2 windows of history. Affected samples are therefore capped like a random short
            # healthy clip with the same probability that a healthy sample comes from one.
            n_win = [self._n_windows(i) for i in range(len(videos))]
            short = [(n, c) for n, c, v in zip(n_win, counts, videos) if v.label == 0 and n < seq_len]
            healthy = sum(c for c, v in zip(counts, videos) if v.label == 0)
            self._short_caps = np.repeat([n for n, _ in short], [c for _, c in short]) if short else np.zeros(0, int)
            self._short_share = len(self._short_caps) / max(healthy, 1)
            print(f"sequence history: {self._short_share:.0%} of healthy samples come from clips with < {seq_len} "
                  f"windows; affected samples are capped to match")

    def __len__(self) -> int:
        return len(self._index)

    def _n_windows(self, vid_idx: int) -> int:
        v = self.videos[vid_idx]
        n = len(self.cache[vid_idx])
        span = min(round(self.segment_seconds * v.fps), n)
        return (n - span) // max(1, round(self.seq_hop * v.fps)) + 1

    def _seq_item(self, index: int):
        vid_idx, seg_idx = self._index[index]
        v, arr = self.videos[vid_idx], self.cache[vid_idx]
        n = len(arr)
        span = min(round(self.segment_seconds * v.fps), n)
        hop = max(1, round(self.seq_hop * v.fps))
        max_start = n - span
        rng = np.random.default_rng((self.seed, self.epoch, index, 3))
        if self.train:
            # history length first (the same distribution for both classes): the full seq_len half the time,
            # otherwise uniform 1..seq_len (live warm-up); then a position that has that much history
            k = self.seq_len if rng.random() < 0.5 else int(rng.integers(1, self.seq_len + 1))
            if v.label == 1 and len(self._short_caps) and rng.random() < self._short_share:
                k = min(k, int(rng.choice(self._short_caps)))
            k = min(k, max_start // hop + 1)
            last = int(rng.integers((k - 1) * hop, max_start + 1))
        else:
            last = max_start if self._counts[vid_idx] == 1 else round(seg_idx * max_start / (self._counts[vid_idx] - 1))
            k = min(self.seq_len, last // hop + 1)  # early in a video there is less history (like live warm-up)
        starts = last - hop * np.arange(k - 1, -1, -1)  # oldest window first
        pos = starts[:, None] + np.linspace(0, span - 1, self.num_frames)[None]  # (k, T) source frame positions
        if self.train and self.slow_prob and rng.random() < self.slow_prob:
            # still-face augmentation for the whole sequence (someone sitting still for several seconds)
            f = rng.uniform(0, 0.5) if rng.random() < 0.7 else 0.0
            pos = pos[0, 0] + (pos - pos[0, 0]) * f
            lo = np.floor(pos).astype(int).clip(0, n - 1)
            hi = np.minimum(lo + 1, n - 1)
            w = (pos - lo).astype(np.float32)[..., None, None]
            x = arr[lo] * (1 - w) + arr[hi] * w
        else:
            x = arr[np.clip(pos.round().astype(int), 0, n - 1)]
        k_, t_ = x.shape[:2]
        if self.train:  # one random geometry for the whole sequence (same flip / pose), per-frame jitter
            x = augment(x.reshape(k_ * t_, *x.shape[2:]), np.random.default_rng((self.seed, self.epoch, index, 1)),
                        **self.aug).reshape(x.shape)
        out = np.full((self.seq_len, *x.shape[1:]), np.nan, dtype=np.float32)
        out[self.seq_len - k_:] = x
        return {
            "video": torch.from_numpy(out),
            "label": torch.tensor(v.label),
            "video_idx": torch.tensor(vid_idx),
            "start_sec": torch.tensor(last / v.fps, dtype=torch.float32),
        }

    def __getitem__(self, index: int):
        if self.seq_len > 1:
            return self._seq_item(index)
        vid_idx, seg_idx = self._index[index]
        v = self.videos[vid_idx]
        idx, start = self.frame_indices(v, seg_idx, index)
        arr = self.cache[vid_idx]
        rng = np.random.default_rng((self.seed, self.epoch, index, 2))
        if self.turn_prob and len(self.turned_frames[vid_idx]) and rng.random() < self.turn_prob:
            span = int(idx[-1] - idx[0])
            c = int(rng.choice(self.turned_frames[vid_idx]))
            s0 = int(np.clip(c - rng.integers(0, span + 1), 0, max(len(arr) - 1 - span, 0)))
            idx, start = idx - idx[0] + s0, s0
        if self.train and self.slow_prob and rng.random() < self.slow_prob:
            # "still face" augmentation: play the window at 0-50% speed (0 = one frozen frame), same label.
            # Healthy training clips are all people talking, so without this the model learns
            # "face held still" = affected. Applied to both classes, stillness stops predicting the label.
            f = rng.uniform(0, 0.5) if rng.random() < 0.7 else 0.0
            pos = np.clip(idx[0] + (np.asarray(idx) - idx[0]) * f, 0, len(arr) - 1)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, len(arr) - 1)
            w = (pos - lo).astype(np.float32)[:, None, None]
            x = arr[lo] * (1 - w) + arr[hi] * w
        else:
            x = arr[np.clip(idx, 0, len(arr) - 1)]
        if self.train:
            x = augment(x, np.random.default_rng((self.seed, self.epoch, index, 1)), **self.aug)
        return {
            "video": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.tensor(v.label),
            "video_idx": torch.tensor(vid_idx),
            "start_sec": torch.tensor(start / v.fps, dtype=torch.float32),
        }


def build_landmark_loaders(cfg: dict, train_v, val_v, test_v):
    d = cfg["data"]
    root = Path(d.get("landmarks_root", "data/aligned/landmarks"))
    common = dict(segment_seconds=d["segment_seconds"], sample_fps=d["sample_fps"],
                  mode=d.get("landmark_mode", "2d"), smooth=d.get("landmark_smooth", 5),
                  seq_len=d.get("seq_len", 1), seq_hop=d.get("seq_hop", 1.0))
    aug = dict(hflip=d.get("hflip", True), **d.get("augment", {}))
    k = d["train_segments_per_video"]
    counts = None
    if d.get("extra_unaffected"):
        # Extra "unaffected" videos from another source (e.g. VCD webcam clips, see scripts/fetch_vcd.py).
        # Their own folds: test-fold clips stay out (scored separately by scripts/eval_webcam.py),
        # val-fold clips join validation, the rest join training.
        n_folds = d.get("n_folds", 5)
        test_f, val_f = cfg["fold"], (cfg["fold"] + 1) % n_folds
        # one videos.json or a list of them (e.g. VCD webcam talking + RAVDESS expressions)
        sources = d["extra_unaffected"]
        sources = [sources] if isinstance(sources, str) else list(sources)
        n_aff = sum(v.label == 1 for v in train_v)
        n_una = sum(v.label == 0 for v in train_v)
        per_orig = max(1, round(n_aff * k / 2 / max(n_una, 1)))
        counts = [k if v.label == 1 else per_orig for v in train_v]
        val_v, train_v, msg = list(val_v), list(train_v), []
        for src in sources:
            extra = [VideoInfo(**r) for r in json.load(open(src))]
            ex_train = [v for v in extra if v.fold not in (test_f, val_f)]
            ex_val = [v for v in extra if v.fold == val_f]
            # equal training segments per class: affected videos keep k each; the unaffected budget
            # (n_affected * k) is split half to the original unaffected videos, half to the extras
            # (shared equally between sources). Short clips (< 1 segment's worth) still get 1.
            per_extra = max(1, round(n_aff * k / 2 / len(sources) / max(len(ex_train), 1)))
            counts += [per_extra] * len(ex_train)
            train_v += ex_train
            val_v += ex_val
            msg.append(f"{Path(src).parent.name} {len(ex_train)} train / {len(ex_val)} val clips x{per_extra}")
        if not d.get("balance_classes", True):
            counts = None
        print(f"extra unaffected: {'; '.join(msg)}; affected x{k}, unaffected x{per_orig}")
    train_ds = LandmarkSegmentDataset(train_v, root, True, aug=aug, segments_per_video=k, seg_counts=counts, seed=cfg["seed"],
                                      slow_prob=d.get("slow_prob", 0.0), turn_prob=d.get("turn_prob", 0.0),
                                      turn_deg=d.get("turn_deg", 12.0), **common)
    val_ds = LandmarkSegmentDataset(val_v, root, False, segments_per_video=d["eval_segments_per_video"], **common)
    test_ds = LandmarkSegmentDataset(test_v, root, False, segments_per_video=d["eval_segments_per_video"], **common)
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(batch_size=d["batch_size"], num_workers=0)  # everything is in memory already
    return (
        DataLoader(train_ds, shuffle=True, generator=g, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ----------------------------------------------------------------------------- 3D: smoothing + frontalisation
# The 2D path above aligns in the image plane only (shift, scale, roll), so a head turned left/right
# or nodding still makes one half of the face look narrower, i.e. asymmetric. The 3D path uses
# MediaPipe's estimated depth (z) to rotate every frame back to a frontal view first.
#
# Head pose comes only from rigid parts of the face that expressions and palsy barely move:
# inner/outer eye corners, the face outline at ear level, and the forehead/nose-bridge midline.
# Mouth, cheeks and brows are left out, so a real droop cannot be "rotated away". It is a pure
# rotation + uniform scale (no per-side stretching), so real asymmetry is preserved.
#
# Before that, the raw landmark trajectories are smoothed with a centred moving average over
# `smooth` frames (at the video's 30 fps). This removes detector jitter, whose size depends on the
# camera (resolution, sharpness, compression) and which the motion features would otherwise read
# as movement. Live, the same centred window costs a delay of smooth // 2 frames.

POSE_PAIRS = [(33, 263), (133, 362), (234, 454), (127, 356)]  # (image-left, image-right): eye corners, outline at ear level
UPPER_MID = [10, 151, 9, 8]       # forehead midline
LOWER_MID = [168, 6, 197, 195]    # nose bridge (not the tip, which sticks out by a person-specific amount)
EYE_L, EYE_R = (33, 133), (362, 263)
KEEP_3D = sorted(set(LANDMARK_IDX) | {p for pr in POSE_PAIRS for p in pr} | set(UPPER_MID) | set(LOWER_MID))
_POS = {m: i for i, m in enumerate(KEEP_3D)}
_OUT = [_POS[m] for m in LANDMARK_IDX]


def to_px3(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(..., 478, 3) MediaPipe normalised landmarks -> (..., len(KEEP_3D), 3) pixel-scaled 3D points.
    MediaPipe's z uses roughly the same scale as x (image width)."""
    return (landmarks[..., KEEP_3D, :].astype(np.float32) * np.array([width, height, width], dtype=np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frontalize(P: np.ndarray) -> np.ndarray:
    """(..., len(KEEP_3D), 3) pixel-scaled points -> (..., NUM_POINTS, 3) in a head-fixed frame:
    origin between the eye centres, x along the eye line (image-left -> image-right), y down the
    midline, z = depth; 1 unit = 3D distance between the eye centres."""
    g = lambda ids: P[..., [_POS[i] for i in ids], :]
    xv = sum(g([b])[..., 0, :] - g([a])[..., 0, :] for a, b in POSE_PAIRS)
    x_hat = _unit(xv)
    down = g(LOWER_MID).mean(-2) - g(UPPER_MID).mean(-2)
    y_hat = _unit(down - (down * x_hat).sum(-1, keepdims=True) * x_hat)
    z_hat = np.cross(x_hat, y_hat)
    R = np.stack([x_hat, y_hat, z_hat], axis=-1)  # columns = head axes in camera coords
    eye_l, eye_r = g(EYE_L).mean(-2), g(EYE_R).mean(-2)
    origin = (eye_l + eye_r) / 2
    scale = np.linalg.norm(eye_r - eye_l, axis=-1)[..., None, None]
    q = np.einsum("...kc,...cd->...kd", P - origin[..., None, :], R) / np.maximum(scale, 1e-9)
    return q[..., _OUT, :].astype(np.float32)


def head_pose(P: np.ndarray) -> np.ndarray:
    """(..., len(KEEP_3D), 3) pixel-scaled points -> (..., 3) head yaw, pitch, roll in degrees, from the
    same head frame frontalize() uses (0, 0, 0 = facing the camera). Pitch has a constant offset
    because the midline axis follows the forehead / nose bridge; compare pitch values relative to each other."""
    g = lambda ids: P[..., [_POS[i] for i in ids], :]
    x_hat = _unit(sum(g([b])[..., 0, :] - g([a])[..., 0, :] for a, b in POSE_PAIRS))
    down = g(LOWER_MID).mean(-2) - g(UPPER_MID).mean(-2)
    y_hat = _unit(down - (down * x_hat).sum(-1, keepdims=True) * x_hat)
    yaw = np.degrees(np.arctan2(x_hat[..., 2], x_hat[..., 0]))
    pitch = np.degrees(np.arctan2(y_hat[..., 2], y_hat[..., 1]))
    roll = np.degrees(np.arctan2(x_hat[..., 1], x_hat[..., 0]))
    return np.stack([yaw, pitch, roll], -1).astype(np.float32)


def load_video_landmarks_3d(npz_path: Path, smooth: int = 5, return_yaw: bool = False):
    """Offline 3D path: (T, NUM_POINTS, 3). Missed detections are linearly interpolated, then every
    coordinate is smoothed with a centred `smooth`-frame moving average (edge-padded), then frontalised."""
    from .faces import _fill_and_smooth

    z = np.load(npz_path)
    w, h = (int(v) for v in z["src_size"])
    P = to_px3(z["landmarks"], w, h)
    found = z["found"] & np.isfinite(P).all(axis=(1, 2))
    if not found.any():
        raise ValueError(f"no face in {npz_path}")
    T, K, _ = P.shape
    P = _fill_and_smooth(np.nan_to_num(P.reshape(T, K * 3)), found, smooth).reshape(T, K, 3)
    if return_yaw:  # also (T,) |yaw| in degrees of the smoothed head
        return frontalize(P), np.abs(head_pose(P)[:, 0])
    return frontalize(P)


class LandmarkFrame:
    """One output frame of StreamingLandmarkSmoother (same role as faces.AlignedFrame)."""

    __slots__ = ("index", "timestamp", "frame", "landmarks", "features", "aligned", "matrix", "pose")

    def __init__(self, index, timestamp, frame, landmarks, features, pose=None):
        self.index, self.timestamp, self.frame, self.landmarks, self.features = index, timestamp, frame, landmarks, features
        self.pose = pose  # (yaw, pitch, roll) degrees of the smoothed head, see head_pose()
        self.aligned = self.matrix = None


class StreamingLandmarkSmoother:
    """Live / causal version of load_video_landmarks_3d, built like faces.StreamingFaceAligner:
    a frame is emitted once the smooth // 2 frames after it have arrived, with the same centred
    moving average (edge padding at track ends) and the same frontalisation.

    Differences from offline, only around missed detections: a gap of <= max_gap frames holds
    the last landmarks (offline interpolates); a longer gap ends the track (features=None) until
    the face is found again."""

    def __init__(self, smooth: int, width: int, height: int, max_gap: int = 15):
        self.width, self.height, self.max_gap = width, height, max_gap
        self.pad = smooth // 2 if smooth > 1 else 0
        self._index = 0
        self._reset()

    def _reset(self):
        self._vals, self._base, self._n, self._pending = [], 0, 0, []
        self._last, self._gap = None, 0

    def _value(self, pos):
        pos = min(max(pos, 0), self._n - 1)
        return self._vals[pos - self._base]

    def _emit(self, final: bool):
        out = []
        while self._pending and (final or self._pending[0][0] + self.pad <= self._n - 1):
            pos, idx, ts, frame, lms = self._pending.pop(0)
            P = np.mean([self._value(p) for p in range(pos - self.pad, pos + self.pad + 1)], axis=0)
            out.append(LandmarkFrame(idx, ts, frame, lms, frontalize(P), head_pose(P)))
        if self._pending:
            keep = max(0, self._pending[0][0] - self.pad)
            if keep > self._base:
                del self._vals[: keep - self._base]
                self._base = keep
        return out

    def push(self, frame, landmarks, timestamp: float = 0.0):
        idx, self._index = self._index, self._index + 1
        out = []
        if landmarks is not None:
            self._last, self._gap = to_px3(landmarks, self.width, self.height).astype(np.float64), 0
        else:
            self._gap += 1
            if self._last is None or self._gap > self.max_gap:
                if self._last is not None:
                    out += self._emit(final=True)
                    self._reset()
                out.append(LandmarkFrame(idx, timestamp, frame, None, None))
                return out
        self._vals.append(self._last)
        self._pending.append((self._n, idx, timestamp, frame, landmarks))
        self._n += 1
        return out + self._emit(final=False)

    def flush(self):
        out = self._emit(final=True)
        self._reset()
        return out
