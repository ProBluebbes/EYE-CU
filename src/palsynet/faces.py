"""Facial landmarks (MediaPipe Face Landmarker, 478 points) and face normalisation.

Normalisation is a *similarity* transform (rotation + uniform scale + translation) that puts
the two eye centres on fixed positions in a square output frame. It deliberately does not
warp the face to a template (no affine shear / per-axis scale, no piecewise warps): that
would partially "correct" the left/right asymmetry the classifier is supposed to detect.

Eye centres are the mean of each eye's inner and outer canthus (not the iris, which moves
with gaze). Their trajectories are gap-filled and temporally smoothed before computing the
transform, so the aligned video does not jitter with per-frame landmark noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# MediaPipe face-mesh indices. "Image-left" = the eye on the left of the image (subject's right).
IMG_LEFT_EYE = (33, 133)    # outer, inner canthus
IMG_RIGHT_EYE = (362, 263)  # inner, outer canthus
NOSE_TIP = 1


@dataclass
class AlignConfig:
    size: int = 224             # output frame is size x size
    eye_y: float = 0.38         # eye-centre row, as a fraction of size
    eye_dist: float = 0.27      # distance between eye centres, as a fraction of size
    smooth_window: int = 7      # centred moving-average window (frames) on eye trajectories; 1 = off


def detect_landmarks(frames_rgb, model_path: str | Path, fps: float):
    """Run the landmarker in VIDEO mode. Returns (T, 478, 3) float32 normalised coords (NaN where
    no face was found) and a (T,) bool detection mask."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    out, found = [], []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames_rgb):
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)), round(i * 1000 / fps))
            if res.face_landmarks:
                out.append(np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32))
                found.append(True)
            else:
                out.append(np.full((478, 3), np.nan, dtype=np.float32))
                found.append(False)
    return np.stack(out), np.array(found)


def _fill_and_smooth(traj: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """traj (T, D): linearly interpolate invalid frames (edge-hold at the ends), then moving-average."""
    t = np.arange(len(traj))
    out = np.empty_like(traj)
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(t, t[valid], traj[valid, d])
    if window > 1 and len(out) >= window:
        k = np.ones(window) / window
        pad = window // 2
        padded = np.pad(out, ((pad, window - 1 - pad), (0, 0)), mode="edge")
        out = np.stack([np.convolve(padded[:, d], k, mode="valid") for d in range(out.shape[1])], 1)
    return out


def eye_points(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(..., 478, 3) normalised landmarks -> (..., 4) pixel eye centres [left_x, left_y, right_x, right_y] (image left/right)."""
    px = landmarks[..., :2] * np.array([width, height], dtype=np.float32)
    return np.concatenate([px[..., list(IMG_LEFT_EYE), :].mean(-2), px[..., list(IMG_RIGHT_EYE), :].mean(-2)], -1)


def similarity_transforms(landmarks: np.ndarray, found: np.ndarray, width: int, height: int, cfg: AlignConfig) -> np.ndarray:
    """Offline: (T, 478, 3) normalised landmarks -> (T, 2, 3) similarity matrices (source px -> aligned px).

    Gaps are linearly interpolated using future detections; live use goes through StreamingFaceAligner.
    """
    if not found.any():
        raise ValueError("no face detected in any frame")
    eyes = _fill_and_smooth(eye_points(landmarks, width, height), found, cfg.smooth_window)
    return eyes_to_matrices(eyes, cfg)


def eyes_to_matrices(eyes: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """(T, 4) smoothed eye centres -> (T, 2, 3) similarity matrices."""
    eye_l, eye_r = eyes[:, :2], eyes[:, 2:]
    s = cfg.size
    dst_l = np.array([s * (0.5 - cfg.eye_dist / 2), s * cfg.eye_y])
    dst_r = np.array([s * (0.5 + cfg.eye_dist / 2), s * cfg.eye_y])
    src_vec, dst_vec = eye_r - eye_l, dst_r - dst_l
    scale = np.linalg.norm(dst_vec) / np.linalg.norm(src_vec, axis=1)
    angle = np.arctan2(dst_vec[1], dst_vec[0]) - np.arctan2(src_vec[:, 1], src_vec[:, 0])
    a, b = scale * np.cos(angle), scale * np.sin(angle)
    m = np.zeros((len(eyes), 2, 3), dtype=np.float32)
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = a, -b, b, a
    # translation: map the source eye midpoint onto the destination eye midpoint
    src_mid, dst_mid = (eye_l + eye_r) / 2, (dst_l + dst_r) / 2
    m[:, 0, 2] = dst_mid[0] - (a * src_mid[:, 0] - b * src_mid[:, 1])
    m[:, 1, 2] = dst_mid[1] - (b * src_mid[:, 0] + a * src_mid[:, 1])
    return m


def warp(frame: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def landmarks_to_aligned(landmarks: np.ndarray, m: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map (T, N, 3) normalised source landmarks into aligned-frame pixel coords (T, N, 2)."""
    px = landmarks[..., :2] * np.array([width, height], dtype=np.float32)
    return np.einsum("tij,tnj->tni", m[:, :, :2], px) + m[:, None, :, 2]


# ----------------------------------------------------------------------------- live / streaming
@dataclass
class AlignedFrame:
    index: int                    # running frame number of the input stream
    timestamp: float              # seconds, as passed to push()
    frame: np.ndarray             # the original RGB frame
    landmarks: np.ndarray | None  # (478, 3) normalised, None if no face this frame
    matrix: np.ndarray | None     # (2, 3) source px -> aligned px, None = no usable face
    aligned: np.ndarray | None    # (size, size, 3) RGB aligned face, None = no usable face


class StreamingFaceAligner:
    """Causal version of `similarity_transforms` for live streams.

    Uses the same eye points, smoothing window and similarity transform as the offline
    preprocessing, so the model sees the same kind of input it was trained on. The price is a fixed
    delay of `smooth_window // 2` frames (3 frames = 0.1 s at 30 fps): a frame is emitted
    once the frames after it that the centred moving average needs have arrived.

    Differences from offline, only around missed detections:
      * a gap of <= max_gap frames holds the last detected eye position (offline interpolates);
      * a longer gap ends the track: frames are emitted with matrix=None ("no face") until the
        face is found again, which starts a new track (fresh edge padding, as at a video start).
    """

    def __init__(self, cfg: AlignConfig, width: int, height: int, max_gap: int = 15):
        self.cfg, self.width, self.height, self.max_gap = cfg, width, height, max_gap
        self.pad = cfg.smooth_window // 2 if cfg.smooth_window > 1 else 0
        self.window = 2 * self.pad + 1
        self._index = 0
        self._reset_track()

    @property
    def delay_frames(self) -> int:
        return self.pad

    def _reset_track(self) -> None:
        self._vals: list[np.ndarray] = []  # filled eye values of the current track
        self._base = 0                     # track position of _vals[0]
        self._n = 0                        # track length so far
        self._pending: list[tuple] = []    # (track_pos, index, timestamp, frame, landmarks)
        self._last_eyes: np.ndarray | None = None
        self._gap = 0

    def _value(self, pos: int) -> np.ndarray:
        pos = min(max(pos, 0), self._n - 1)  # edge padding at both ends of the track
        return self._vals[pos - self._base]

    def _emit(self, final: bool) -> list[AlignedFrame]:
        out = []
        while self._pending and (final or self._pending[0][0] + self.pad <= self._n - 1):
            pos, idx, ts, frame, lms = self._pending.pop(0)
            eyes = np.mean([self._value(p) for p in range(pos - self.pad, pos + self.pad + 1)], axis=0)
            m = eyes_to_matrices(eyes[None].astype(np.float64), self.cfg)[0]
            out.append(AlignedFrame(idx, ts, frame, lms, m, warp(frame, m, self.cfg.size)))
        # drop values no pending frame will need any more
        if self._pending:
            keep_from = max(0, self._pending[0][0] - self.pad)
            if keep_from > self._base:
                del self._vals[: keep_from - self._base]
                self._base = keep_from
        return out

    def push(self, frame: np.ndarray, landmarks: np.ndarray | None, timestamp: float = 0.0) -> list[AlignedFrame]:
        """Add one RGB frame and its landmarks (None if no face). Returns the frames now ready, in order."""
        idx, self._index = self._index, self._index + 1
        out: list[AlignedFrame] = []
        if landmarks is not None:
            self._last_eyes, self._gap = eye_points(landmarks, self.width, self.height).astype(np.float64), 0
        else:
            self._gap += 1
            if self._last_eyes is None or self._gap > self.max_gap:
                if self._last_eyes is not None:  # gap too long: close the track
                    out += self._emit(final=True)
                    self._reset_track()
                out.append(AlignedFrame(idx, timestamp, frame, None, None, None))
                return out
        self._vals.append(self._last_eyes)
        self._pending.append((self._n, idx, timestamp, frame, landmarks))
        self._n += 1
        return out + self._emit(final=False)

    def flush(self) -> list[AlignedFrame]:
        """End of stream: emit the frames still waiting for future context."""
        out = self._emit(final=True)
        self._reset_track()
        return out


class LandmarkDetector:
    """Thin wrapper around MediaPipe Face Landmarker (VIDEO mode) returning (478, 3) arrays or None."""

    def __init__(self, model_path: str | Path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def __call__(self, frame_rgb: np.ndarray, timestamp_s: float) -> np.ndarray | None:
        ms = max(round(timestamp_s * 1000), self._last_ms + 1)  # MediaPipe needs strictly increasing timestamps
        self._last_ms = ms
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)), ms)
        if not res.face_landmarks:
            return None
        return np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._lm.close()
