"""Video source -> MediaPipe face landmarks -> per-frame asymmetry features.

Run on its own to see the overlay and live feature values:
    python -m strokewatch.vision --source 0
    python -m strokewatch.vision --source 0 --record clips/normal.mp4
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

from . import ROOT, load_config

# MediaPipe face-mesh indices. L/R are the PATIENT's left/right
# (an unmirrored webcam shows the patient's left on the right of the image).
MOUTH_R, MOUTH_L = 61, 291
EYE_OUT_R, EYE_OUT_L = 33, 263
EYE_IN_R, EYE_IN_L = 133, 362
EYE_LIDS_R = [(160, 144), (159, 145), (158, 153)]  # (upper, lower) pairs
EYE_LIDS_L = [(385, 380), (386, 374), (387, 373)]
LIP_IN_TOP, LIP_IN_BOT = 13, 14
BROW_R = [70, 63, 105, 66, 107]
BROW_L = [300, 293, 334, 296, 336]
KEY_POINTS = [MOUTH_R, MOUTH_L, EYE_OUT_R, EYE_OUT_L, EYE_IN_R, EYE_IN_L, *BROW_R, *BROW_L]

# OpenCV camera frame (x right, y down, z forward) <-> OpenGL frame used by the
# MediaPipe pose matrix (x right, y up, z toward viewer).
_FLIP = np.diag([1.0, -1.0, -1.0])


@dataclass
class FrameFeatures:
    """Sign convention: positive mouth_diff / eye_ratio / brow_diff = patient's LEFT side lower/weaker."""
    t: float
    mouth_diff: float | None = None  # (left corner y - right corner y) / eye distance
    eye_ratio: float | None = None   # log(right eye opening / left eye opening)
    brow_diff: float | None = None   # (right brow height - left brow height) / eye distance
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    jaw_open: float | None = None
    lip_gap: float | None = None
    valid: bool = False
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def rotation_from_matrix(m: np.ndarray) -> np.ndarray:
    """Pure rotation (OpenGL frame) from MediaPipe's 4x4 facial transformation matrix."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=float)[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = -r
    return r


def euler_deg(r_gl: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees. 0,0,0 = facing the camera upright."""
    fwd = r_gl[:, 2]  # face's forward axis in camera space
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    pitch = math.degrees(math.atan2(fwd[1], fwd[2]))
    roll = math.degrees(math.atan2(r_gl[1, 0], r_gl[0, 0]))
    return yaw, pitch, roll


def compute_features(landmarks, blendshapes, matrix, w: int, h: int, t: float, vc: dict
                     ) -> FrameFeatures:
    """Features from one face's landmarks. Pure function so it can be unit-tested."""
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in landmarks], dtype=float)
    r_gl = rotation_from_matrix(matrix)
    yaw, pitch, roll = euler_deg(r_gl)
    ff = FrameFeatures(t=t, yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(roll, 1))

    bs = {c.category_name: c.score for c in blendshapes} if blendshapes else {}
    ff.jaw_open = round(bs.get("jawOpen", 0.0), 3)
    if abs(yaw) > vc["max_yaw_deg"]:
        ff.reason = "head turned"
        return ff
    if abs(pitch) > vc["max_pitch_deg"] or abs(roll) > vc["max_roll_deg"]:
        ff.reason = "head tilted"
        return ff

    # Undo head rotation, y down.
    # "2d" (default): only level the line between the outer eye corners (removes roll).
    #   On one recorded head sweep (Nov 13, 2026) this halved the turn-induced error vs
    #   "3d", because MediaPipe's depth estimates leak into y. Remaining turn/nod bias is
    #   handled by the pose-matched baseline in detector.py.
    # "3d": full rotation removal using the facial transformation matrix.
    if vc.get("correction", "2d") == "3d":
        r_cv = _FLIP @ r_gl @ _FLIP
        q = (pts - pts.mean(axis=0)) @ r_cv
    else:
        p2 = pts[:, :2] - pts[:, :2].mean(axis=0)
        v = p2[EYE_OUT_L] - p2[EYE_OUT_R]
        a = math.atan2(v[1], v[0])
        c, s = math.cos(a), math.sin(a)
        q = p2 @ np.array([[c, -s], [s, c]])  # rotate by -a: eye line becomes horizontal
    iod = float(np.linalg.norm(q[EYE_OUT_L] - q[EYE_OUT_R]))
    if iod < 1e-6:
        ff.reason = "no face"
        return ff
    y = q[:, 1] / iod  # normalized, y down (bigger = lower on the face)

    # Talking gate: inner-lip gap at the mouth CENTER (a droop moves the corner, not
    # the center), plus jawOpen as a loose backstop. jawOpen alone rises with a droop.
    ff.lip_gap = round(float(y[LIP_IN_BOT] - y[LIP_IN_TOP]), 4)
    if ff.lip_gap > vc["lip_gap_max"] or bs.get("jawOpen", 0.0) > vc["jaw_open_max"]:
        ff.reason = "talking / mouth open"
        return ff

    ff.mouth_diff = float(y[MOUTH_L] - y[MOUTH_R])

    eye_c_r = (y[EYE_OUT_R] + y[EYE_IN_R]) / 2
    eye_c_l = (y[EYE_OUT_L] + y[EYE_IN_L]) / 2
    brow_h_r = eye_c_r - y[BROW_R].mean()
    brow_h_l = eye_c_l - y[BROW_L].mean()
    ff.brow_diff = float(brow_h_r - brow_h_l)

    blinking = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) > vc["blink_max"]
    if blinking:
        ff.reason = "blinking (eye feature skipped)"
    else:
        open_r = np.mean([y[b] - y[a] for a, b in EYE_LIDS_R])
        open_l = np.mean([y[b] - y[a] for a, b in EYE_LIDS_L])
        if open_r > 1e-4 and open_l > 1e-4:
            ff.eye_ratio = float(math.log(open_r / open_l))
    ff.valid = True
    return ff


class FaceAnalyzer:
    def __init__(self, cfg: dict):
        self.vc = cfg["vision"]
        opts = mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ROOT / self.vc["model_path"])),
            running_mode=mpv.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mpv.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[FrameFeatures, np.ndarray | None]:
        """Returns features and the landmark pixel coords (N x 2) for drawing, or None."""
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        res = self.landmarker.detect_for_video(img, ms)
        if not res.face_landmarks:
            return FrameFeatures(t=t, reason="no face"), None
        lms = res.face_landmarks[0]
        ff = compute_features(lms, res.face_blendshapes[0] if res.face_blendshapes else None,
                              res.facial_transformation_matrixes[0], w, h, t, self.vc)
        px = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
        return ff, px

    def close(self):
        self.landmarker.close()


def draw_overlay(frame: np.ndarray, px: np.ndarray | None, ff: FrameFeatures,
                 status: dict | None = None) -> np.ndarray:
    """Draw landmarks, symmetry lines and a short text readout onto frame (in place)."""
    draw_landmarks(frame, px, ff)
    return draw_readout(frame, ff, status)


def draw_landmarks(frame: np.ndarray, px: np.ndarray | None, ff: FrameFeatures) -> np.ndarray:
    """Landmarks + symmetry lines only (in place); green = frame used, red = skipped."""
    ok_col = (90, 220, 90) if ff.valid else (60, 60, 230)
    if px is not None:
        for x, y in px[::3]:
            cv2.circle(frame, (int(x), int(y)), 1, (150, 150, 150), -1, cv2.LINE_AA)
        for i in KEY_POINTS:
            cv2.circle(frame, tuple(int(v) for v in px[i]), 3, ok_col, -1, cv2.LINE_AA)
        for a, b in [(MOUTH_R, MOUTH_L), (EYE_OUT_R, EYE_OUT_L)]:
            cv2.line(frame, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]),
                     ok_col, 2, cv2.LINE_AA)
    return frame


def draw_readout(frame: np.ndarray, ff: FrameFeatures, status: dict | None = None) -> np.ndarray:
    """Short text readout of pose/features/skip reason in the top-left corner (in place)."""
    lines = []
    if status:
        lines.append(f"{status['state']}  score {status['score']:.1f}/{status['threshold']:.1f}")
    if ff.yaw is not None:
        lines.append(f"yaw {ff.yaw:+.0f} pitch {ff.pitch:+.0f} roll {ff.roll:+.0f}")
    if ff.mouth_diff is not None:
        eye = f"{ff.eye_ratio:+.3f}" if ff.eye_ratio is not None else "  -  "
        lines.append(f"mouth {ff.mouth_diff:+.3f} eye {eye} brow {ff.brow_diff:+.3f}")
    if ff.reason:
        lines.append(f"skip: {ff.reason}" if not ff.valid else ff.reason)
    for i, text in enumerate(lines):
        x, y = 10, 22 + 22 * i
        # Outline = thin black copies offset by 1 px. (A thick stroke changes glyph spacing in
        # this OpenCV version, so it drifts out from under the white text.)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 1), (1, 0)):
            cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                        cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


_ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(frame: np.ndarray, degrees_cw: int) -> np.ndarray:
    """Rotate by the camera mount angle (0/90/180/270 clockwise) so the face is upright."""
    code = _ROTATE_CODES[int(degrees_cw) % 360]
    return frame if code is None else cv2.rotate(frame, code)


class VideoSource:
    """Webcam index ("0") or video file.

    clock="wall": t is time.time() (live use; onset shows a real clock time).
    clock="video": t is the file's own timeline (replay).
    realtime/loop only apply to files. rotate = camera mount angle, clockwise degrees.
    """

    def __init__(self, source: str | int, width: int | None = None, height: int | None = None,
                 clock: str = "wall", realtime: bool = False, loop: bool = False,
                 rotate: int = 0):
        self.source = str(source)
        self.is_file = not self.source.isdigit()
        self.clock, self.realtime, self.loop = clock, realtime, loop
        if int(rotate) % 360 not in _ROTATE_CODES:
            raise ValueError(f"camera.rotate must be 0, 90, 180 or 270, got {rotate}")
        self.rotate = int(rotate) % 360
        self.width, self.height = width, height
        self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1 <= self.fps <= 240):
            self.fps = 30.0
        self._idx = 0
        self._next_wall = time.monotonic()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source if self.is_file else int(self.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {self.source!r} "
                               "(on macOS, grant your terminal camera access)")
        if not self.is_file and self.width and self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def reopen(self):
        """Release and reopen (e.g. after a USB camera dropped out). Raises if it can't open."""
        self.cap.release()
        self._open()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.release()
            self._open()
            ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        if self.is_file and self.realtime:
            self._next_wall += 1.0 / self.fps
            delay = self._next_wall - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_wall = time.monotonic()
        t = self._idx / self.fps if (self.is_file and self.clock == "video") else time.time()
        self._idx += 1
        return True, rotate_frame(frame, self.rotate), t

    def release(self):
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description="Show landmarks + features live.")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--record", default=None, help="also save the raw video to this .mp4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = VideoSource(args.source, cfg["camera"]["width"], cfg["camera"]["height"],
                      realtime=True, rotate=cfg["camera"].get("rotate", 0))
    an = FaceAnalyzer(cfg)
    writer = None
    print("q / Esc to quit")
    while True:
        ok, frame, t = src.read()
        if not ok:
            break
        if args.record:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src.fps, (w, h))
            writer.write(frame)
        ff, px = an.process(frame, t)
        cv2.imshow("strokewatch.vision", draw_overlay(frame, px, ff))
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    if writer:
        writer.release()
    src.release()
    an.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
