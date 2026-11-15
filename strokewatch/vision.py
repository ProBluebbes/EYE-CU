"""Video source -> MediaPipe face landmarks -> per-frame asymmetry features (+ eye gaze),
and MediaPipe pose landmarks -> arm heights (used by the BE-FAST check, befast.py).

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
# Iris centres (the face_landmarker model returns 478 points; 468-477 are the irises).
# Which centre belongs to which eye is decided geometrically, see horizontal_gaze().
IRIS_CENTERS = (468, 473)
# Each eye's corners ordered from the patient's RIGHT to the patient's LEFT.
EYE_SPANS = ((EYE_OUT_R, EYE_IN_R), (EYE_IN_L, EYE_OUT_L))

# MediaPipe pose indices (patient's left/right, like the face).
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
POSE_LINES = [(SH_L, SH_R), (SH_L, EL_L), (EL_L, WR_L), (SH_R, EL_R), (EL_R, WR_R)]

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
    head_speed: float | None = None  # deg/s, from the previous frame (MotionGate)
    # Horizontal eye position: iris offset from the middle of the eye opening, as a fraction
    # of the eye width, averaged over both eyes. + = looking to the patient's LEFT.
    # About +-0.15..0.25 at full sideways gaze. None if eyes closed / head turned > 30 deg.
    gaze_h: float | None = None
    blink: float | None = None       # max eyeBlink blendshape (eyes closed ~ > 0.5)
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


def upright_y(pts: np.ndarray, r_gl: np.ndarray, method: str) -> np.ndarray | None:
    """Vertical landmark positions with head rotation removed, divided by the outer
    eye-corner distance; y down (bigger = lower on the face). None if degenerate.

    "2d": level the line between the outer eye corners (removes roll only).
    "3d": full rotation removal with MediaPipe's facial transformation matrix (its depth
          estimates leak into y, so turning the head biases it differently from 2d).
    "blend" (default): average of the two. Their head-turn errors partly cancel: on
          Mac/iPhone/overhead clips the worst leftover turn bias, fitted without the
          tested setup, was 1.4 vs 2.3 (2d) and 2.6 (3d) points of score (Nov 14 2026).
    """
    if method == "blend":
        a, b = upright_y(pts, r_gl, "2d"), upright_y(pts, r_gl, "3d")
        return None if a is None or b is None else (a + b) / 2
    if method == "3d":
        q = (pts - pts.mean(axis=0)) @ (_FLIP @ r_gl @ _FLIP)
    else:
        p2 = pts[:, :2] - pts[:, :2].mean(axis=0)
        v = p2[EYE_OUT_L] - p2[EYE_OUT_R]
        a = math.atan2(v[1], v[0])
        c, s = math.cos(a), math.sin(a)
        q = p2 @ np.array([[c, -s], [s, c]])  # rotate by -a: eye line becomes horizontal
    iod = float(np.linalg.norm(q[EYE_OUT_L] - q[EYE_OUT_R]))
    return None if iod < 1e-6 else q[:, 1] / iod


def compute_features(landmarks, blendshapes, matrix, w: int, h: int, t: float, vc: dict
                     ) -> FrameFeatures:
    """Features from one face's landmarks. Pure function so it can be unit-tested."""
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in landmarks], dtype=float)
    r_gl = rotation_from_matrix(matrix)
    yaw, pitch, roll = euler_deg(r_gl)
    ff = FrameFeatures(t=t, yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(roll, 1))

    bs = {c.category_name: c.score for c in blendshapes} if blendshapes else {}
    ff.jaw_open = round(bs.get("jawOpen", 0.0), 3)
    blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
    ff.blink = round(blink, 3)

    # Gaze before the head-turn and talking gates: in a stroke the head often turns WITH the
    # eyes (live test, Nov 14 2026: head 20-24 deg while holding the eyes right), and talking
    # doesn't move the eyes. Only up to gaze_max_yaw_deg, beyond that the far eye is hidden.
    # horizontal_gaze() is pure dot products / eye width, so in-plane rotation (roll), shift
    # and scale cancel: raw image points give the same value as roll-levelled ones.
    if (blink <= vc["blink_max"] and len(pts) >= 478
            and abs(yaw) <= vc.get("gaze_max_yaw_deg", 30) and abs(pitch) <= vc["max_pitch_deg"]):
        g = horizontal_gaze(pts[:, :2])
        ff.gaze_h = None if g is None else round(g, 4)

    if abs(yaw) > vc["max_yaw_deg"]:
        ff.reason = "head turned"
        return ff
    if abs(pitch) > vc["max_pitch_deg"] or abs(roll) > vc["max_roll_deg"]:
        ff.reason = "head tilted"
        return ff

    y = upright_y(pts, r_gl, vc.get("correction", "blend"))
    if y is None:
        ff.reason = "no face"
        return ff

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

    if blink > vc["blink_max"]:
        ff.reason = "blinking (eye feature skipped)"
    else:
        open_r = np.mean([y[b] - y[a] for a, b in EYE_LIDS_R])
        open_l = np.mean([y[b] - y[a] for a, b in EYE_LIDS_L])
        if open_r > 1e-4 and open_l > 1e-4:
            ff.eye_ratio = float(math.log(open_r / open_l))
    remove_yaw_bias(ff, vc)
    ff.valid = True
    return ff


YAW_BIAS_FEATURES = ("mouth_diff", "eye_ratio", "brow_diff")


def yaw_bias_table(vc: dict) -> dict | None:
    """The vision.yaw_bias table if it applies to the configured correction, else None."""
    tb = vc.get("yaw_bias")
    if not tb or tb.get("correction") != vc.get("correction", "blend"):
        return None
    return tb


def remove_yaw_bias(ff: FrameFeatures, vc: dict) -> None:
    """Subtract the fake asymmetry that turning the head causes (vision.yaw_bias, fitted
    on recorded head-turn clips by scripts/fit_yaw_bias.py). In place."""
    tb = yaw_bias_table(vc)
    if tb is None or ff.yaw is None:
        return
    for f in YAW_BIAS_FEATURES:
        v = getattr(ff, f)
        if v is not None and f in tb:
            setattr(ff, f, v - float(np.interp(ff.yaw, tb["knots"], tb[f])))


class MotionGate:
    """Skips frames while the head is turning/tilting faster than max_dps (deg/s).
    Landmarks lag behind a moving head, so moving frames carry about twice the mouth error
    of still ones (Nov 14 2026 clips: 90th percentile 1.2-1.8 mm above 50 deg/s vs 0.45-0.9 mm
    below 10). A resting patient loses almost nothing. 0 = off."""

    def __init__(self, max_dps: float):
        self.max_dps = max_dps or 0.0
        self.reset()

    def reset(self):
        self.prev: tuple[float, float, float, float] | None = None  # t, yaw, pitch, roll

    def apply(self, ff: FrameFeatures) -> None:
        if not self.max_dps or ff.yaw is None:
            return
        cur = (ff.t, ff.yaw, ff.pitch or 0.0, ff.roll or 0.0)
        prev, self.prev = self.prev, cur
        if prev is None or cur[0] - prev[0] <= 0 or cur[0] - prev[0] > 1.0:
            return
        speed = max(abs(a - b) for a, b in zip(cur[1:], prev[1:])) / (cur[0] - prev[0])
        ff.head_speed = round(speed, 1)
        if speed > self.max_dps and ff.valid:
            ff.valid, ff.reason = False, "head moving"


def horizontal_gaze(p2: np.ndarray) -> float | None:
    """Mean over both eyes of (iris centre - eye-opening middle), projected on the line from
    the eye's patient-right corner to its patient-left corner, divided by the eye width.
    + = looking to the patient's LEFT. Works in any in-plane rotation (pure dot products)."""
    mids = [(p2[a] + p2[b]) / 2 for a, b in EYE_SPANS]
    vals = []
    for iris in IRIS_CENTERS:
        p = p2[iris]
        k = int(np.argmin([np.linalg.norm(p - m) for m in mids]))  # nearest eye
        a, b = EYE_SPANS[k]
        v = p2[b] - p2[a]
        w2 = float(v @ v)
        if w2 < 1e-9:
            continue
        vals.append(float((p - mids[k]) @ v) / w2)
    return float(np.mean(vals)) if vals else None


@dataclass
class PoseFeatures:
    """Arm heights (wrist above shoulder) in shoulder widths (+ = higher), from MediaPipe pose.
    'Up' is perpendicular to the shoulder line, so a leaning body doesn't look like drift.
    None = joint not visible."""
    t: float
    arm_l: float | None = None   # wrist above shoulder / shoulder width
    arm_r: float | None = None
    shoulders: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def compute_pose_features(landmarks, w: int, h: int, t: float, vis_min: float = 0.5
                          ) -> PoseFeatures:
    """Pure function (testable): pose landmarks -> PoseFeatures."""
    xy = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=float)

    def ok(i: int) -> bool:
        p = landmarks[i]
        vis = p.visibility if getattr(p, "visibility", None) is not None else 1.0
        return vis >= vis_min and -0.02 <= p.x <= 1.02 and -0.02 <= p.y <= 1.02

    def up_and_width(a: int, b: int) -> tuple[np.ndarray, float]:
        v = xy[a] - xy[b]
        n = float(np.linalg.norm(v))
        up = np.array([v[1], -v[0]]) / max(n, 1e-6)
        if up[1] > 0:  # image y is down; 'up' must point up the image (also if mirrored)
            up = -up
        return up, n

    pf = PoseFeatures(t=t)
    if ok(SH_L) and ok(SH_R):
        up, width = up_and_width(SH_L, SH_R)
        if width > 1:
            pf.shoulders = True
            for wr, sh, attr in ((WR_L, SH_L, "arm_l"), (WR_R, SH_R, "arm_r")):
                if ok(wr):
                    setattr(pf, attr, round(float((xy[wr] - xy[sh]) @ up) / width, 4))
    return pf


class PoseAnalyzer:
    """MediaPipe Pose Landmarker; only run while the BE-FAST check needs the arms."""

    def __init__(self, cfg: dict):
        vc = cfg["vision"]
        path = ROOT / vc.get("pose_model_path", "models/pose_landmarker_lite.task")
        if not path.exists():
            raise FileNotFoundError(f"pose model missing: {path} (see README setup)")
        self.vis_min = vc.get("pose_min_visibility", 0.5)
        opts = mpv.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mpv.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mpv.PoseLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[PoseFeatures, np.ndarray | None]:
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self.landmarker.detect_for_video(img, ms)
        if not res.pose_landmarks:
            return PoseFeatures(t=t), None
        lms = res.pose_landmarks[0]
        pf = compute_pose_features(lms, w, h, t, self.vis_min)
        vis = [(p.visibility if p.visibility is not None else 1.0) >= self.vis_min for p in lms]
        px = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
        px[~np.array(vis)] = np.nan
        return pf, px

    def close(self):
        self.landmarker.close()


def draw_pose(frame: np.ndarray, px: np.ndarray | None) -> np.ndarray:
    """Arm skeleton (in place); joints that aren't visible (NaN) are left out."""
    if px is None:
        return frame
    for a, b in POSE_LINES:
        if np.isnan(px[a]).any() or np.isnan(px[b]).any():
            continue
        cv2.line(frame, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]),
                 (240, 180, 60), 3, cv2.LINE_AA)
    for i in (SH_L, SH_R, EL_L, EL_R, WR_L, WR_R):
        if not np.isnan(px[i]).any():
            cv2.circle(frame, tuple(int(v) for v in px[i]), 5, (255, 220, 120), -1, cv2.LINE_AA)
    return frame


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
        self.last_lms: np.ndarray | None = None
        self._last_ms = -1
        self._motion = MotionGate(self.vc.get("max_head_speed_dps", 0))
        if self.vc.get("yaw_bias") and yaw_bias_table(self.vc) is None:
            print(f"warning: vision.yaw_bias was fitted for correction "
                  f"{self.vc['yaw_bias'].get('correction')!r}, not {self.vc.get('correction')!r}; "
                  "head-turn bias is NOT removed. Refit with scripts/fit_yaw_bias.py.")

    def process(self, frame_bgr: np.ndarray, t: float) -> tuple[FrameFeatures, np.ndarray | None]:
        """Returns features and the landmark pixel coords (N x 2) for drawing, or None."""
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        res = self.landmarker.detect_for_video(img, ms)
        if not res.face_landmarks:
            self._motion.reset()
            self.last_lms = None
            return FrameFeatures(t=t, reason="no face"), None
        lms = res.face_landmarks[0]
        # normalised (478, 3) landmarks of this frame, for the learned palsy models (strokewatch/palsy.py)
        self.last_lms = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
        ff = compute_features(lms, res.face_blendshapes[0] if res.face_blendshapes else None,
                              res.facial_transformation_matrixes[0], w, h, t, self.vc)
        self._motion.apply(ff)
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
        if len(px) >= 478 and ff.gaze_h is not None:
            for i in IRIS_CENTERS:
                cv2.circle(frame, tuple(int(v) for v in px[i]), 3, (255, 200, 0), -1, cv2.LINE_AA)
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
    if ff.gaze_h is not None:
        side = "L" if ff.gaze_h > 0 else "R"
        lines.append(f"gaze {abs(ff.gaze_h):.2f} {side}")
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
