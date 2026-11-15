"""Multi-bed core: one worker thread per bed + a manager that tracks alerts. All in memory.

Bed (thread): frames -> features -> Detector -> latest status, full-size JPEG (bed page)
and a small thumbnail JPEG (dashboard grid). Also the BE-FAST extras (befast.py): a passive
GazeMonitor, and a voice-guided ExamRunner started on a facial-droop ALERT, a gaze deviation
or by staff. The face detector is paused while a check runs.
  CameraBed: webcam index, video file or RTSP URL (anything cv2.VideoCapture opens).
  PhoneBed:  JPEG frames pushed by a phone's browser (web/phone.html) over a WebSocket.
  SimBed:    scripted patient from mock.py, no camera.
BedManager: owns the beds, raises an Alert when a bed enters ALERT, handles
acknowledge / resolve / reset, attaches BE-FAST check results to alerts (or raises one),
and builds the dashboard snapshot.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .befast import STOPPED, ExamRunner, GazeMonitor, exam_cfg
from . import palsy
from .detector import Detector
from .localize import localize
from .mock import SimScenario
from .vision import (FaceAnalyzer, FrameFeatures, PoseAnalyzer, PoseFeatures, VideoSource,
                     draw_landmarks, draw_pose, draw_readout)

BED_DEFAULTS = {"analysis_fps": 15, "sim_fps": 10, "thumb_fps": 4, "thumb_width": 320,
                "offline_after_s": 3.0}
FULL_FRAME_IDLE_S = 2.0  # stop encoding full-size JPEGs when no bed page has asked for 2 s


def bed_cfg(cfg: dict) -> dict:
    return {**BED_DEFAULTS, **(cfg.get("beds") or {})}


class Bed(threading.Thread):
    kind = "camera"

    def __init__(self, bed_id: str, name: str, cfg: dict, fps: float):
        super().__init__(daemon=True, name=f"bed-{bed_id}")
        self.bed_id, self.bed_name, self.cfg = bed_id, name, cfg
        self.bc = bed_cfg(cfg)
        self.interval = 1.0 / fps
        self.detector = Detector(cfg)
        # combined droop detection (StrokeWatch + palsy model vs own baseline); None = StrokeWatch alone
        self.combined = palsy.new_bed_state(fps, cfg["detector"]["alert_seconds"])
        if self.combined is not None:
            self.detector.native_alerts = False
        self.on_alert = None  # set by BedManager: fn(bed, status_dict)
        self.on_say = None    # set by BedManager: fn(bed, text, step_key) -> speak to the patient
        self.on_exam_done = None  # set by BedManager: fn(bed, result_dict)
        self.ec = exam_cfg(cfg)
        self.gaze = GazeMonitor(cfg)
        self.exam: ExamRunner | None = None
        self.last_exam: dict | None = None
        self._exam_req: tuple[str, str] | None = None  # ("start", trigger) | ("cancel", why)
        self._next_auto_exam = 0.0     # frame time; gaze-triggered checks wait for this
        self._alert_exam_done = False  # one automatic check per facial-droop ALERT
        self._pose: PoseAnalyzer | None = None
        self._pose_error: str | None = None
        self.lock = threading.Lock()
        self.frame_id = 0
        self.jpeg: bytes | None = None
        self.thumb_id = 0
        self.thumb: bytes | None = None
        self.status: dict | None = None
        self.last_frame_wall = 0.0
        self.created_wall = time.time()
        self.fps = 0.0
        self._last_loop = None
        self._next_thumb = 0.0
        self._full_wanted_until = 0.0
        self._reset = threading.Event()
        self._calibrate = False
        self._stop_evt = threading.Event()
        self._jpeg_q = [int(cv2.IMWRITE_JPEG_QUALITY), cfg["server"]["jpeg_quality"]]
        self._thumb_q = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

    # ------------------------------------------------------------ control (any thread)
    def reset(self, calibrate: bool = False):
        """Rebuild the baseline (applied by the worker thread on its next frame).
        calibrate=True: guided baseline with head turns (prompts in the status)."""
        self._calibrate = calibrate
        self._reset.set()

    def stop(self, timeout: float = 2.0):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        else:
            self.close()

    def request_exam(self, trigger: str = "requested by staff"):
        """Start a BE-FAST check on the next frame (any thread)."""
        self._exam_req = ("start", trigger)

    def cancel_exam(self, why: str = "cancelled by staff"):
        self._exam_req = ("cancel", why)

    def want_full_frames(self):
        """Called by bed-page video streams; full-size JPEGs are only encoded while wanted."""
        self._full_wanted_until = time.monotonic() + FULL_FRAME_IDLE_S

    def latest(self) -> tuple[int, bytes | None, dict | None]:
        with self.lock:
            return self.frame_id, self.jpeg, self.status

    def latest_thumb(self) -> tuple[int, bytes | None]:
        with self.lock:
            return self.thumb_id, self.thumb

    # ------------------------------------------------------------ subclass hooks
    def capture(self) -> tuple[np.ndarray, FrameFeatures, np.ndarray | None] | None:
        """Next (frame, features, landmark px) at the bed's analysis rate, or None if no frame."""
        raise NotImplementedError

    def close(self):
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def pose_features(self, frame: np.ndarray, t: float
                      ) -> tuple[PoseFeatures | None, np.ndarray | None]:
        """Arm/leg heights for the BE-FAST check. The pose model is loaded on first use."""
        if self._pose is None and self._pose_error is None:
            try:
                self._pose = PoseAnalyzer(self.cfg)
            except Exception as e:  # missing model file etc.: check runs without arms/legs
                self._pose_error = str(e)
                print(f"{self.bed_name}: pose analysis unavailable: {e}")
        if self._pose is None:
            return None, None
        return self._pose.process(frame, t)

    def pose_problem(self) -> str | None:
        """Why arms/legs can't be checked (None = fine). Loads the model if needed."""
        if self._pose is None and self._pose_error is None:
            self.pose_features(np.zeros((64, 64, 3), np.uint8), 0.0)
        return self._pose_error

    # ------------------------------------------------------------ BE-FAST check
    def _say(self, text: str, step: str | None):
        if self.on_say and text:
            try:
                self.on_say(self, text, step)
            except Exception as e:  # speech is best-effort; never kill the bed thread
                print(f"{self.bed_name}: could not speak: {e}")

    def _start_exam(self, trigger: str, t: float):
        self.exam = ExamRunner(self.cfg, trigger, gaze_center=self.gaze.center,
                               pose_error=self.pose_problem())
        print(f"{self.bed_name}: BE-FAST check started ({trigger})")

    def _exam_tick(self, frame: np.ndarray, ff: FrameFeatures, det_state: str):
        """Handle staff requests and automatic triggers, then advance a running check.
        Returns pose landmark px to draw (or None)."""
        req, self._exam_req = self._exam_req, None
        if req and req[0] == "cancel" and self.exam:
            self.exam.cancel(req[1])
            self._say(STOPPED, None)
        elif req and req[0] == "start" and self.exam is None:
            self._start_exam(req[1], ff.t)
        if self.exam is None and self.ec["auto"]:
            if det_state == "ALERT" and not self._alert_exam_done:
                self._alert_exam_done = True
                self._start_exam(f"facial droop ({self.detector.side})", ff.t)
            elif (self.gaze.deviation and det_state != "baselining"
                  and ff.t >= self._next_auto_exam):
                self._start_exam(f"gaze held to the {self.gaze.deviation.lower()}", ff.t)
        ex, ppx = self.exam, None
        if ex is None:
            return None
        pose = None
        if ex.needs_pose:
            pose, ppx = self.pose_features(frame, ff.t)
        for text in ex.update(ff, pose):
            self._say(text, ex.step.key if ex.step and not ex.finished else "done")
        if ex.finished:
            self.exam = None
            self._next_auto_exam = ff.t + self.ec["cooldown_s"]
            self.gaze.clear_window()
            if ex.result is not None:
                self.last_exam = ex.result
                print(f"{self.bed_name}: BE-FAST check done: {ex.result['summary']}")
                if self.on_exam_done:
                    self.on_exam_done(self, ex.result)
            else:
                self.last_exam = {"cancelled": ex.cancelled, "trigger": ex.trigger,
                                  "started_at": ex.started_at, "finished_at": ff.t,
                                  "summary": f"Check stopped: {ex.cancelled}"}
        return ppx

    # ------------------------------------------------------------ worker
    def run(self):
        try:
            while not self._stop_evt.is_set():
                item = self.capture()
                if item is None:
                    self._stop_evt.wait(0.1)
                    continue
                self._process(*item)
        finally:
            self.close()

    def _combined_tick(self, frame: np.ndarray, ff: FrameFeatures, px: np.ndarray | None, st: dict):
        """Feed the palsy model and apply the combined rule; the bed's score / threshold become the combined
        score and its line (dashboard, bed page and alerts use them), StrokeWatch's own are kept as sw_*."""
        c, d = self.combined, self.detector
        if self.exam is None:  # deliberate eye/arm movements during a BE-FAST check: pause like the detector
            analyzer = getattr(self, "analyzer", None)  # camera / phone beds; simulated beds have none
            lms = analyzer.last_lms if (analyzer is not None and px is not None) else None
            h, w = frame.shape[:2]
            if c.push(lms, w, h, ff.t, ff.mouth_diff if ff.valid else None):
                if c.decide(ff.t, d.state, d.score, d.dc["threshold"]) and d.state != "ALERT":
                    d.state = "ALERT"
                    d.side = c.side(d.side)
                    d.onset_time = ff.t - c.above
                    st.update(state="ALERT", side=d.side, onset_time=d.onset_time)
        cs = c.status()
        st["sw_score"], st["sw_threshold"] = st["score"], st["threshold"]
        st.update(score=cs["score"], threshold=cs["line"], pre_threshold=round(cs["line"] * 0.625, 2),
                  above_seconds=cs["above_seconds"], combined=cs)

    def _process(self, frame: np.ndarray, ff: FrameFeatures, px: np.ndarray | None):
        if self._reset.is_set():
            self._reset.clear()
            self.detector.reset(calibrate=self._calibrate)
            self.gaze.reset()
            if self.combined is not None:
                self.combined.reset()
            if self.exam:
                self.exam.cancel("bed reset")
                self.exam = None
            self._alert_exam_done = False
            self._next_auto_exam = 0.0
        prev_state = self.detector.state
        ppx = self._exam_tick(frame, ff, prev_state)
        if self.exam is not None:  # deliberate eye/arm movements: pause the face detector
            st = self.detector.update(FrameFeatures(t=ff.t, reason="BE-FAST check running")
                                      ).to_dict()
        else:
            st = self.detector.update(ff).to_dict()
            self.gaze.update(ff)
        now = time.monotonic()
        if self._last_loop is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_loop, 1e-3)
        self._last_loop = now
        st["fps"] = round(self.fps, 1)
        st["features"] = {k: getattr(ff, k) for k in
                          ("mouth_diff", "eye_ratio", "brow_diff", "yaw", "pitch", "roll",
                           "gaze_h", "blink")}
        st["gaze"] = self.gaze.status()
        if self.combined is not None:
            self._combined_tick(frame, ff, px, st)
        st["exam"] = self.exam.status() if self.exam else {"active": False}
        st["last_exam"] = self.last_exam

        draw_landmarks(frame, px, ff)
        draw_pose(frame, ppx)
        thumb = None
        if now >= self._next_thumb:
            self._next_thumb = now + 1.0 / self.bc["thumb_fps"]
            h, w = frame.shape[:2]
            tw = self.bc["thumb_width"]
            small = cv2.resize(frame, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA)
            thumb = cv2.imencode(".jpg", small, self._thumb_q)[1].tobytes()
        jpeg = None
        if now < self._full_wanted_until:
            draw_readout(frame, ff)
            jpeg = cv2.imencode(".jpg", frame, self._jpeg_q)[1].tobytes()

        with self.lock:
            self.frame_id += 1
            self.status = st
            self.last_frame_wall = time.time()
            if jpeg is not None:
                self.jpeg = jpeg
            if thumb is not None:
                self.thumb_id += 1
                self.thumb = thumb
        if st["state"] == "ALERT" and prev_state != "ALERT" and self.on_alert:
            self.on_alert(self, st)


class CameraBed(Bed):
    kind = "camera"

    def __init__(self, bed_id: str, name: str, cfg: dict, source: str):
        super().__init__(bed_id, name, cfg, bed_cfg(cfg)["analysis_fps"])
        self.source = source
        # Open on the main thread: that's where macOS shows the camera permission prompt.
        self.src = VideoSource(source, cfg["camera"]["width"], cfg["camera"]["height"],
                               clock="wall", realtime=True, loop=True,
                               rotate=cfg["camera"].get("rotate", 0))
        self.analyzer = FaceAnalyzer(cfg)
        self._next_due = 0.0

    def capture(self):
        # Read every camera frame (keeps latency low) but only analyze at analysis_fps.
        while not self._stop_evt.is_set():
            ok, frame, t = self.src.read()
            if not ok:
                self._reconnect()
                return None
            now = time.monotonic()
            if now + 0.005 >= self._next_due:
                self._next_due = max(self._next_due + self.interval, now)
                ff, px = self.analyzer.process(frame, t)
                return frame, ff, px
        return None

    def _reconnect(self):
        """Camera stopped delivering frames: retry every 2 s (the dashboard shows offline)."""
        while not self._stop_evt.wait(2.0):
            try:
                self.src.reopen()
                return
            except RuntimeError:
                continue

    def close(self):
        self.src.release()
        self.analyzer.close()
        super().close()


class PhoneBed(Bed):
    """Frames pushed by a phone's Safari page. The phone rotates/scales them before sending."""
    kind = "phone"

    def __init__(self, bed_id: str, name: str, cfg: dict, device_id: str):
        super().__init__(bed_id, name, cfg, bed_cfg(cfg)["analysis_fps"])
        self.device_id = device_id
        self.analyzer = FaceAnalyzer(cfg)
        self.sock = None  # the phone's current WebSocket; only touched on the server's event loop
        self._frame: tuple[bytes, float] | None = None
        self._frame_evt = threading.Event()
        self._next_due = 0.0

    def push(self, jpeg: bytes):
        """Called for every frame the phone sends; only the newest waiting frame is kept."""
        with self.lock:
            self._frame = (jpeg, time.time())
        self._frame_evt.set()

    def capture(self):
        if not self._frame_evt.wait(0.5):
            return None
        wait = self._next_due - time.monotonic()  # cap at analysis_fps
        if wait > 0 and self._stop_evt.wait(wait):
            return None
        self._frame_evt.clear()
        with self.lock:
            item, self._frame = self._frame, None
        if item is None:
            return None
        self._next_due = max(self._next_due + self.interval, time.monotonic())
        data, t = item
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        ff, px = self.analyzer.process(frame, t)
        return frame, ff, px

    def close(self):
        self.analyzer.close()
        super().close()


class SimBed(Bed):
    kind = "sim"

    def __init__(self, bed_id: str, name: str, cfg: dict, scenario: SimScenario, seed: int = 0):
        super().__init__(bed_id, name, cfg, bed_cfg(cfg)["sim_fps"])
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.t0 = time.time()
        self._next_due = time.monotonic()

    def capture(self):
        wait = self._next_due - time.monotonic()
        if wait > 0 and self._stop_evt.wait(wait):
            return None
        self._next_due = max(self._next_due + self.interval, time.monotonic())
        t = time.time()
        el = t - self.t0
        if self.scenario.is_offline(el):
            return None
        cue = self.exam.cue if self.exam else None
        ff = self.scenario.features(t, el, self.rng, cue)
        self._sim_pose = self.scenario.pose(t, el, cue, self.rng) if cue else None
        return self.scenario.draw(ff, el, cue), ff, None

    def pose_features(self, frame, t):
        return getattr(self, "_sim_pose", None), None

    def pose_problem(self):
        return None


@dataclass
class Alert:
    id: int
    bed_id: str
    bed_name: str
    side: str | None                # patient's WEAK side (signs), not the side of the brain
    onset_time: float | None
    raised_at: float
    score: float
    kind: str = "face"              # face (droop detector) | exam (BE-FAST check)
    findings: list = field(default_factory=list)  # human-readable signs, newest last
    notes: list = field(default_factory=list)     # not signs: e.g. a check part inconclusive
    exam: dict | None = None        # latest BE-FAST check result for this alert
    status: str = "active"          # active -> acknowledged -> resolved
    acked_at: float | None = None
    resolved_at: float | None = None
    resolution: str | None = None   # false_alarm | confirmed | reset | removed
    face_side: str | None = None    # side of the face droop, if the detector saw one
    localization: dict | None = None  # localize.localize(): hedged "may be a ... stroke"

    def relocalize(self):
        self.localization = localize(self.face_side, self.exam)
        if self.localization and self.localization["weak_side"]:
            self.side = self.localization["weak_side"]


RESOLUTIONS = {"false_alarm", "confirmed", "reset", "removed"}
MAX_ALERTS = 200


class BedManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bc = bed_cfg(cfg)
        self.beds: dict[str, Bed] = {}
        self.alerts: list[Alert] = []
        self.lock = threading.Lock()
        self._next_alert_id = 1
        self._next_bed_num = 1
        self.started = False
        self.say_hook = None  # set by the server: fn(bed, text, step) -> phone / local speaker

    def allocate(self) -> tuple[str, str]:
        """Next free (bed_id, default name), e.g. ("bed9", "Bed 9")."""
        with self.lock:
            n = self._next_bed_num
            self._next_bed_num += 1
        return f"bed{n}", f"Bed {n}"

    def add(self, bed: Bed):
        """Add a bed; it starts right away if the manager is already running (phones)."""
        with self.lock:
            if bed.bed_id in self.beds:
                raise ValueError(f"duplicate bed id {bed.bed_id!r}")
            bed.on_alert = self._raise_alert
            bed.on_say = self._say
            bed.on_exam_done = self._exam_done
            self.beds[bed.bed_id] = bed
        if self.started:
            bed.start()

    def remove(self, bed_id: str) -> bool:
        """Stop and forget a bed (blocks up to ~2 s while its thread winds down)."""
        with self.lock:
            bed = self.beds.pop(bed_id, None)
            if bed is None:
                return False
            for a in self.alerts:
                if a.bed_id == bed_id and a.status != "resolved":
                    self._close(a, "removed")
        bed.stop()
        return True

    def bed_list(self) -> list[Bed]:
        """Snapshot of the beds; iterate this, not self.beds (phones add/remove beds live)."""
        with self.lock:
            return list(self.beds.values())

    def phone_bed(self, device_id: str) -> PhoneBed | None:
        return next((b for b in self.bed_list()
                     if isinstance(b, PhoneBed) and b.device_id == device_id), None)

    def start(self):
        self.started = True
        for bed in self.bed_list():
            if not bed.is_alive():
                bed.start()

    def stop(self):
        beds = self.bed_list()
        for bed in beds:
            bed._stop_evt.set()  # signal all first so they wind down in parallel
        for bed in beds:
            bed.stop()

    # ------------------------------------------------------------ alerts
    def _open_alert(self, bed_id: str) -> Alert | None:  # caller holds self.lock
        return next((a for a in self.alerts if a.bed_id == bed_id and a.status != "resolved"),
                    None)

    def _raise_alert(self, bed: Bed, st: dict):  # called from the bed's thread
        finding = f"Face: possible new {st.get('side')} facial droop"
        with self.lock:
            a = self._open_alert(bed.bed_id)
            if a is not None:  # e.g. a BE-FAST check alert is already open: add the droop
                if finding not in a.findings:
                    a.findings.insert(0, finding)
                    a.status, a.acked_at = "active", None  # new sign -> alarm again
                a.face_side = st.get("side")
                a.relocalize()
                return
            a = Alert(
                id=self._next_alert_id, bed_id=bed.bed_id, bed_name=bed.bed_name,
                side=st.get("side"), onset_time=st.get("onset_time"), raised_at=time.time(),
                score=st.get("score", 0.0), findings=[finding], face_side=st.get("side"))
            a.relocalize()
            self.alerts.insert(0, a)
            self._next_alert_id += 1
            del self.alerts[MAX_ALERTS:]
        print(f"ALERT {bed.bed_name}: possible new {st.get('side')} droop")

    def _say(self, bed: Bed, text: str, step: str | None):  # called from the bed's thread
        if self.say_hook:
            self.say_hook(bed, text, step)

    def _exam_done(self, bed: Bed, res: dict):  # called from the bed's thread
        """Attach a finished BE-FAST check to the bed's open alert, or raise one if flagged."""
        with self.lock:
            a = self._open_alert(bed.bed_id)
            if a is not None:
                a.exam = res
                new = [f for f in res["findings"] if f not in a.findings]
                a.findings.extend(new)
                a.notes = list(res.get("notes") or [])  # about the latest check only
                if a.side is None and res["weak_side"] in ("LEFT", "RIGHT"):
                    a.side = res["weak_side"]
                a.relocalize()  # face droop + this check together
                res["localization"] = a.localization
                return
            if not res["flagged"]:
                return
            a = Alert(
                id=self._next_alert_id, bed_id=bed.bed_id, bed_name=bed.bed_name,
                side=res["weak_side"] if res["weak_side"] in ("LEFT", "RIGHT") else None,
                onset_time=res["started_at"], raised_at=time.time(), score=0.0, kind="exam",
                findings=list(res["findings"]), notes=list(res.get("notes") or []), exam=res)
            a.relocalize()
            self.alerts.insert(0, a)
            self._next_alert_id += 1
            del self.alerts[MAX_ALERTS:]
        print(f"ALERT {bed.bed_name}: BE-FAST check: {res['summary']}")

    def request_exam(self, bed_id: str) -> bool:
        bed = self.beds.get(bed_id)
        if bed is None:
            return False
        bed.request_exam()
        return True

    def cancel_exam(self, bed_id: str) -> bool:
        bed = self.beds.get(bed_id)
        if bed is None:
            return False
        bed.cancel_exam()
        return True

    def _find(self, alert_id: int) -> Alert | None:
        return next((a for a in self.alerts if a.id == alert_id), None)

    def acknowledge(self, alert_id: int) -> bool:
        with self.lock:
            a = self._find(alert_id)
            if a is None or a.status != "active":
                return False
            a.status, a.acked_at = "acknowledged", time.time()
            return True

    def resolve(self, alert_id: int, resolution: str) -> bool:
        """Close the alert and rebuild that bed's baseline so monitoring can continue."""
        if resolution not in RESOLUTIONS:
            raise ValueError(resolution)
        with self.lock:
            a = self._find(alert_id)
            if a is None or a.status == "resolved":
                return False
            self._close(a, resolution)
            bed = self.beds.get(a.bed_id)
        if bed:
            bed.reset()
        return True

    def reset_bed(self, bed_id: str, calibrate: bool = False) -> bool:
        bed = self.beds.get(bed_id)
        if bed is None:
            return False
        with self.lock:
            for a in self.alerts:
                if a.bed_id == bed_id and a.status != "resolved":
                    self._close(a, "reset")
        bed.reset(calibrate=calibrate)
        return True

    @staticmethod
    def _close(a: Alert, resolution: str):
        now = time.time()
        a.status, a.resolution, a.resolved_at = "resolved", resolution, now
        a.acked_at = a.acked_at or now

    # ------------------------------------------------------------ snapshots
    def bed_status(self, bed: Bed, detail: bool = False) -> dict:
        """Status for one bed; detail=True adds per-feature values (bed page)."""
        _, _, st = bed.latest()
        with bed.lock:
            last = bed.last_frame_wall
        now = time.time()
        out = dict(st) if st else {"state": "starting", "score": 0.0, "valid": False,
                                   "threshold": self.cfg["detector"]["threshold"],
                                   "pre_threshold": self.cfg["detector"]["pre_threshold"],
                                   "baseline_progress": 0.0, "invalid_reason": None}
        out.update(bed_id=bed.bed_id, bed_name=bed.bed_name, kind=bed.kind,
                   detector_state=out["state"],
                   last_frame_age=round(now - last, 1) if last else None)
        if isinstance(bed, PhoneBed):
            out["phone_connected"] = bed.sock is not None
        if st is not None and now - last > self.bc["offline_after_s"]:
            out["state"] = "offline"
        elif st is None and now - bed.created_wall > 10:  # never produced a frame
            out["state"] = "offline"
        if not detail:
            out.pop("features", None)
            out.pop("z", None)
            le = out.get("last_exam")
            if le:  # dashboard only needs the gist
                out["last_exam"] = {k: le.get(k) for k in ("summary", "flagged", "finished_at",
                                                           "trigger", "cancelled",
                                                           "inconclusive", "notes")}
        out.setdefault("exam", {"active": False})
        out.setdefault("gaze", None)
        out.setdefault("last_exam", None)
        with self.lock:
            open_alert = self._open_alert(bed.bed_id)
            out["alert"] = asdict(open_alert) if open_alert else None
            if not detail and open_alert and open_alert.exam:
                out["alert"]["exam"] = None  # full result is on the bed page
        # An alert raised by the BE-FAST check (face detector still "monitoring") shows as ALERT.
        if open_alert and out["state"] not in ("offline", "starting"):
            out["state"] = "ALERT"
            out["side"] = open_alert.side or out.get("side")
        return out

    def snapshot(self) -> dict:
        beds = [self.bed_status(b) for b in self.bed_list()]
        with self.lock:
            alerts = [asdict(a) for a in self.alerts]
        return {"type": "state", "server_time": time.time(), "beds": beds, "alerts": alerts}
