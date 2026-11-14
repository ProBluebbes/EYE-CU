"""Multi-bed core: one worker thread per bed + a manager that tracks alerts. All in memory.

Bed (thread): frames -> features -> Detector -> latest status, full-size JPEG (bed page)
and a small thumbnail JPEG (dashboard grid).
  CameraBed: webcam index, video file or RTSP URL (anything cv2.VideoCapture opens).
  PhoneBed:  JPEG frames pushed by a phone's browser (web/phone.html) over a WebSocket.
  SimBed:    scripted patient from mock.py, no camera.
BedManager: owns the beds, raises an Alert when a bed enters ALERT, handles
acknowledge / resolve / reset, and builds the dashboard snapshot.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .detector import Detector
from .mock import SimScenario
from .vision import FaceAnalyzer, FrameFeatures, VideoSource, draw_landmarks, draw_readout

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
        self.on_alert = None  # set by BedManager: fn(bed, status_dict)
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
        self._stop_evt = threading.Event()
        self._jpeg_q = [int(cv2.IMWRITE_JPEG_QUALITY), cfg["server"]["jpeg_quality"]]
        self._thumb_q = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

    # ------------------------------------------------------------ control (any thread)
    def reset(self):
        """Rebuild the baseline (applied by the worker thread on its next frame)."""
        self._reset.set()

    def stop(self, timeout: float = 2.0):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        else:
            self.close()

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
        pass

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

    def _process(self, frame: np.ndarray, ff: FrameFeatures, px: np.ndarray | None):
        if self._reset.is_set():
            self._reset.clear()
            self.detector.reset()
        prev_state = self.detector.state
        st = self.detector.update(ff).to_dict()
        now = time.monotonic()
        if self._last_loop is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_loop, 1e-3)
        self._last_loop = now
        st["fps"] = round(self.fps, 1)
        st["features"] = {k: getattr(ff, k) for k in
                          ("mouth_diff", "eye_ratio", "brow_diff", "yaw", "pitch", "roll")}

        draw_landmarks(frame, px, ff)
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
        ff = self.scenario.features(t, el, self.rng)
        return self.scenario.draw(ff, el), ff, None


@dataclass
class Alert:
    id: int
    bed_id: str
    bed_name: str
    side: str | None
    onset_time: float | None
    raised_at: float
    score: float
    status: str = "active"          # active -> acknowledged -> resolved
    acked_at: float | None = None
    resolved_at: float | None = None
    resolution: str | None = None   # false_alarm | confirmed | reset | removed


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
    def _raise_alert(self, bed: Bed, st: dict):  # called from the bed's thread
        with self.lock:
            if any(a.bed_id == bed.bed_id and a.status != "resolved" for a in self.alerts):
                return
            self.alerts.insert(0, Alert(
                id=self._next_alert_id, bed_id=bed.bed_id, bed_name=bed.bed_name,
                side=st.get("side"), onset_time=st.get("onset_time"), raised_at=time.time(),
                score=st.get("score", 0.0)))
            self._next_alert_id += 1
            del self.alerts[MAX_ALERTS:]
        print(f"ALERT {bed.bed_name}: possible new {st.get('side')} droop")

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

    def reset_bed(self, bed_id: str) -> bool:
        bed = self.beds.get(bed_id)
        if bed is None:
            return False
        with self.lock:
            for a in self.alerts:
                if a.bed_id == bed_id and a.status != "resolved":
                    self._close(a, "reset")
        bed.reset()
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
        with self.lock:
            open_alert = next((a for a in self.alerts
                               if a.bed_id == bed.bed_id and a.status != "resolved"), None)
            out["alert"] = asdict(open_alert) if open_alert else None
        return out

    def snapshot(self) -> dict:
        beds = [self.bed_status(b) for b in self.bed_list()]
        with self.lock:
            alerts = [asdict(a) for a in self.alerts]
        return {"type": "state", "server_time": time.time(), "beds": beds, "alerts": alerts}
