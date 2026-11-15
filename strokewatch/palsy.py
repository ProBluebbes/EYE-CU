"""Combined droop detection (the default): StrokeWatch's baseline-change score + a learned facial-palsy model
(palsynet 3D-CNN, src/palsynet) measured against this patient's own first seconds.

Why: StrokeWatch's own detector is precise but skips every frame with the head moving or tilted, so with a
restless patient it may never finish its baseline. The learned model keeps scoring while the head moves; raw it
reacts to some people's faces and movements, but relative to the patient's own first 20 s it does not. On six
recorded clips with a synthetic one-sided mouth droop (scripts/droop_benchmark.py, scripts/droop_eval.py) this
caught 18/36 droops with no false alerts; StrokeWatch alone caught 6/36 (all on the still clip).

Decision, every model update (0.5 s):
    combined = StrokeWatch score / its threshold   (0 while it is still building its baseline)
             + logit(p) - median logit(p) over the model's first `baseline_s` seconds   (0 until that is built)
    p = 3D-CNN p(affected) on the last 2 s, averaged over the last `avg_windows` updates (5 fold models averaged)
    ALERT when combined >= `line` (1.74) continuously for detector.alert_seconds (6 s).

Configured in config.yaml `palsy:`; turned off with `server --no-palsy` or when the model folder is missing (then
StrokeWatch's own detector raises alerts as before). Models are loaded once and shared by all beds; inference runs
on one background thread so beds keep their analysis rate.
"""
from __future__ import annotations

import math
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import ROOT

_model: "PalsyModel | None" = None
_cfg: dict | None = None


class PalsyModel:
    """One trained run folder (its fold models averaged), shared by every bed."""

    def __init__(self, run_dir: str | Path, device: str = "auto"):
        sys.path.insert(0, str(ROOT / "src"))
        import torch

        from palsynet.engine import get_device
        from palsynet.models import build_model

        self.torch = torch
        self.device = get_device(device)
        d = Path(run_dir)
        if not d.is_absolute():
            d = ROOT / d
        ckpts = sorted(d.rglob("best.pt")) if d.is_dir() else ([d] if d.exists() else [])
        if not ckpts:
            raise FileNotFoundError(f"no best.pt under {d}")
        self.models, cfg = [], None
        for c in ckpts:
            ck = torch.load(c, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            if cfg["data"].get("input") != "landmarks" or cfg["data"].get("landmark_mode") != "3d":
                raise ValueError(f"{c} is not a 3D landmark model")
            m = build_model({**cfg["model"], "pretrained": False}, cfg["data"])
            m.load_state_dict(ck["model"])
            self.models.append(m.to(self.device).eval())
        self.name = cfg["model"]["name"]
        self.num_frames = round(cfg["data"]["segment_seconds"] * cfg["data"]["sample_fps"])
        self.sample_fps = cfg["data"]["sample_fps"]
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="palsy")

    def infer(self, window: np.ndarray) -> float:
        """(T, 152, 3) frontalised landmarks -> p(affected), mean of the fold models."""
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            return float(np.mean([m(x).float().softmax(-1)[0, 1].item() for m in self.models]))


def configure(cfg: dict, enabled: bool = True, model_dir: str | None = None) -> bool:
    """Load the model named in cfg['palsy'] (or model_dir). Returns False (StrokeWatch alone) if unavailable."""
    global _model, _cfg
    pc = dict(cfg.get("palsy") or {})
    if not enabled or not pc.get("enabled", True):
        print("droop detection: StrokeWatch detector only (palsy model off)")
        return False
    path = model_dir or pc.get("model_dir")
    try:
        _model = PalsyModel(path, pc.get("device", "auto"))
    except Exception as e:  # missing folder / torch: run as before rather than not at all
        print(f"warning: palsy model not loaded ({e}); droop alerts come from StrokeWatch's detector alone")
        _model = None
        return False
    _cfg = pc
    print(f"droop detection: StrokeWatch + {_model.name} ({len(_model.models)} folds, {_model.device}) "
          f"vs the patient's first {pc.get('baseline_s', 20)} s, line {pc.get('line', 1.74)}")
    return True


def enabled() -> bool:
    return _model is not None


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class CombinedDroop:
    """One bed: landmark stream for the model + the combined score and its 6 s persistence."""

    def __init__(self, analysis_fps: float, alert_seconds: float):
        from palsynet.landmarks import StreamingLandmarkSmoother

        self.M, self.pc = _model, _cfg
        self._Smoother = StreamingLandmarkSmoother
        self.max_gap = max(1, round(0.5 * analysis_fps))
        self.line = float(self.pc.get("line", 1.74))
        self.baseline_s = float(self.pc.get("baseline_s", 20))
        self.hop_s = float(self.pc.get("hop_s", 0.5))
        self.smooth = int(self.pc.get("smooth_frames", 3))
        self.alert_seconds = alert_seconds
        self.hist = deque(maxlen=int(self.pc.get("avg_windows", 4)))
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        """New patient / rebuilt baseline (called with the detector's reset)."""
        self.smoother, self.size = None, None
        self.buf: deque = deque(maxlen=self.M.num_frames)
        self.next_sample = self.next_pred = None
        self.pending: Future | None = None
        self.hist.clear()
        self.base_logits: list[float] = []
        self.base_start: float | None = None
        self.base_median: float | None = None
        self.base_mouth: list[float] = []
        self.mouth_median: float | None = None
        self.recent_mouth: deque = deque(maxlen=15)
        self.p_avg = None
        self.rel = None          # model logit relative to the personal baseline
        self.sw_term = 0.0
        self.score = 0.0
        self.above = 0.0
        self.last_t: float | None = None
        self.state = "waiting for a face"

    # ---------------------------------------------------------------- landmarks -> model windows
    def push(self, landmarks: np.ndarray | None, w: int, h: int, t: float, mouth_diff: float | None):
        """One analysed frame. Returns True when a new model score has arrived (then call decide())."""
        new = self._collect(t)
        if mouth_diff is not None:
            self.recent_mouth.append(mouth_diff)
        if self.smoother is None or self.size != (w, h):
            self.smoother, self.size = self._Smoother(self.smooth, w, h, max_gap=self.max_gap), (w, h)
            self.buf.clear()
            self.next_sample = None
        for lf in self.smoother.push(None, landmarks, t):
            if lf.features is None:
                self.buf.clear()
                self.next_sample = None
                self.state = "no face"
                continue
            if self.next_sample is None:
                self.next_sample = lf.timestamp
            if lf.timestamp + 1e-6 >= self.next_sample:
                self.buf.append(lf.features)
                self.next_sample += 1.0 / self.M.sample_fps
            if len(self.buf) < self.M.num_frames:
                self.state = "collecting"
            elif self.pending is None and (self.next_pred is None or lf.timestamp + 1e-6 >= self.next_pred):
                self.pending = self.M.worker.submit(self.M.infer, np.stack(self.buf))
                self.next_pred = lf.timestamp + self.hop_s
        return new

    def _collect(self, t: float) -> bool:
        if self.pending is None or not self.pending.done():
            return False
        fut, self.pending = self.pending, None
        try:
            p = fut.result()
        except Exception as e:  # never take the bed down because of a model error
            self.state = f"model error: {e}"
            return False
        self.hist.append(p)
        self.p_avg = float(np.mean(self.hist))
        lg = _logit(self.p_avg)
        if self.base_median is None:  # building the personal baseline (first baseline_s of model scores)
            if self.base_start is None:
                self.base_start = t
            self.base_logits.append(lg)
            if self.recent_mouth:
                self.base_mouth.append(float(np.median(self.recent_mouth)))
            if t - self.base_start >= self.baseline_s:
                self.base_median = float(np.median(self.base_logits))
                self.mouth_median = float(np.median(self.base_mouth)) if self.base_mouth else None
        self.rel = None if self.base_median is None else lg - self.base_median
        self.state = "scoring" if self.base_median is not None else "learning this face"
        return True

    # ---------------------------------------------------------------- combined decision
    def decide(self, t: float, det_state: str, det_score: float, det_threshold: float) -> bool:
        """Update the combined score at a model step. Returns True when it has been above the line for
        alert_seconds (the bed then raises its ALERT)."""
        self.sw_term = 0.0 if det_state == "baselining" else det_score / det_threshold
        self.score = self.sw_term + (self.rel if self.rel is not None else 0.0)
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), 1.0)
        self.last_t = t
        if self.score >= self.line:
            self.above += dt
        else:
            self.above = 0.0
        return self.above >= self.alert_seconds

    def side(self, detector_side: str | None) -> str | None:
        """Weak side: StrokeWatch's when it has one, else the mouth corner that dropped vs the personal
        baseline (mouth_diff > 0 = patient's left lower)."""
        if detector_side:
            return detector_side
        if self.mouth_median is None or not self.recent_mouth:
            return None
        return "LEFT" if float(np.median(self.recent_mouth)) - self.mouth_median > 0 else "RIGHT"

    def status(self) -> dict:
        return {"state": self.state, "model": self.M.name, "p": self.p_avg, "relative": self.rel,
                "sw_term": round(self.sw_term, 3), "score": round(self.score, 3), "line": self.line,
                "above_seconds": round(self.above, 2),
                "baseline_progress": 1.0 if self.base_median is not None else (
                    0.0 if self.base_start is None or self.last_t is None
                    else min(1.0, (self.last_t - self.base_start) / self.baseline_s))}


def new_bed_state(analysis_fps: float, alert_seconds: float) -> CombinedDroop | None:
    return CombinedDroop(analysis_fps, alert_seconds) if _model is not None else None
