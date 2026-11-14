"""Auto-baseline + change scoring + alert logic.

BASELINING: collect `baseline_seconds` of valid frames, remembering each frame's head
pose (yaw, pitch) and features. Pre-existing droop becomes part of the baseline.

MONITORING, pose-matched (pose_match_deg > 0, default): each frame is compared only with
baseline frames whose yaw AND pitch are within pose_match_deg. The landmark-based features
have a pose-dependent bias (one recorded head sweep, Nov 13 2026: ~1 mm of fake
mouth asymmetry per 10 deg of yaw), so matching on pose cancels it. If fewer than
min_pose_matches baseline frames are that close, the min_pose_matches nearest ones are
used instead, provided they are all within pose_max_deg. Frames farther than that from
every baseline pose are skipped ("pose not in baseline").
Centre = median of the matched baseline values; spread = robust spread of baseline
residuals around their own pose-matched medians (1.4826 * MAD), floored at min_spread.
pose_match_deg: 0 restores the original single median/MAD baseline.

yaw_noise widens the spread as the head turns away from the baseline's median yaw (by
yaw_noise[f] per 10 deg, added in quadrature), so head-turn bias that vision.yaw_bias
doesn't remove (other people / cameras) fades the score instead of raising an alert.
yaw_gain does the opposite for real changes: the landmarks follow a droop less when the
head is turned (mostly the far side), so the deviation is scaled back up by the measured
loss, using the curve for whichever side the deviation points to.

Score = max over features of |EWMA of weighted robust z|. Score >= threshold for
alert_seconds of valid time -> ALERT (held until reset). Onset = first crossing of
pre_threshold in that episode; dropping below pre_threshold ends the episode.

Run on its own for a synthetic self-test:  python -m strokewatch.detector
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import load_config
from .vision import FrameFeatures

FEATURES = ("mouth_diff", "eye_ratio", "brow_diff")
MIN_FEATURE_SAMPLES = 20
MAX_SPREAD_SAMPLES = 1500  # subsample for the O(n^2) residual-spread computation

# Per-session head-turn correction, fitted on the baseline frames (see _fit_session_bias).
SESSION_KNOTS = np.arange(-60, 61, 10.0)
SESSION_RIDGE = 0.5  # pulls each knot toward 0 like this many frames of evidence would
SESSION_SMOOTH = 5  # curvature penalty, same units
SESSION_MIN_SUPPORT = 8  # frames (hat-weighted) a knot needs to count as covered
SESSION_MIN_RANGE = 10.0  # baseline yaw must span this (5th-95th percentile) to fit at all

CALIB_PROMPTS = (
    "Calibrating. Look straight at the camera and relax your face.",
    "Slowly turn your head to one side, as far as is comfortable, and hold it there.",
    "Now slowly turn your head to the other side, and hold it there.",
    "Now look straight at the camera again.",
)
CALIB_DONE = "Calibration done. Monitoring."


@dataclass
class Status:
    t: float
    state: str  # "baselining" | "monitoring" | "ALERT"
    score: float
    threshold: float
    side: str | None
    onset_time: float | None
    baseline_progress: float
    invalid_reason: str | None
    # extras for the UI / replay
    pre_threshold: float = 0.0
    z: dict = field(default_factory=dict)
    above_seconds: float = 0.0
    alert_seconds: float = 0.0
    valid: bool = False
    # guided calibration (optional baseline with head turns)
    prompt: str | None = None
    calibrating: bool = False
    calibrated_range: list | None = None  # [min, max] baseline yaw the session fit covers

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(self, cfg: dict):
        self.dc = cfg["detector"]
        self.reset()

    def reset(self, calibrate: bool = False):
        """Start a new baseline. calibrate=True guides the patient through head turns
        (prompts in Status.prompt) so the per-session turn correction covers them."""
        self.state = "baselining"
        self.calibrating = calibrate
        self.calib_step = 0
        self.step_time = 0.0  # time spent doing what the current step asks
        self.wait_time = 0.0  # time since the current step started (for the timeout)
        self.centre_yaw: float | None = None
        self.side1_sign = 0.0
        self.prompt: str | None = CALIB_PROMPTS[0] if calibrate else None
        self.prompt_until: float | None = None
        self.session_bias: dict[str, np.ndarray] = {}
        self.calibrated_range: list | None = None
        self.samples: list[tuple] = []  # (yaw, pitch, mouth, eye, brow), NaN = missing
        self.baseline_time = 0.0
        self.base_pose: np.ndarray | None = None  # (n, 2)
        self.base_vals: np.ndarray | None = None  # (n, len(FEATURES))
        self.base_yaw = 0.0
        self.active: list[str] = []
        self.spread: dict[str, float] = {}
        self.last_t: float | None = None
        self.ewma = {f: 0.0 for f in FEATURES}
        self.score = 0.0
        self.side: str | None = None
        self.above_time = 0.0
        self.episode_onset: float | None = None
        self.onset_time: float | None = None

    # ---------------------------------------------------------------- baseline
    @staticmethod
    def _pose(ff: FrameFeatures) -> tuple[float, float]:
        return (ff.yaw or 0.0, ff.pitch or 0.0)

    def _match(self, yaw: float, pitch: float) -> np.ndarray | None:
        """Boolean mask of baseline frames with a similar head pose, or None if the pose
        is too far from anything seen during baseline.

        Uses every baseline frame within pose_match_deg (yaw and pitch). If that gives
        fewer than min_pose_matches, falls back to the min_pose_matches nearest frames,
        as long as the farthest of them is within pose_max_deg."""
        n = len(self.base_pose)
        tol = self.dc.get("pose_match_deg", 0) or 0
        if tol <= 0:
            return np.ones(n, dtype=bool)
        k = min(self.dc.get("min_pose_matches", 30), n)
        d = np.max(np.abs(self.base_pose - np.array([yaw, pitch])), axis=1)
        m = d <= tol
        if m.sum() >= k:
            return m
        near = np.argpartition(d, k - 1)[:k]
        if d[near].max() > max(tol, self.dc.get("pose_max_deg", tol)):
            return None
        m = np.zeros(n, dtype=bool)
        m[near] = True
        return m

    def _finish_baseline(self):
        arr = np.array(self.samples, dtype=float)
        self.base_pose, self.base_vals = arr[:, :2], arr[:, 2:]
        yaw = self.base_pose[:, 0]
        self.base_yaw = (self.centre_yaw if self.centre_yaw is not None
                         else float(np.median(yaw)))
        self.samples = []
        n = len(arr)
        self.active = [f for j, f in enumerate(FEATURES)
                       if np.sum(~np.isnan(self.base_vals[:, j])) >= MIN_FEATURE_SAMPLES]
        for j, f in enumerate(FEATURES):
            if f not in self.active:
                continue
            delta = self._fit_session_bias(yaw, self.base_vals[:, j], self.dc["min_spread"][f])
            if delta is not None:
                self.session_bias[f] = delta
                self.base_vals[:, j] -= np.interp(yaw, SESSION_KNOTS, delta)
        if self.session_bias:
            lo, hi = np.percentile(yaw, [2, 98])
            self.calibrated_range = [round(float(lo)), round(float(hi))]
        # Robust spread of each baseline frame around its own pose-matched median.
        idx = np.linspace(0, n - 1, min(n, MAX_SPREAD_SAMPLES)).astype(int)
        resid = {f: [] for f in self.active}
        for i in idx:
            m = self._match(*self.base_pose[i])
            if m is None:
                continue
            for j, f in enumerate(FEATURES):
                if f in resid and not np.isnan(self.base_vals[i, j]):
                    col = self.base_vals[m, j]
                    resid[f].append(self.base_vals[i, j] - np.nanmedian(col))
        for f in self.active:
            r = np.asarray(resid[f])
            mad = 1.4826 * float(np.median(np.abs(r - np.median(r)))) if len(r) >= MIN_FEATURE_SAMPLES else 0.0
            self.spread[f] = max(mad, self.dc["min_spread"][f])
        self.state = "monitoring"

    def _fit_session_bias(self, yaw: np.ndarray, v: np.ndarray, unit: float) -> np.ndarray | None:
        """Leftover head-turn bias of one feature for THIS patient and camera, learned from
        the baseline frames: a smooth curve over yaw (values at SESSION_KNOTS), 0 at the
        baseline's own angle. Only fitted if the baseline contains head turns (a guided
        calibration, or a patient who happened to look around); beyond the turns it saw,
        it holds the edge values. Corrects what the shared vision.yaw_bias curve gets wrong
        for this face/camera. None = not enough turn coverage."""
        ok = ~np.isnan(v)
        if ok.sum() < MIN_FEATURE_SAMPLES:
            return None
        y, x = v[ok] / unit, yaw[ok]
        lo, hi = np.percentile(x, [5, 95])
        if hi - lo < SESSION_MIN_RANGE:
            return None
        k = len(SESSION_KNOTS)
        eye = np.eye(k)
        basis = np.stack([np.interp(x, SESSION_KNOTS, eye[i]) for i in range(k)], axis=1)
        covered = np.flatnonzero(basis.sum(axis=0) >= SESSION_MIN_SUPPORT)
        if len(covered) < 2:
            return None
        d2 = np.diff(eye, 2, axis=0)
        A = np.vstack([np.hstack([basis, np.ones((len(y), 1))]),
                       np.hstack([eye * math.sqrt(SESSION_RIDGE), np.zeros((k, 1))]),
                       np.hstack([d2 * math.sqrt(SESSION_SMOOTH), np.zeros((len(d2), 1))])])
        b = np.concatenate([y, np.zeros(k + len(d2))])
        delta = np.linalg.lstsq(A, b, rcond=None)[0][:k] * unit
        delta[:covered[0]] = delta[covered[0]]
        delta[covered[-1] + 1:] = delta[covered[-1]]
        return delta - float(np.interp(self.base_yaw, SESSION_KNOTS, delta))

    @property
    def baseline_progress(self) -> float:
        if self.state != "baselining":
            return 1.0
        if self.calibrating:
            cc = self._calib_cfg()
            frac = {0: self.baseline_time / cc["centre_s"], 1: self.step_time / cc["hold_s"],
                    2: self.step_time / cc["hold_s"],
                    3: self.step_time / cc["centre_s"]}.get(self.calib_step, 0.0)
            return min(1.0, (self.calib_step + min(frac, 0.99)) / len(CALIB_PROMPTS))
        return min(1.0, self.baseline_time / self.dc["baseline_seconds"])

    # ---------------------------------------------------------------- calibration
    def _calib_cfg(self) -> dict:
        return {"turn_deg": 15.0, "centre_s": 5.0, "hold_s": 2.5, "step_timeout_s": 25.0,
                **(self.dc.get("calibration") or {})}

    def _calib_tick(self, ff: FrameFeatures, dt: float):
        """Advance the guided calibration: centre -> hold one side -> hold the other side ->
        centre. Only valid (still, face-tracked) frames count, so turns are learned from
        held positions, which is what the patient does between movements."""
        cc = self._calib_cfg()
        step = self.calib_step
        if step == 0:
            if self.baseline_time >= cc["centre_s"] and self.samples:
                self.centre_yaw = float(np.median([s[0] for s in self.samples]))
                self._next_step()
            return
        self.wait_time += dt
        off = (ff.yaw - self.centre_yaw) if ff.valid and ff.yaw is not None else None
        if step in (1, 2):
            other = step == 1 or self.side1_sign == 0 or (off is not None and off * self.side1_sign < 0)
            if off is not None and abs(off) >= cc["turn_deg"] and other:
                self.step_time += dt
                if step == 1:
                    self.side1_sign = math.copysign(1.0, off)
            if self.step_time >= cc["hold_s"] or self.wait_time >= cc["step_timeout_s"]:
                self._next_step()
        elif step == 3:
            if off is not None and abs(off) < 8:
                self.step_time += dt
            if self.step_time >= cc["centre_s"] or self.wait_time >= cc["step_timeout_s"]:
                self._next_step()

    def _next_step(self):
        self.calib_step += 1
        self.step_time = self.wait_time = 0.0
        self.prompt = CALIB_PROMPTS[self.calib_step] if self.calib_step < len(CALIB_PROMPTS) else None

    def _baseline_done(self) -> bool:
        if len(self.samples) < self.dc["min_baseline_frames"]:
            return False
        if self.calibrating:
            return self.calib_step >= len(CALIB_PROMPTS)
        return self.baseline_time >= self.dc["baseline_seconds"]

    # ---------------------------------------------------------------- update
    def update(self, ff: FrameFeatures) -> Status:
        dt = 0.0 if self.last_t is None else min(max(ff.t - self.last_t, 0.0), self.dc["max_dt"])
        self.last_t = ff.t
        valid, reason = ff.valid, ff.reason

        if self.prompt_until is not None and ff.t >= self.prompt_until:
            self.prompt = self.prompt_until = None
        if self.state == "baselining":
            if valid:
                vals = [np.nan if getattr(ff, f) is None else getattr(ff, f) for f in FEATURES]
                self.samples.append((*self._pose(ff), *vals))
                self.baseline_time += dt
            if self.calibrating:
                self._calib_tick(ff, dt)
            if self._baseline_done():
                self._finish_baseline()
                if self.calibrating:
                    self.prompt, self.prompt_until = CALIB_DONE, ff.t + 6.0
        elif valid:
            skip = self._score(ff, dt)
            if skip:
                valid, reason = False, skip
        return self._status(ff.t, valid, reason)

    def _spread(self, f: str, ff: FrameFeatures) -> float:
        """Baseline spread, widened by yaw_noise[f] per 10 deg of head turn away from the
        baseline's angle (head-turn bias the yaw_bias table doesn't fully remove), times
        yaw_noise_calibrated inside the range of turns a calibration covered."""
        k = (self.dc.get("yaw_noise") or {}).get(f, 0.0)
        if not k:
            return self.spread[f]
        yaw = self._pose(ff)[0]
        r = self.calibrated_range
        if f in self.session_bias and r and r[0] - 5 <= yaw <= r[1] + 5:
            k *= self.dc.get("yaw_noise_calibrated", 1.0)  # this turn was seen in the baseline
        return math.hypot(self.spread[f], k * abs(yaw - self.base_yaw) / 10.0)

    def _gain(self, f: str, ff: FrameFeatures, dev: float) -> float:
        """How much of a real change feature f still shows at this head turn, relative to
        the baseline's turn (yaw_gain table; 1 = no table). The table is for a change on
        the patient's LEFT (dev > 0); a RIGHT change uses the mirror image (-yaw)."""
        tb = self.dc.get("yaw_gain") or {}
        if f not in tb:
            return 1.0
        sgn = 1.0 if dev >= 0 else -1.0
        g = lambda yaw: float(np.interp(sgn * yaw, tb["knots"], tb[f]))
        return min(max(g(self._pose(ff)[0]) / g(self.base_yaw), 0.5), 2.0)

    def _score(self, ff: FrameFeatures, dt: float) -> str | None:
        """Update the score; returns a skip reason if the frame can't be scored."""
        m = self._match(*self._pose(ff))
        if m is None:
            return "pose not in baseline"
        a = 1.0 - math.exp(-dt / self.dc["ewma_tau_s"]) if dt > 0 else 0.0
        for j, f in enumerate(FEATURES):
            v = getattr(ff, f)
            if f not in self.active or v is None:
                continue  # e.g. blink: hold that feature's smoothed value
            col = self.base_vals[m, j]
            if np.sum(~np.isnan(col)) < 5:
                continue
            if f in self.session_bias:
                v -= float(np.interp(self._pose(ff)[0], SESSION_KNOTS, self.session_bias[f]))
            dev = v - float(np.nanmedian(col))
            z = self.dc["weights"][f] * dev / (self._spread(f, ff) * self._gain(f, ff, dev))
            self.ewma[f] += a * (z - self.ewma[f])
        dom = max(self.active, key=lambda f: abs(self.ewma[f]))
        self.score = abs(self.ewma[dom])
        current_side = "LEFT" if self.ewma[dom] > 0 else "RIGHT"

        if self.state == "ALERT":
            return None  # held until reset; score keeps updating for display
        self.side = current_side
        thr, pre = self.dc["threshold"], self.dc["pre_threshold"]
        if self.score >= pre and self.episode_onset is None:
            self.episode_onset = ff.t
        if self.score >= thr:
            self.above_time += dt
        elif self.score < pre:
            self.above_time = 0.0
            self.episode_onset = None
        if self.above_time >= self.dc["alert_seconds"]:
            self.state = "ALERT"
            self.onset_time = self.episode_onset
        return None

    def _status(self, t: float, valid: bool, reason: str | None) -> Status:
        return Status(
            t=t,
            state=self.state,
            score=round(self.score, 3),
            threshold=self.dc["threshold"],
            side=self.side if self.state != "baselining" else None,
            onset_time=self.onset_time,
            baseline_progress=round(self.baseline_progress, 3),
            invalid_reason=reason,
            pre_threshold=self.dc["pre_threshold"],
            z={f: round(self.ewma[f], 2) for f in self.active},
            above_seconds=round(self.above_time, 2),
            alert_seconds=self.dc["alert_seconds"],
            valid=valid,
            prompt=self.prompt,
            calibrating=self.calibrating and self.state == "baselining",
            calibrated_range=self.calibrated_range,
        )


def _selftest():
    """Synthetic run with the real config: pre-existing droop + a strong turn-dependent bias
    (1.6 mm per 10 deg, more than vision.yaw_bias leaves), head sweeping also during the
    baseline (so the per-session correction must learn the bias), then a worse droop."""
    rng = np.random.default_rng(0)
    det = Detector(load_config())
    fps, pre_droop, bias = 30, 0.04, -0.0018  # bias: mouth_diff per degree of yaw
    alert_at = None
    for i in range(int(90 * fps)):
        t = i / fps
        yaw = 15 * math.sin(t * 0.6)  # head keeps turning, also during baseline
        pitch = rng.normal(0, 1)
        md = (pre_droop + bias * yaw + rng.normal(0, 0.003)
              + (0.03 * min(1, (t - 60) / 3) if t > 60 else 0))  # +0.03 ≈ 2.7 mm worse
        ff = FrameFeatures(t=t, yaw=yaw, pitch=pitch, mouth_diff=md,
                           eye_ratio=rng.normal(0, 0.04), brow_diff=rng.normal(0, 0.006), valid=True)
        if 35 < t < 40:  # head turned -> invalid, wild values ignored
            ff = FrameFeatures(t=t, mouth_diff=0.3, valid=False, reason="head turned")
        st = det.update(ff)
        if st.state == "ALERT" and alert_at is None:
            alert_at = t
            print(f"ALERT at t={t:.1f}s side={st.side} onset={st.onset_time:.1f}s score={st.score:.1f}")
        if i % (10 * fps) == 0:
            print(f"t={t:5.1f} {st.state:11s} score={st.score:5.2f} progress={st.baseline_progress}")
        if st.state == "monitoring" and t < 60:
            assert st.score < det.dc["pre_threshold"], f"false episode at t={t:.1f}: {st.score}"
    assert "mouth_diff" in det.session_bias, "session bias not learned from a sweeping baseline"
    assert alert_at is not None and 60 < alert_at < 75, alert_at
    print("selftest OK")


def _selftest_calibration():
    """Guided calibration: a synthetic patient follows the prompts (centre, hold one side,
    hold the other, centre). Afterwards a neutral face held turned must stay quiet and a
    1.8 mm droop with the head turned must alert."""
    rng = np.random.default_rng(1)
    det = Detector(load_config())
    det.reset(calibrate=True)
    bias, fps, t, prompts = -0.0018, 10, 0.0, []

    def frame(yaw, droop=0.0):
        nonlocal t
        md = 0.01 + bias * yaw + droop + rng.normal(0, 0.002)
        st = det.update(FrameFeatures(t=t, yaw=yaw, pitch=0.0, mouth_diff=md, eye_ratio=rng.normal(0, 0.03),
                                      brow_diff=rng.normal(0, 0.005), valid=True))
        t += 1 / fps
        if st.prompt and (not prompts or prompts[-1] != st.prompt):
            prompts.append(st.prompt)
        return st

    target = {0: 0.0, 1: 28.0, 2: -26.0, 3: 0.0}
    yaw = 0.0
    while det.state == "baselining" and t < 120:
        goal = target.get(det.calib_step, 0.0)
        yaw += max(-2.0, min(2.0, goal - yaw))  # turn at 20 deg/s, then hold
        frame(yaw + rng.normal(0, 0.3))
    assert det.state == "monitoring", f"calibration did not finish (step {det.calib_step})"
    assert prompts[:4] == list(CALIB_PROMPTS), prompts
    print(f"calibration done at t={t:.1f}s, range {det.calibrated_range}, prompts seen {len(prompts)}")
    for _ in range(int(20 * fps)):
        st = frame(22.0 + rng.normal(0, 0.3))
        assert st.score < det.dc["pre_threshold"], f"neutral turned face scored {st.score}"
    for _ in range(int(15 * fps)):
        st = frame(22.0 + rng.normal(0, 0.3), droop=0.02)  # 0.02 ≈ 1.8 mm as tracked
    assert st.state == "ALERT", f"turned droop not detected: score {st.score}"
    print("calibration selftest OK")


if __name__ == "__main__":
    _selftest()
    _selftest_calibration()
