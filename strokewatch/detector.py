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

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(self, cfg: dict):
        self.dc = cfg["detector"]
        self.reset()

    def reset(self):
        self.state = "baselining"
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
        self.base_yaw = float(np.median(self.base_pose[:, 0]))
        self.samples = []
        n = len(arr)
        self.active = [f for j, f in enumerate(FEATURES)
                       if np.sum(~np.isnan(self.base_vals[:, j])) >= MIN_FEATURE_SAMPLES]
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

    @property
    def baseline_progress(self) -> float:
        if self.state != "baselining":
            return 1.0
        return min(1.0, self.baseline_time / self.dc["baseline_seconds"])

    # ---------------------------------------------------------------- update
    def update(self, ff: FrameFeatures) -> Status:
        dt = 0.0 if self.last_t is None else min(max(ff.t - self.last_t, 0.0), self.dc["max_dt"])
        self.last_t = ff.t
        valid, reason = ff.valid, ff.reason

        if valid and self.state == "baselining":
            vals = [np.nan if getattr(ff, f) is None else getattr(ff, f) for f in FEATURES]
            self.samples.append((*self._pose(ff), *vals))
            self.baseline_time += dt
            if (self.baseline_time >= self.dc["baseline_seconds"]
                    and len(self.samples) >= self.dc["min_baseline_frames"]):
                self._finish_baseline()
        elif valid:
            skip = self._score(ff, dt)
            if skip:
                valid, reason = False, skip
        return self._status(ff.t, valid, reason)

    def _spread(self, f: str, ff: FrameFeatures) -> float:
        """Baseline spread, widened by yaw_noise[f] per 10 deg of head turn away from the
        baseline's median yaw (head-turn bias the yaw_bias table doesn't fully remove)."""
        k = (self.dc.get("yaw_noise") or {}).get(f, 0.0)
        if not k:
            return self.spread[f]
        extra = k * abs(self._pose(ff)[0] - self.base_yaw) / 10.0
        return math.hypot(self.spread[f], extra)

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
            z = self.dc["weights"][f] * (v - float(np.nanmedian(col))) / self._spread(f, ff)
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
        )


def _selftest():
    """Synthetic run: pre-existing droop + pose-dependent bias, head sweeps, then a worse droop."""
    rng = np.random.default_rng(0)
    cfg = load_config()
    # The 2.7 mm worsening below needs the tuned settings (pose matching + 0.006 floor),
    # whatever config.yaml currently has.
    cfg["detector"]["pose_match_deg"] = 3
    cfg["detector"]["min_spread"]["mouth_diff"] = 0.006
    det = Detector(cfg)
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
    assert alert_at is not None and 60 < alert_at < 75, alert_at
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
