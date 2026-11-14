"""BE-FAST extras on top of the facial-droop monitor: Eyes (gaze) and Arms/legs (drift).

GazeMonitor (passive, every frame): horizontal eye position from the iris landmarks the face
model already gives (vision.horizontal_gaze), relative to the patient's own baseline. Eyes held
to one side for `gaze.window_s` with little movement = "gaze deviation" (in a hemispheric
stroke the eyes deviate TOWARD the lesion, i.e. away from the weak side).

ExamRunner (active, ~40 s): a voice-guided bedside check. It starts automatically when the
face detector raises a facial-droop ALERT or the GazeMonitor flags a deviation (or when staff
press "Run BE-FAST check"). It speaks short instructions (phone speaker for phone beds, this
computer's speaker for webcam beds, see ServerVoice) and measures the responses:
  look straight / left / right            -> gaze palsy (can't look to one side) or fixed deviation
  close eyes, both arms out, palms up, 10 s -> arm drift (pronator-drift style, MediaPipe Pose)
  lift both legs, 5 s                     -> leg drift (only if hips + knees are in view)
The face detector is paused during the check (deliberate eye/arm movements aren't droop).
Results go to the bed's alert (or raise one): see beds.BedManager._exam_done.

LEFT/RIGHT always = the PATIENT's side. gaze_h > 0 = looking to the patient's left.
All thresholds are provisional (synthetic data only), like the face ones were at first.

Self-test (simulated patients through the real runner):  python -m strokewatch.befast
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .vision import FrameFeatures, PoseFeatures

GAZE_DEFAULTS = {
    "baseline_samples": 150,  # gaze samples (head facing camera) for the patient's centre
    "center_max": 0.06,       # the baseline centre is clipped to this (an old deviation isn't fully absorbed)
    "window_s": 30,           # deviation must hold this long (20 s at 0.08: ~2.6 false/h on sim)
    "deviation": 0.08,        # |median gaze - centre| above this = deviated (~half of full sideways)
    "agree_fraction": 0.9,    # share of window samples on that side (= fixed, not glancing)
    "max_yaw_deg": 20,
    "max_gap_s": 10,          # no gaze samples for this long -> window restarts
}
EXAM_DEFAULTS = {
    "auto": True,             # start automatically on facial-droop ALERT / gaze deviation
    "cooldown_s": 300,        # min seconds between automatic checks for gaze deviation
    "legs": True,             # include the leg step (skipped anyway if legs aren't in view)
    "server_voice": True,     # webcam beds: speak through this computer's speaker
    "alert_on_no_response": True,
    "max_gap_s": 5,           # no camera frames for this long -> check cancelled
    "min_samples": 5,
    "gaze_min_excursion": 0.04,   # weaker of the two directions below this x stronger = palsy
    "gaze_ratio": 0.45,
    "arm_raise_gain": 0.4,        # wrist this much higher (shoulder widths) than at rest = raised
    "arm_up_min": -0.9,           # if the resting height wasn't seen: wrist at least this high
                                  # (live: raised -0.4..+1.0 by camera angle, lowered -1.3..-1.6)
    "raise_asym": 0.4,            # "one limb not raised" needs this height difference too
    "arm_drift": 0.20,            # drop over the hold (shoulder widths) ...
    "arm_drift_asym": 0.15,       # ... and this much more than the other arm
    "arm_fell": -0.9,             # ends below this while the other arm is still up = fell
    "leg_raise_gain": 0.20,       # leg height gain over rest counted as "leg raised"
    "leg_drift": 0.20,
    "leg_drift_asym": 0.15,
}


def gaze_cfg(cfg: dict) -> dict:
    return {**GAZE_DEFAULTS, **(cfg.get("gaze") or {})}


def exam_cfg(cfg: dict) -> dict:
    return {**EXAM_DEFAULTS, **(cfg.get("exam") or {})}


def other(side: str) -> str:
    return "RIGHT" if side == "LEFT" else "LEFT"


# ============================================================================ passive gaze
class GazeMonitor:
    def __init__(self, cfg: dict):
        self.gc = gaze_cfg(cfg)
        self.reset()

    def reset(self):
        self._base: list[float] = []
        self.center: float | None = None
        self.clear_window()

    def clear_window(self):
        """Forget the rolling window (after a check, so a deviation must build up again)."""
        self.win: deque[tuple[float, float]] = deque()
        self.value: float | None = None       # rolling median, relative to centre
        self.deviation: str | None = None     # "LEFT" | "RIGHT" | None
        self.since: float | None = None       # when the current deviation was first seen

    def update(self, ff: FrameFeatures):
        g, gc = ff.gaze_h, self.gc
        if g is None or ff.yaw is None or abs(ff.yaw) > gc["max_yaw_deg"]:
            return
        if self.center is None:
            self._base.append(g)
            if len(self._base) >= gc["baseline_samples"]:
                c = float(np.median(self._base))
                self.center = float(np.clip(c, -gc["center_max"], gc["center_max"]))
                self._base = []
            return
        t = ff.t
        if self.win and t - self.win[-1][0] > gc["max_gap_s"]:
            self.win.clear()
        self.win.append((t, g - self.center))
        while self.win and self.win[0][0] < t - gc["window_s"]:
            self.win.popleft()
        vals = np.array([d for _, d in self.win])
        self.value = float(np.median(vals))
        covered = t - self.win[0][0] >= 0.9 * gc["window_s"] and len(vals) >= 2 * gc["window_s"]
        side = None
        if covered and abs(self.value) >= gc["deviation"]:
            same = np.mean(np.sign(self.value) * vals >= gc["deviation"] / 2)
            if same >= gc["agree_fraction"]:
                side = "LEFT" if self.value > 0 else "RIGHT"
        if side != self.deviation:
            self.since = t if side else None
        self.deviation = side

    def status(self) -> dict:
        n = self.gc["baseline_samples"]
        return {"ready": self.center is not None,
                "progress": 1.0 if self.center is not None else round(len(self._base) / n, 3),
                "center": None if self.center is None else round(self.center, 3),
                "value": None if self.value is None else round(self.value, 3),
                "deviation": self.deviation, "since": self.since,
                "threshold": self.gc["deviation"]}


# ============================================================================ voice check
@dataclass(frozen=True)
class Step:
    key: str
    say: str          # spoken instruction
    label: str        # short on-screen text (phone screen faces the patient)
    settle: float     # seconds from the prompt to the start of measuring: speaking time (macOS
                      # `say`) + 2.5 s. On one person's webcam calibration (Nov 14 2026) the
                      # eyes moved ~2 s after the prompt ended; patients may be slower.
    measure: float = 0.0
    kind: str | None = None  # "gaze" | "arms" | "legs" | None

    @property
    def length(self) -> float:
        return self.settle + self.measure


STEPS = (
    Step("intro", "This is StrokeWatch. Please follow a few quick instructions.",
         "Quick check: please follow the instructions", 4.5),
    Step("center", "Keep your head still, and look straight at the camera.",
         "Look straight at the camera", 4.5, 2.5, "gaze"),
    Step("look_left", "Without moving your head, look to your left.",
         "Look to your LEFT", 4.0, 4.0, "gaze"),
    Step("look_right", "Now look to your right.", "Look to your RIGHT", 3.0, 4.0, "gaze"),
    Step("arms", "Close your eyes. Hold both arms straight out in front of you, palms up, "
                 "and keep them there.", "Close your eyes. Both arms out in front, palms up. Hold.",
         8.5, 10.0, "arms"),
    Step("arms_rest", "Thank you. You can lower your arms and open your eyes.",
         "Lower your arms, open your eyes", 4.0),
    Step("legs", "Now lift both legs up, and hold them there.", "Lift both legs and hold",
         5.5, 5.0, "legs"),
    Step("legs_rest", "Thank you. Lower your legs.", "Lower your legs", 2.5),
)
LEG_STEPS = ("legs", "legs_rest")
LIMB_STEPS = ("arms", "arms_rest", *LEG_STEPS)
COMPONENT_NAMES = {"eyes": "Eyes", "arms": "Arm drift", "legs": "Leg drift"}


def _comp(status: str, detail: str, side: str | None = None, **values) -> dict:
    """status: normal | abnormal | no_response | not_assessed | skipped."""
    return {"status": status, "detail": detail, "side": side,
            "values": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in values.items()}}


class ExamRunner:
    """Steps through STEPS on frame time (ff.t). Feed every frame to update(); it returns texts
    to speak. When `finished`, `result` holds the evaluation (None if cancelled)."""

    def __init__(self, cfg: dict, trigger: str, gaze_center: float | None = None,
                 pose_error: str | None = None):
        self.ec = exam_cfg(cfg)
        self.gc = gaze_cfg(cfg)
        self.trigger = trigger
        self.gaze_center = gaze_center
        self.pose_error = pose_error
        self.steps = [s for s in STEPS
                      if not (pose_error and s.key in LIMB_STEPS)
                      and not (not self.ec["legs"] and s.key in LEG_STEPS)]
        self.i = -1
        self.step_t0 = 0.0
        self.started_at: float | None = None
        self.last_t: float | None = None
        self.data: dict[str, list] = {k: [] for k in ("center", "look_left", "look_right",
                                                      "arms", "arms_pre", "legs", "legs_pre")}
        self.turned = 0           # gaze-step frames lost to a turned head
        self.arm_frames = self.legs_seen = 0
        self.skipped: set[str] = set()
        self.finished = False
        self.cancelled: str | None = None
        self.result: dict | None = None

    # ------------------------------------------------------------ state
    @property
    def step(self) -> Step | None:
        return self.steps[self.i] if 0 <= self.i < len(self.steps) else None

    @property
    def needs_pose(self) -> bool:
        st = self.step
        return not self.finished and st is not None and st.key in ("arms", "legs")

    @property
    def cue(self) -> tuple[str, float] | None:
        """(step key, seconds since its prompt) for simulated patients."""
        st = self.step
        if self.finished or st is None or self.last_t is None:
            return None
        return st.key, self.last_t - self.step_t0

    def status(self) -> dict:
        st = self.step
        if self.finished or st is None:
            return {"active": False}
        el = (self.last_t or self.step_t0) - self.step_t0
        return {"active": True, "trigger": self.trigger, "started_at": self.started_at,
                "step": st.key, "label": st.label, "say": st.say,
                "phase": "measuring" if st.kind and el >= st.settle else "listening",
                "index": self.i + 1, "count": len(self.steps),
                "progress": round(min(1.0, el / st.length), 3)}

    def cancel(self, reason: str):
        self.cancelled, self.finished, self.result = reason, True, None

    # ------------------------------------------------------------ frames
    def update(self, ff: FrameFeatures, pose: PoseFeatures | None = None) -> list[str]:
        if self.finished:
            return []
        t = ff.t
        if self.started_at is None:
            self.started_at = t
        if self.last_t is not None and t - self.last_t > self.ec["max_gap_s"]:
            self.cancel("camera frames stopped")
            return []
        self.last_t = t
        out = self._advance(t) if self.i < 0 else []
        st = self.step
        if st is None:
            return out
        el = t - self.step_t0
        if st.key == "arms" and pose is not None:
            self.arm_frames += 1
            self.legs_seen += bool(pose.legs_visible)
        if st.kind and st.settle <= el < st.length:
            self._collect(st, ff, pose)
        elif st.kind in ("arms", "legs") and el < 1.5 and pose is not None:
            self.data[f"{st.kind}_pre"].append(pose)  # still listening: limbs at rest
        if el >= st.length:
            out += self._advance(t)
        return out

    def _advance(self, t: float) -> list[str]:
        while True:
            self.i += 1
            if self.i >= len(self.steps):
                self.result = self.evaluate(t)
                self.finished = True
                flagged = self.result["flagged"]
                return ["That's all, thank you." + (" A nurse has been told." if flagged else "")]
            st = self.steps[self.i]
            if st.key in LEG_STEPS and not self._legs_in_view():
                self.skipped.add(st.key)
                continue
            self.step_t0 = t
            return [st.say]

    def _legs_in_view(self) -> bool:
        return self.arm_frames > 0 and self.legs_seen >= 0.5 * self.arm_frames

    def _collect(self, st: Step, ff: FrameFeatures, pose: PoseFeatures | None):
        if st.kind == "gaze":
            if ff.gaze_h is not None:
                self.data[st.key].append(ff.gaze_h)
            elif ff.reason == "head turned":
                self.turned += 1
        elif st.kind == "arms":
            self.data["arms"].append((ff.t, pose.arm_l if pose else None,
                                      pose.arm_r if pose else None, ff.blink))
        elif st.kind == "legs" and pose is not None:
            self.data["legs"].append(pose)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, t: float) -> dict:
        comps = {"eyes": self._eval_gaze()}
        if self.pose_error:
            comps["arms"] = _comp("not_assessed", f"no pose analysis ({self.pose_error})")
            comps["legs"] = _comp("not_assessed", f"no pose analysis ({self.pose_error})")
        else:
            comps["arms"] = self._eval_arms()
            if not self.ec["legs"]:
                comps["legs"] = _comp("skipped", "leg step turned off (exam.legs)")
            elif "legs" in self.skipped:
                comps["legs"] = _comp("skipped", "legs not in view of the camera")
            else:
                comps["legs"] = self._eval_legs()

        findings = [f"{COMPONENT_NAMES[k]}: {c['detail']}" for k, c in comps.items()
                    if c["status"] == "abnormal"]
        assessed = [c for c in comps.values() if c["status"] in ("normal", "abnormal", "no_response")]
        no_response = bool(assessed) and all(c["status"] == "no_response" for c in assessed)
        if no_response and self.ec["alert_on_no_response"]:
            findings.append("Did not follow the check's instructions "
                            "(possible confusion, aphasia or weakness)")
        sides = {c["side"] for c in comps.values() if c["status"] == "abnormal" and c["side"]}
        weak = sides.pop() if len(sides) == 1 else ("both/mixed" if sides else None)
        not_done = [COMPONENT_NAMES[k].lower() for k, c in comps.items()
                    if c["status"] in ("not_assessed", "skipped")]
        summary = "; ".join(findings) if findings else "No abnormal findings"
        return {"trigger": self.trigger, "started_at": self.started_at, "finished_at": t,
                "components": comps, "findings": findings, "flagged": bool(findings),
                "abnormal": any(c["status"] == "abnormal" for c in comps.values()),
                "no_response": no_response, "weak_side": weak, "summary": summary,
                "not_assessed": not_done}

    def _eval_gaze(self) -> dict:
        ec, n = self.ec, self.ec["min_samples"]
        d = self.data
        if len(d["look_left"]) < n or len(d["look_right"]) < n:
            why = "head turned instead of the eyes" if self.turned >= n else \
                "eyes not tracked (face out of view or eyes closed)"
            return _comp("not_assessed", why)
        measured_center = float(np.median(d["center"])) if len(d["center"]) >= n else None
        ref = self.gaze_center if self.gaze_center is not None else (measured_center or 0.0)
        ex_l = float(np.percentile(d["look_left"], 80)) - ref
        ex_r = ref - float(np.percentile(d["look_right"], 20))
        vals = dict(left=ex_l, right=ex_r, reference=ref)
        if measured_center is not None:
            vals["straight"] = measured_center - ref
        big, small = max(ex_l, ex_r), min(ex_l, ex_r)
        if big < ec["gaze_min_excursion"]:
            return _comp("no_response", "eyes did not move to either side", **vals)
        problems, side = [], None
        if small < ec["gaze_ratio"] * big:
            side = "LEFT" if ex_l < ex_r else "RIGHT"
            problems.append(f"cannot look fully to the {side.lower()} "
                            f"(moved {max(small, 0):.2f} vs {big:.2f} to the {other(side).lower()})")
        if self.gaze_center is not None and measured_center is not None \
                and abs(measured_center - ref) >= self.gc["deviation"]:
            dev = "LEFT" if measured_center > ref else "RIGHT"
            problems.append(f"eyes stay turned to the {dev.lower()} when asked to look ahead")
            side = side or other(dev)  # eyes deviate away from the weak side
        if problems:
            return _comp("abnormal", "; ".join(problems), side, **vals)
        return _comp("normal", "full eye movement to both sides", **vals)

    @staticmethod
    def _start_end(series: list[tuple[float, float]]) -> tuple[float, float]:
        """Start = highest 2-s median in the first 4 s (people are often still raising the limb
        when measuring starts: seen on the first live run, Nov 14 2026, so a rise-then-sink
        would net to zero; 2-s windows keep the max of noisy medians from inflating the
        start). End = median of the last 2 s. Short series: first/last third."""
        ts = np.array([t for t, _ in series])
        hs = np.array([h for _, h in series])
        last = ts >= ts[-1] - 2.0
        early = ts <= ts[0] + 4.0
        if early.sum() < 3 or last.sum() < 3:
            k = max(1, len(hs) // 3)
            return float(np.median(hs[:k])), float(np.median(hs[-k:]))
        peaks = [np.median(hs[(ts >= t) & (ts <= t + 2.0)]) for t in ts[early]
                 if t + 2.0 <= ts[-1] and ((ts >= t) & (ts <= t + 2.0)).sum() >= 3]
        start = float(max(peaks)) if peaks else float(np.median(hs[early]))
        return start, float(np.median(hs[last]))

    def _eval_arms(self) -> dict:
        rows, ec = self.data["arms"], self.ec
        n_min = max(ec["min_samples"], 0.4 * len(rows))
        sl = [(t, l) for t, l, _, _ in rows if l is not None]
        sr = [(t, r) for t, _, r, _ in rows if r is not None]
        blinks = [b for *_, b in rows if b is not None]
        eyes = "" if not blinks or np.mean(np.array(blinks) > 0.5) >= 0.5 else " (eyes stayed open)"
        return self._limb_verdict("arm", sl, sr, n_min, eyes, rest=self._rest("arms", "arm", "arm"),
                                  gain=ec["arm_raise_gain"], up_min=ec["arm_up_min"],
                                  drift=ec["arm_drift"], asym=ec["arm_drift_asym"],
                                  fell=ec["arm_fell"], unit="shoulder widths")

    def _rest(self, kind: str, joint_l: str, joint_r: str) -> tuple[float | None, float | None]:
        """Median limb height while the patient was still hearing the prompt."""
        pre = self.data[f"{kind}_pre"]

        def med(attr):
            v = [getattr(p, attr) for p in pre if getattr(p, attr) is not None]
            return float(np.median(v)) if v else None
        return med(f"{joint_l}_l"), med(f"{joint_r}_r")

    def _eval_legs(self) -> dict:
        rows, ec = self.data["legs"], self.ec
        n_min = max(ec["min_samples"], 0.4 * len(rows))

        def series(rows, side):  # ankles if mostly visible, else knees
            a = [(p.t, getattr(p, f"leg_{side}")) for p in rows if getattr(p, f"leg_{side}") is not None]
            k = [(p.t, getattr(p, f"knee_{side}")) for p in rows if getattr(p, f"knee_{side}") is not None]
            return (a, "leg") if len(a) >= 0.6 * max(len(rows), 1) else (k, "knee")

        (sl, jl), (sr, jr) = series(rows, "l"), series(rows, "r")
        return self._limb_verdict("leg", sl, sr, n_min, "", rest=self._rest("legs", jl, jr),
                                  gain=ec["leg_raise_gain"], up_min=None, drift=ec["leg_drift"],
                                  asym=ec["leg_drift_asym"], fell=None, unit="body widths")

    def _limb_verdict(self, limb, sl, sr, n_min, note, rest, gain, up_min, drift, asym, fell,
                      unit) -> dict:
        vis_l, vis_r = len(sl) >= n_min, len(sr) >= n_min
        if not (vis_l or vis_r):
            return _comp("not_assessed", f"{limb}s not in view of the camera")
        if not (vis_l and vis_r):
            seen = "left" if vis_l else "right"
            return _comp("not_assessed", f"only the {seen} {limb} was in view")
        jumpy = [side for side, ser in (("left", sl), ("right", sr)) if self._jumpy(ser)]
        if jumpy:
            return _comp("not_assessed", f"{limb} positions jumped around ({' and '.join(jumpy)}): "
                                         f"tracking unreliable, or the {limb}s kept moving{note}")
        s_l, e_l = self._start_end(sl)
        s_r, e_r = self._start_end(sr)
        vals = dict(start_left=s_l, start_right=s_r, end_left=e_l, end_right=e_r,
                    drop_left=s_l - e_l, drop_right=s_r - e_r)
        vals.update(rest_left=rest[0], rest_right=rest[1])

        def raised(start, r):  # higher than at rest; if rest wasn't seen: plainly up (arms ~ 0)
            if r is not None:
                return start - r >= gain
            return up_min is None or start >= up_min
        up_l, up_r = raised(s_l, rest[0]), raised(s_r, rest[1])
        if not (up_l or up_r):
            return _comp("no_response", f"{limb}s were not raised", **vals)
        if up_l != up_r:
            if abs(s_l - s_r) >= self.ec["raise_asym"]:
                weak = "LEFT" if not up_l else "RIGHT"
                return _comp("abnormal", f"{weak.lower()} {limb} not raised (the other was){note}",
                             weak, **vals)
            # borderline: both about as high -> judge them on drift
        d_l, d_r = s_l - e_l, s_r - e_r
        # One-sided drift needs the OTHER limb to have stayed up (dropped less than half as
        # much): on the first live run both arms were lowered together near the end and the
        # left merely dropped a little further, which isn't a one-sided sign.
        flag_l = (d_l >= drift and d_l - d_r >= asym and d_r < 0.5 * d_l) or \
            (fell is not None and e_l < fell and e_r > fell + 0.5)
        flag_r = (d_r >= drift and d_r - d_l >= asym and d_l < 0.5 * d_r) or \
            (fell is not None and e_r < fell and e_l > fell + 0.5)
        if flag_l != flag_r:
            weak = "LEFT" if flag_l else "RIGHT"
            dw, do = (d_l, d_r) if flag_l else (d_r, d_l)
            return _comp("abnormal", f"{weak.lower()} {limb} drifted down "
                                     f"({dw:.2f} vs {do:.2f} {unit} in {self._hold(limb)} s){note}",
                         weak, **vals)
        if d_l >= 1.0 and d_r >= 1.0:
            return _comp("not_assessed", f"both {limb}s were lowered before the end of the hold"
                                         f"{note}", **vals)
        if d_l >= drift and d_r >= drift:
            return _comp("normal", f"both {limb}s sank about equally (tiredness?){note}", **vals)
        return _comp("normal", f"both {limb}s held steady{note}", **vals)

    @staticmethod
    def _jumpy(series: list[tuple[float, float]]) -> bool:
        """True if the per-second medians make big moves (> 0.3) that reverse direction twice
        or more, i.e. up-down-up (pose tracking flipping, seen on the first live run with arms
        pointing at a laptop camera). A steady hold, a late raise, a slow sink or a single drop
        all pass."""
        if len(series) < 6:
            return False
        ts = np.array([t for t, _ in series])
        hs = np.array([h for _, h in series])
        meds = []
        for t in np.arange(ts[0], ts[-1] + 1e-9, 1.0):
            m = (ts >= t) & (ts < t + 1.0)
            if m.sum() >= 3:
                meds.append(float(np.median(hs[m])))
        if len(meds) < 3:
            return False
        big = [np.sign(d) for d in np.diff(meds) if abs(d) > 0.3]
        reversals = sum(1 for a, b in zip(big, big[1:]) if a != b)
        return reversals >= 2

    @staticmethod
    def _hold(limb: str) -> int:
        return int(next(s.measure for s in STEPS if s.key == f"{limb}s"))


# ============================================================================ speaker
class ServerVoice:
    """Speaks on this computer (macOS `say`, else espeak/spd-say) for webcam beds.
    A new prompt cuts off the previous one."""

    def __init__(self, enabled: bool = True):
        self.cmd = None
        if enabled:
            for c in ("say", "espeak-ng", "espeak", "spd-say"):
                if shutil.which(c):
                    self.cmd = c
                    break
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def say(self, text: str):
        if not self.cmd or not text:
            return
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            try:
                self._proc = subprocess.Popen([self.cmd, text], stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            except OSError:
                self.cmd = None


# ============================================================================ self-test
def _run_sim(scenario, cfg, el0: float, fps: float = 10, gaze_center: float = 0.0):
    rng = np.random.default_rng(1)
    ex = ExamRunner(cfg, "selftest", gaze_center=gaze_center)
    said, el = [], el0
    while not ex.finished and el < el0 + 120:
        cue = ex.cue
        ff = scenario.features(1000 + el, el, rng, cue)
        pose = scenario.pose(1000 + el, el, cue, rng) if ex.needs_pose else None
        said += ex.update(ff, pose)
        el += 1 / fps
    return ex.result, said


def _selftest():
    from . import load_config
    from .mock import SimScenario

    cfg = load_config()
    # normal patient: everything normal, legs skipped (sim legs aren't in view)
    res, said = _run_sim(SimScenario(label="normal"), cfg, el0=50)
    print("normal:", res["summary"], "|", {k: c["status"] for k, c in res["components"].items()})
    assert not res["flagged"], res
    assert res["components"]["eyes"]["status"] == "normal"
    assert res["components"]["arms"]["status"] == "normal"
    assert res["components"]["legs"]["status"] == "skipped"
    assert said[0].startswith("This is StrokeWatch") and said[-1] == "That's all, thank you."

    # new LEFT weakness: eyes deviate RIGHT, can't look left, left arm drifts
    sc = SimScenario(label="left stroke", side=1, gaze_dev_at=10, arm_weak_at=10)
    res, said = _run_sim(sc, cfg, el0=50)
    print("left stroke:", res["summary"], "| weak side", res["weak_side"])
    assert res["components"]["eyes"]["status"] == "abnormal", res["components"]["eyes"]
    assert res["components"]["arms"]["status"] == "abnormal", res["components"]["arms"]
    assert res["components"]["arms"]["side"] == "LEFT" and res["weak_side"] == "LEFT"
    assert said[-1].endswith("A nurse has been told.")

    # late raiser: both arms still rising when measuring starts, then the left sinks
    ex = ExamRunner(cfg, "selftest", gaze_center=0.0)
    ex.data["arms_pre"] = [PoseFeatures(t=0, arm_l=-1.5, arm_r=-1.5)]
    for i in range(100):
        t = i / 10
        rise = min(0.3, -1.0 + t)            # reaches the top at ~1.3 s
        left = rise - max(0.0, t - 3) * 0.1  # then the left sinks 0.7
        ex.data["arms"].append((t, left, rise, 0.9))
    arms = ex._eval_arms()
    print("late raiser:", arms["status"], "-", arms["detail"])
    assert arms["status"] == "abnormal" and arms["side"] == "LEFT", arms

    # patterns from the first live run: flipping tracking, and both arms lowered together
    def arms_case(left, right):
        e = ExamRunner(cfg, "selftest", gaze_center=0.0)
        e.data["arms_pre"] = [PoseFeatures(t=0, arm_l=-1.6, arm_r=-1.6)]
        e.data["arms"] = [(i / 30, left(i / 30), right(i / 30), 0.8) for i in range(300)]
        return e._eval_arms()
    flip = lambda t: 0.3 if int(t) % 3 else -1.0  # noqa: E731
    c = arms_case(flip, flip)
    print("flipping tracking:", c["status"], "-", c["detail"])
    assert c["status"] == "not_assessed", c
    both_down = lambda off: (lambda t: (1.0 - off) if t < 6.5 else -1.3)  # noqa: E731
    c = arms_case(both_down(0.2), both_down(0.0))
    print("both lowered early:", c["status"], "-", c["detail"])
    assert c["status"] != "abnormal", c

    # no response at all (e.g. confused/aphasic, or a photo): flagged as such
    ex = ExamRunner(cfg, "selftest", gaze_center=0.0)
    t = 0.0
    while not ex.finished:
        ex.update(FrameFeatures(t=t, yaw=0.0, gaze_h=0.01, blink=0.05, valid=True),
                  PoseFeatures(t=t, arm_l=-0.45, arm_r=-0.6, shoulders=True))
        t += 0.1
    res = ex.result
    print("no response:", res["summary"])
    assert res["no_response"] and res["flagged"] and not res["abnormal"], res

    # passive monitor: deviation flagged after the window, not before
    gm = GazeMonitor(cfg)
    rng = np.random.default_rng(2)
    healthy, deviated = SimScenario(label="healthy"), SimScenario(label="dev", side=1, gaze_dev_at=0)
    flagged_at = None
    for i in range(int(130 * 10)):
        el = i / 10
        gm.update((healthy if el < 60 else deviated).features(el, el, rng))
        if gm.deviation and flagged_at is None:
            flagged_at = el
            assert gm.deviation == "RIGHT", gm.deviation
        if el < 60:
            assert gm.deviation is None, (el, gm.status())
    print(f"gaze deviation flagged at {flagged_at:.1f}s (deviation from 60 s)")
    assert flagged_at is not None and 60 < flagged_at < 65 + gm.gc["window_s"]
    print("befast selftest OK")


if __name__ == "__main__":
    _selftest()
