"""BE-FAST extras on top of the facial-droop monitor: Eyes (gaze) and Arms/legs (drift).

GazeMonitor (passive, every frame): horizontal eye position from the iris landmarks the face
model already gives (vision.horizontal_gaze), relative to the patient's own baseline. Eyes held
to one side for `gaze.window_s` with little movement = "gaze deviation" (in a hemispheric
stroke the eyes deviate TOWARD the lesion, i.e. away from the weak side).

ExamRunner (active, ~40 s): a voice-guided bedside check. It starts automatically when the
face detector raises a facial-droop ALERT or the GazeMonitor flags a deviation (or when staff
press "Run BE-FAST check"). It speaks short instructions (the phone for phone beds, the bed
page in the browser for webcam beds; recorded clips from voice.py) and measures the responses:
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
from dataclasses import dataclass, replace

import numpy as np

from .localize import localize
from .vision import FrameFeatures, PoseFeatures

GAZE_DEFAULTS = {
    "baseline_samples": 150,  # gaze samples (head facing camera) for the patient's centre
    "center_max": 0.06,       # the baseline centre is clipped to this (an old deviation isn't fully absorbed)
    "window_s": 30,           # deviation must hold this long (20 s at 0.08: ~2.6 false/h on sim)
    "deviation": 0.08,        # |median gaze - centre| above this = deviated (~half of full sideways)
    "agree_fraction": 0.9,    # share of window samples on that side (= fixed, not glancing)
    "max_yaw_deg": 30,        # head turn up to which gaze samples count (see vision.gaze_max_yaw_deg)
    "max_gap_s": 10,          # no gaze samples for this long -> window restarts
}
EXAM_DEFAULTS = {
    "auto": True,             # start automatically on facial-droop ALERT / gaze deviation
    "cooldown_s": 300,        # min seconds between automatic checks for gaze deviation
    "legs": True,             # include the leg step (skipped anyway if legs aren't in view)
    "server_voice": False,    # webcam beds: the bed page speaks; True = also the server's speaker
    "alert_on_no_response": True,
    "max_gap_s": 5,           # no camera frames for this long -> check cancelled
    "min_samples": 5,
    "head_turn_deg": 8,           # head moved this far on a look step = "moved the head"
    "gaze_min_excursion": 0.04,   # neither direction beyond this = "eyes did not move"
    "gaze_full_excursion": 0.08,  # the better direction must reach this to judge at all
    "gaze_ratio": 0.45,           # weaker direction below this x stronger = palsy
    "gaze_wrong_way": 0.06,       # eyes held this far the OPPOSITE way = looked the wrong way
    "gaze_hold_tol": 0.04,        # a step's fixation = samples within +-this of one position ...
    "gaze_hold_min": 0.5,         # ... covering this share of the step, else "eyes kept moving"
    "gaze_min_coverage": 0.4,     # eyes must be tracked in this share of a step's frames
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
                      # `say`) + 2.5 s; stretched to fit a longer recorded clip (voice.py). On one person's webcam calibration (Nov 14 2026) the
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
DONE_OK = "That's all, thank you."
DONE_FLAGGED = "That's all, thank you. A nurse has been told."
STOPPED = "The check was stopped."
# Pause after a prompt ends before measuring (see Step.settle), plus playback start latency.
REACT_S, PLAY_LATENCY_S = 2.5, 0.3
# Asked only when the first try at the eye steps wasn't clean (not followed, or a possible sign):
# one try can't tell a palsy from eyes that just wandered (on sim, a patient ignoring the check
# read as "abnormal" 17% of the time), but a palsy is fixed and shows up again.
RETRY_STEPS = (
    Step("center_2", "Let's do the eyes once more. Keep your head still, and only move your "
                     "eyes. Look straight at the camera.", "Look straight at the camera", 7.0, 2.5,
         "gaze"),
    Step("look_left_2", "Without moving your head, look to your left.", "Look to your LEFT",
         4.0, 4.0, "gaze"),
    Step("look_right_2", "Now look to your right.", "Look to your RIGHT", 3.0, 4.0, "gaze"),
)
LEG_STEPS = ("legs", "legs_rest")
LIMB_STEPS = ("arms", "arms_rest", *LEG_STEPS)
COMPONENT_NAMES = {"eyes": "Eyes", "arms": "Arm drift", "legs": "Leg drift"}


GAZE_STEPS = ("center", "look_left", "look_right")
GAZE_KEYS = (*GAZE_STEPS, *(s.key for s in RETRY_STEPS))
NONCOMPLIANT = ("no_response", "inconclusive")


def fit_to_voice(st: Step) -> Step:
    """Measuring must not start before the patient has heard the whole instruction: with a
    recorded voice (often slower and calmer than macOS `say`) wait for the clip to end."""
    from .voice import seconds
    secs = seconds(st.say)
    if secs is None:
        return st
    return replace(st, settle=max(st.settle, round(secs + PLAY_LATENCY_S + REACT_S, 2)))


def _comp(status: str, detail: str, side: str | None = None, issues: list[str] | None = None,
          **values) -> dict:
    """status: normal | abnormal | no_response | inconclusive | not_assessed | skipped.
    inconclusive = the patient didn't follow the instructions well enough for a valid result
    (`issues` says how); not_assessed = the camera couldn't measure it."""
    return {"status": status, "detail": detail, "side": side, "issues": list(issues or []),
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
        self.steps = [fit_to_voice(s) for s in STEPS
                      if not (pose_error and s.key in LIMB_STEPS)
                      and not (not self.ec["legs"] and s.key in LEG_STEPS)]
        self.i = -1
        self.step_t0 = 0.0
        self.started_at: float | None = None
        self.last_t: float | None = None
        self.data: dict[str, list] = {k: [] for k in (*GAZE_KEYS, "arms", "arms_pre", "legs",
                                                      "legs_pre")}
        self.yaws: dict[str, list[float]] = {k: [] for k in GAZE_KEYS}
        # gaze-step frames while measuring, and why the eyes weren't measured in some of them
        self.gaze_frames = {k: {"frames": 0, "closed": 0, "away": 0, "lost": 0}
                            for k in GAZE_KEYS}
        self.gaze_first: tuple[dict, dict] | None = None  # first try, if it was retried
        self.blink_max = (cfg.get("vision") or {}).get("blink_max", 0.45)
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
        if self.step is not None and self.step.key == "look_right" and self.gaze_first is None:
            first = self._eval_gaze_pass("")
            if first[0]["status"] in ("inconclusive", "abnormal"):
                self.gaze_first = first
                self.steps[self.i + 1:self.i + 1] = [fit_to_voice(s) for s in RETRY_STEPS]
        while True:
            self.i += 1
            if self.i >= len(self.steps):
                self.result = self.evaluate(t)
                self.finished = True
                flagged = self.result["flagged"]
                return [DONE_FLAGGED if flagged else DONE_OK]
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
            fr = self.gaze_frames[st.key]
            fr["frames"] += 1
            if ff.yaw is not None:
                self.yaws[st.key].append(ff.yaw)
            if ff.gaze_h is not None:
                self.data[st.key].append(ff.gaze_h)
            elif ff.yaw is None:
                fr["lost"] += 1                      # no face
            elif abs(ff.yaw) > self.gc["max_yaw_deg"]:
                fr["away"] += 1                      # head turned too far for the eyes
            elif ff.blink is not None and ff.blink > self.blink_max:
                fr["closed"] += 1
            else:
                fr["lost"] += 1
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
        assessed = [c for c in comps.values()
                    if c["status"] in ("normal", "abnormal", *NONCOMPLIANT)]
        # Followed none of the instructions (possible confusion/aphasia). Needs two or more
        # parts: one part alone not followed is reported as inconclusive, not alerted. Except
        # when the eyes were the only part that could be checked at all (no pose model, limbs
        # out of view): otherwise an eyes-only check could never alert on this.
        only_part = len(assessed) == 1 and all(
            c["status"] in ("not_assessed", "skipped") for k, c in comps.items() if k != "eyes")
        no_response = ((len(assessed) >= 2 or only_part)
                       and all(c["status"] in NONCOMPLIANT for c in assessed))
        if no_response and self.ec["alert_on_no_response"]:
            findings.append("Did not follow the check's instructions "
                            "(possible confusion, aphasia or weakness)")
        sides = {c["side"] for c in comps.values() if c["status"] == "abnormal" and c["side"]}
        weak = sides.pop() if len(sides) == 1 else ("both/mixed" if sides else None)
        not_done = [COMPONENT_NAMES[k].lower() for k, c in comps.items()
                    if c["status"] in ("not_assessed", "skipped")]
        inconclusive = [k for k, c in comps.items() if c["status"] == "inconclusive"]
        notes = [f"{COMPONENT_NAMES[k]} check inconclusive, instructions not followed: "
                 f"{'; '.join(comps[k]['issues']) or comps[k]['detail']}" for k in inconclusive]
        if findings:
            summary = "; ".join(findings)
            if inconclusive and not no_response:
                summary += f"; {self._names(inconclusive)} inconclusive (instructions not followed)"
        elif inconclusive:
            summary = "; ".join(f"{COMPONENT_NAMES[k]} inconclusive, instructions not followed: "
                                f"{'; '.join(comps[k]['issues']) or comps[k]['detail']}"
                                for k in inconclusive)
            ok = [COMPONENT_NAMES[k].lower() for k, c in comps.items() if c["status"] == "normal"]
            if ok:
                summary += f". Normal: {', '.join(ok)}"
        else:
            summary = "No abnormal findings"
        res = {"trigger": self.trigger, "started_at": self.started_at, "finished_at": t,
               "components": comps, "findings": findings, "flagged": bool(findings),
               "abnormal": any(c["status"] == "abnormal" for c in comps.values()),
               "no_response": no_response, "weak_side": weak, "summary": summary,
               "not_assessed": not_done, "inconclusive": inconclusive, "notes": notes}
        # hedged "may be a right hemispheric stroke" wording (beds.py redoes it with the face)
        res["localization"] = localize(None, res)
        return res

    @staticmethod
    def _names(keys: list[str]) -> str:
        n = [COMPONENT_NAMES[k] for k in keys]
        return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]

    def _gaze_step(self, key: str) -> dict:
        """How one gaze step went: tracked enough? where did the eyes settle, and for how much
        of the step (fixation = the most samples within +-gaze_hold_tol of one position; a
        late start or one glance back at the screen still leaves most samples in it)."""
        ec, vals = self.ec, np.sort(np.array(self.data[key], dtype=float))
        fr = self.gaze_frames[key]
        frames = max(fr["frames"], len(vals))
        seen = len(vals) >= max(ec["min_samples"], ec["gaze_min_coverage"] * frames)
        why = None
        if not seen:
            miss = max(frames - len(vals), 1)
            why = max(("closed", "away", "lost"), key=lambda k: fr[k])
            if fr[why] < 0.5 * miss:
                why = "lost"
        hold, pos = 0.0, None
        if len(vals):
            ends = np.searchsorted(vals, vals + 2 * ec["gaze_hold_tol"], side="right")
            i = int(np.argmax(ends - np.arange(len(vals))))
            hold = float(ends[i] - i) / len(vals)
            pos = float(np.median(vals[i:ends[i]]))
        yaws = self.yaws[key]
        return {"n": len(vals), "frames": frames, "seen": seen, "why": why, "hold": hold,
                "pos": pos, "steady": seen and hold >= ec["gaze_hold_min"],
                "yaw": float(np.median(yaws)) if yaws else None}

    def _eval_gaze(self) -> dict:
        """One clean try decides. Otherwise (a retry was asked) a sign counts only if both
        tries show it; a first try that wasn't followed is forgiven if the second one was."""
        if self.gaze_first is None:
            return self._eval_gaze_pass("")[0]
        (c1, s1), (c2, s2) = self.gaze_first, self._eval_gaze_pass("_2")
        first = f"1st try {c1['status'].replace('_', ' ')}: {c1['detail']}"
        if c2["status"] == "normal":
            return {**c2, "detail": f"{c2['detail']} on the 2nd try ({first})"}
        if c2["status"] == "abnormal" and c1["status"] in ("abnormal", "inconclusive"):
            keep = [k for k in ("palsy", "deviation")
                    if s2.get(k) and s1.get(k) and s2[k][0] == s1[k][0]]
            if keep:
                side = s2["palsy"][0] if "palsy" in keep else other(s2["deviation"][0])
                return _comp("abnormal", "; ".join(s2[k][1] for k in keep) + " (on both tries)",
                             side, issues=c2["issues"], **c2["values"])
        both = [m for m in c1["issues"] if m in c2["issues"]]
        issues = [f"{m} (both tries)" for m in both] \
            + [f"1st try: {m}" for m in c1["issues"] if m not in both] \
            + [f"2nd try: {m}" for m in c2["issues"] if m not in both]
        if c2["status"] == "abnormal":
            why = (f"possible sign on the 2nd try only, not seen on the 1st: {c2['detail']}"
                   if c1["status"] != "abnormal" else
                   f"the two tries disagree ({first}; 2nd try: {c2['detail']})")
            return _comp("inconclusive", why, issues=issues or [why], **c2["values"])
        if c1["status"] == "abnormal":
            why = f"possible sign on the 1st try ({c1['detail']}) not confirmed, 2nd try " \
                  f"{c2['status'].replace('_', ' ')}: {c2['detail']}"
            return _comp("inconclusive", why, issues=issues or [why], **c2["values"])
        if c2["status"] == "not_assessed":
            return _comp("inconclusive", f"{first}; 2nd try: {c2['detail']}",
                         issues=issues, **c1["values"])
        return _comp("inconclusive", "instructions not followed on either try: "
                     + "; ".join(issues), issues=issues, **c2["values"])

    def _eval_gaze_pass(self, sfx: str) -> tuple[dict, dict]:
        """Gaze palsy / fixed deviation from one try (keys center{sfx}, look_left{sfx}, ...),
        but only if the patient followed it. Not following (eyes closed, head turned away,
        eyes wandering, looking the wrong way, barely moving) -> "inconclusive" with the
        reasons, never "normal" or "abnormal". Also returns the signs seen, for comparing
        tries: {"palsy": (weak side, text), "deviation": (side, text)}."""
        ec, gc = self.ec, self.gc
        st = {k: self._gaze_step(k + sfx) for k in GAZE_STEPS}
        word = {"center": "look straight at the camera", "look_left": "look left",
                "look_right": "look right"}
        side_of = {"look_left": "left", "look_right": "right"}
        hard: list[str] = []   # the check wasn't followed: no verdict at all
        soft: dict[str, list[str]] = {"center": [], "look_left": [], "look_right": []}

        # 1. Were the eyes measurable in the look steps, and if not, was that the patient's doing?
        unseen = [k for k in ("look_left", "look_right") if not st[k]["seen"]]
        for k in unseen:
            if st[k]["why"] == "closed":
                hard.append(f"kept the eyes closed when asked to {word[k]}")
            elif st[k]["why"] == "away":
                hard.append(f"turned the head away when asked to {word[k]}")
        if unseen:
            if hard:
                return _comp("inconclusive", "instructions not followed: " + "; ".join(hard),
                             issues=hard), {}
            return _comp("not_assessed", "eyes not tracked (face out of view or poor lighting)"), {}

        # 2. Reference = where the eyes settled when asked to look straight, if they settled
        # there and it isn't itself deviated from the patient's usual centre (then the eyes
        # can't look straight and the usual centre is the fairer reference).
        c = st["center"]
        mc = c["pos"] if c["steady"] else None
        if not c["seen"]:
            soft["center"].append(
                "kept the eyes closed when asked to look straight at the camera"
                if c["why"] == "closed" else "eyes not seen when asked to look straight")
        elif not c["steady"]:
            soft["center"].append(f"eyes kept moving when asked to look straight at the camera "
                                  f"(held still {c['hold']:.0%} of the time)")
        if mc is not None and (self.gaze_center is None
                               or abs(mc - self.gaze_center) < gc["deviation"]):
            ref = mc
        else:
            ref = self.gaze_center if self.gaze_center is not None else 0.0
        has_ref = mc is not None or self.gaze_center is not None
        start = mc if mc is not None else ref   # where a look step's movement starts from

        # 3. Did each look step go the asked way and stay there?
        def wrong_way() -> list[str]:
            return [k for k, sg in (("look_left", 1.0), ("look_right", -1.0))
                    if sg * (st[k]["pos"] - start) <= -ec["gaze_wrong_way"]]
        wrong = wrong_way()
        # Opposite way BOTH times = left and right mixed up, but the eyes still went one way and
        # then the other, which is all the check needs: judge each side by where the eyes
        # actually went. (Both steps to the same side stays "not followed", below.)
        swapped = len(wrong) == 2
        if swapped:
            st["look_left"], st["look_right"] = st["look_right"], st["look_left"]
            word.update(look_left="look right (looked left instead)",
                        look_right="look left (looked right instead)")
            wrong = wrong_way()
        for k in ("look_left", "look_right"):
            if k not in wrong and not st[k]["steady"]:
                hard.append(f"eyes kept moving instead of looking {side_of[k]} "
                            f"(held still {st[k]['hold']:.0%} of the time)")
        if wrong:
            k = wrong[0]
            hard.append(f"looked {side_of['look_right' if k == 'look_left' else 'look_left']} "
                        f"when asked to look {side_of[k]}")

        p = st["look_left"]["n"], st["look_right"]["n"]
        # Excursion = where the eyes stayed (the fixation), not their extreme: eyes roaming
        # around pass through a side without holding it.
        ex_l = st["look_left"]["pos"] - ref
        ex_r = ref - st["look_right"]["pos"]
        big, small = max(ex_l, ex_r), min(ex_l, ex_r)
        # moved at all between instructions? (staring at one spot, e.g. a TV, can otherwise
        # read as a fixed deviation + palsy)
        moved = max(st["look_left"]["pos"] - start, start - st["look_right"]["pos"])
        forced = None  # side the eyes stayed turned to through every instruction
        if not wrong:
            if moved < ec["gaze_min_excursion"] or big < ec["gaze_min_excursion"]:
                held = start - (self.gaze_center if self.gaze_center is not None else 0.0)
                # Eyes held to one side through every instruction is not "not following": a
                # large hemispheric stroke can lock both eyes toward the lesion (forced gaze
                # deviation, NIHSS gaze 2), often with aphasia/neglect so the patient can't
                # follow either. A sign (rare false alarm: staring at a TV, ignoring the voice).
                # Eyes stuck in the MIDDLE is no sign: still "not followed".
                off = [st[k]["pos"] - self.gaze_center for k in GAZE_STEPS
                       if self.gaze_center is not None and st[k]["seen"]]
                if len(off) == 3 and (all(d >= gc["deviation"] for d in off)
                                      or all(d <= -gc["deviation"] for d in off)):
                    forced = "LEFT" if held > 0 else "RIGHT"
                else:
                    where = f" (kept looking {'left' if held > 0 else 'right'})" \
                        if abs(held) >= gc["deviation"] else ""
                    hard.append(f"eyes stayed in the same place for every instruction{where}")
            elif big < ec["gaze_full_excursion"]:
                hard.append(f"eyes moved only slightly to either side ({big:.2f}, "
                            f"at least {ec['gaze_full_excursion']:.2f} expected)")

        # 4. Head kept still? (head turn toward the patient's left = + yaw)
        y0 = st["center"]["yaw"] if st["center"]["yaw"] is not None else 0.0
        head = {}
        for k in ("look_left", "look_right"):
            y = st[k]["yaw"]
            head[k] = None if y is None else y - y0
            if head[k] is not None and abs(head[k]) >= ec["head_turn_deg"]:
                soft[k].append(f"moved the head when asked to {word[k]} "
                               f"({abs(head[k]):.0f} deg {'left' if head[k] > 0 else 'right'})")
        vals = dict(left=ex_l, right=ex_r, reference=ref,
                    head_left=head["look_left"], head_right=head["look_right"],
                    held_center=c["hold"], held_left=st["look_left"]["hold"],
                    held_right=st["look_right"]["hold"], samples_left=p[0], samples_right=p[1],
                    swapped=swapped)
        if mc is not None and self.gaze_center is not None:
            vals["straight"] = mc - self.gaze_center
        all_soft = [m for k in GAZE_STEPS for m in soft[k]]
        if hard:
            issues = hard + all_soft
            return _comp("inconclusive", "instructions not followed: " + "; ".join(issues),
                         issues=issues, **vals), {}

        if forced:  # eyes never left one side: palsy/excursion comparisons mean nothing here
            sign = (f"eyes stayed turned to the {forced.lower()} through every instruction "
                    f"(possible forced gaze deviation)")
            note = f" (note: {'; '.join(all_soft)})" if all_soft else ""
            # same key as a deviation when looking ahead, so the two tries can confirm each other
            return _comp("abnormal", sign + note, other(forced), issues=all_soft, forced=True,
                         **vals), {"deviation": (forced, sign)}

        # 5. Verdict from a followed check. A sign must not rest on a step that wasn't followed.
        problems, side, doubts, signs = [], None, [], {}
        if small < ec["gaze_ratio"] * big:
            weak = "LEFT" if ex_l < ex_r else "RIGHT"
            k = f"look_{weak.lower()}"
            sign = (f"cannot look fully to the {weak.lower()} (moved {max(small, 0):.2f} vs "
                    f"{big:.2f} to the {other(weak).lower()})")
            signs["palsy"] = (weak, sign)
            why_not = soft[k] + ([] if has_ref else soft["center"])
            if why_not:
                doubts += [sign]
                all_soft = why_not + [m for m in all_soft if m not in why_not]
            else:
                side = weak
                problems.append(sign)
        if mc is not None and self.gaze_center is not None and abs(mc - ref) >= gc["deviation"]:
            dev = "LEFT" if mc > ref else "RIGHT"
            signs["deviation"] = (dev, f"eyes stay turned to the {dev.lower()} when asked to "
                                       f"look ahead")
            problems.append(signs["deviation"][1])
            side = side or other(dev)  # eyes deviate away from the weak side
        info = all_soft + (["mixed up left and right (looked right first), judged by where the "
                            "eyes went"] if swapped else [])
        note = f" (note: {'; '.join(info)})" if info else ""
        if problems:
            return _comp("abnormal", "; ".join(problems) + note, side, issues=all_soft, **vals), signs
        if doubts:
            return _comp("inconclusive", f"instructions not followed: {'; '.join(all_soft)}, so "
                                         f"a possible sign can't be trusted ({doubts[0]})",
                         issues=all_soft, **vals), signs
        return _comp("normal", "full eye movement to both sides" + note, issues=all_soft,
                     **vals), signs

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
    """Speaks on the server computer for webcam beds (off by default: the bed page plays the
    prompts in the browser instead). Plays the recorded clip (macOS `afplay`) if there is one,
    else macOS `say` / espeak / spd-say. A new prompt cuts off the previous one."""

    def __init__(self, enabled: bool = True):
        self.cmd = None
        self.player = shutil.which("afplay") if enabled else None
        if enabled:
            for c in ("say", "espeak-ng", "espeak", "spd-say"):
                if shutil.which(c):
                    self.cmd = c
                    break
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def say(self, text: str):
        if not text:
            return
        from .voice import clip_path
        clip = clip_path(text)
        argv = ([self.player, str(clip)] if self.player and clip and clip.exists()
                else [self.cmd, text] if self.cmd else None)
        if argv is None:
            return
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            try:
                self._proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            except OSError:
                self.cmd = self.player = None


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
    # left-sided signs (eyes can't look left) -> possible RIGHT hemispheric stroke
    assert res["localization"]["hemisphere"] == "RIGHT", res["localization"]
    print("  ->", res["localization"]["text"])
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

    # ---- gaze: only a followed check gives a verdict
    def gaze_case(center, left, right, yaws=None, closed=0, frames=None):
        e = ExamRunner(cfg, "selftest", gaze_center=0.0)
        r = np.random.default_rng(3)
        for k, v in (("center", center), ("look_left", left), ("look_right", right)):
            v = v if callable(v) else (lambda x: (lambda i: x))(v)
            e.data[k] = [v(i) + float(r.normal(0, 0.008)) for i in range(40) if v(i) is not None]
            e.gaze_frames[k].update(frames=frames or 40, closed=closed if k != "center" else 0)
            e.yaws[k] = [(yaws or {}).get(k, 0.0)] * 40
        return e._eval_gaze_pass("")[0]

    def expect(name, c, status):
        print(f"gaze {name}: {c['status']} - {c['detail']}")
        assert c["status"] == status, c

    expect("followed", gaze_case(0.0, 0.18, -0.18), "normal")
    expect("palsy left", gaze_case(0.0, 0.02, -0.18), "abnormal")
    # live run 2: looked left by turning the head, eyes stayed centred -> not a palsy
    expect("head turned instead of eyes", gaze_case(
        0.0, 0.01, -0.15, yaws={"look_left": 12.0, "look_right": -2.0}), "inconclusive")
    expect("looked the wrong way", gaze_case(0.0, -0.15, -0.18), "inconclusive")
    # left and right mixed up (right first, then left): still both directions -> judged
    expect("left/right mixed up", gaze_case(0.0, -0.17, 0.17), "normal")
    expect("mixed up, can't look left", gaze_case(0.0, -0.18, 0.07), "abnormal")
    expect("barely moved", gaze_case(0.0, 0.05, -0.05), "inconclusive")
    expect("did not move", gaze_case(0.0, 0.01, -0.01), "inconclusive")
    # eyes held to one side through every instruction: possible forced gaze deviation (a sign;
    # weak side = the other side). Eyes stuck in the middle stay "not followed" (above).
    c = gaze_case(0.12, 0.12, 0.12)
    expect("eyes held left throughout", c, "abnormal")
    assert c["side"] == "RIGHT", c
    c = gaze_case(-0.15, -0.14, -0.16)
    expect("eyes held right throughout", c, "abnormal")
    assert c["side"] == "LEFT", c
    expect("held right, then looked left", gaze_case(-0.15, 0.05, -0.16), "abnormal")
    scan = lambda i: 0.14 * np.sin(i / 2.2)  # noqa: E731  eyes sweeping around the room
    expect("eyes wandering", gaze_case(scan, scan, scan), "inconclusive")
    expect("eyes closed", gaze_case(0.0, lambda i: None, lambda i: None, closed=40),
           "inconclusive")
    expect("face lost (camera problem)", gaze_case(0.0, lambda i: None, lambda i: None),
           "not_assessed")
    # slow responder (eyes move 1.5 s into measuring) and one glance back at the screen: fine
    late = lambda x: (lambda i: 0.0 if i < 15 else x)  # noqa: E731
    expect("slow responder", gaze_case(0.0, late(0.18), late(-0.18)), "normal")
    glance = lambda i: 0.0 if 20 <= i < 26 else 0.18  # noqa: E731
    expect("glanced back once", gaze_case(0.0, glance, -0.18), "normal")
    # unsteady "look straight" + asymmetry, with a patient centre to fall back on: still a sign
    expect("unsteady centre, palsy", gaze_case(scan, 0.02, -0.18), "abnormal")

    # a patient ignoring the whole check (eyes roam, arms never move), through the real runner
    class Ignores(SimScenario):
        def gaze(self, el, rng, cue):
            return super().gaze(el, rng, None)

        def pose(self, t, el, cue, rng):
            return super().pose(t, el, None, rng)
    res, _ = _run_sim(Ignores(label="ignores"), cfg, el0=50)
    print("ignores the check:", res["summary"], "|",
          {k: c["status"] for k, c in res["components"].items()})
    assert res["components"]["eyes"]["status"] in ("inconclusive", "not_assessed"), res
    assert not res["abnormal"], res

    # forced gaze deviation through the real runner: eyes stay right whatever is asked, on
    # both tries -> eyes abnormal, weak side LEFT, alert
    class Forced(SimScenario):
        def gaze(self, el, rng, cue):
            return -0.15 + float(rng.normal(0, 0.008))
    res, said = _run_sim(Forced(label="forced"), cfg, el0=50)
    eyes = res["components"]["eyes"]
    print("forced gaze deviation:", eyes["status"], "-", eyes["detail"])
    assert eyes["status"] == "abnormal" and eyes["side"] == "LEFT", eyes
    assert "both tries" in eyes["detail"] and res["flagged"], res
    assert any(x.startswith("Let's do the eyes once more") for x in said), said

    # eyes the only part that could be checked (no pose model) and not followed: alert too
    e = ExamRunner(cfg, "selftest", gaze_center=0.0, pose_error="pose model missing")
    rr = np.random.default_rng(4)
    for k in GAZE_STEPS:
        e.data[k] = [0.14 * np.sin(i / 2.2) + float(rr.normal(0, 0.008)) for i in range(40)]
        e.gaze_frames[k].update(frames=40)
        e.yaws[k] = [0.0] * 40
    res = e.evaluate(10.0)
    print("eyes-only, not followed:", res["summary"])
    assert res["components"]["eyes"]["status"] == "inconclusive", res
    assert res["no_response"] and res["flagged"], res

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
