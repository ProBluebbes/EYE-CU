"""Simulated patients ("sim beds") for building/demoing the dashboard without cameras.

Synthetic FrameFeatures go through the REAL Detector (see beds.SimBed), so the states are
realistic. Each scenario is a timeline in seconds since the server started:
  - head turns and talking at intervals (frames get skipped, as with a real patient)
  - optionally a pre-existing droop (absorbed into the baseline)
  - optionally a NEW droop starting at `droop_at` (-> ALERT a few seconds later)
  - optionally a camera dropout window (bed shows "camera offline")
  - optionally eyes fixed to one side (gaze_dev_at) and a weak arm (arm_weak_at), on the side
    given by `side` (eyes deviate AWAY from the weak side). Simulated patients follow the
    BE-FAST check's voice instructions (cue), so the check can be demoed without a camera.

    python -m strokewatch.mock [--scenario predroop]     (same as server --mock)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .vision import FrameFeatures, PoseFeatures


def periodic(first: float, every: float, length: float, until: float = 4 * 3600
             ) -> list[tuple[float, float]]:
    return [(s, s + length) for s in np.arange(first, until, every)]


@dataclass
class SimScenario:
    label: str
    droop_at: float | None = None   # seconds after start; None = never
    side: int = 1                   # +1 = patient's LEFT droops, -1 = RIGHT
    base_droop: float = 0.0         # pre-existing droop, present from the start
    new_droop: float = 0.07         # extra droop added at droop_at (ramps over 4 s)
    turns: list = field(default_factory=list)    # [(start, end)] head turned
    talks: list = field(default_factory=list)    # [(start, end)] talking
    offline: list = field(default_factory=list)  # [(start, end)] no camera frames
    gaze_dev_at: float | None = None  # eyes held toward the lesion (away from `side`) from here
    arm_weak_at: float | None = None  # the `side` arm is weak from here (drifts / can't lift)
    ignores_eye_check: bool = False   # keeps staring at one spot (a TV) through the eye steps
    _gaze: float = field(default=0.0, repr=False)   # current fixation point (wanders)
    _pose: PoseFeatures | None = field(default=None, repr=False)

    @staticmethod
    def _in(windows, el) -> bool:
        return any(a < el < b for a, b in windows)

    def is_offline(self, el: float) -> bool:
        return self._in(self.offline, el)

    def droop(self, el: float) -> float:
        d = self.base_droop
        if self.droop_at is not None and el > self.droop_at:
            d += self.new_droop * min(1.0, (el - self.droop_at) / 4)
        return self.side * d

    def _after(self, at: float | None, el: float) -> bool:
        return at is not None and el > at

    def gaze(self, el: float, rng: np.random.Generator, cue: tuple[str, float] | None) -> float:
        """Horizontal gaze (+ = patient's left). Healthy eyes wander between fixations;
        deviated eyes stay toward the lesion and can't cross far to the weak side."""
        dev = self._after(self.gaze_dev_at, el)
        key, ce = cue if cue else (None, 0.0)
        key = key.removesuffix("_2") if key else key  # the eye steps' second try
        toward = -self.side  # lesion side, as a gaze sign
        if self.ignores_eye_check and key in ("center", "look_left", "look_right"):
            return 0.12 + float(rng.normal(0, 0.01))
        if key in ("center", "look_left", "look_right") and ce > 1.6:
            want = {"center": 0.0, "look_left": 0.18, "look_right": -0.18}[key]
            if dev:  # can reach the lesion side; barely gets past midline the other way
                want = {"center": toward * 0.11, "look_left": 0.18, "look_right": -0.18}[key]
                if np.sign(want) == -toward:
                    want = -toward * 0.02
            return want + float(rng.normal(0, 0.01))
        if dev:
            return toward * 0.16 + float(rng.normal(0, 0.012))
        if rng.random() < 0.04:  # a new fixation every ~2.5 s at 10 fps
            self._gaze = float(np.clip(rng.normal(0, 0.06), -0.15, 0.15))
        return self._gaze + float(rng.normal(0, 0.01))

    def features(self, t: float, el: float, rng: np.random.Generator,
                 cue: tuple[str, float] | None = None) -> FrameFeatures:
        droop = self.droop(el)
        ff = FrameFeatures(t=t, yaw=float(rng.normal(0, 3)), pitch=float(rng.normal(-5, 2)),
                           roll=float(rng.normal(0, 2)), valid=True,
                           mouth_diff=droop + float(rng.normal(0, 0.006)),
                           eye_ratio=float(rng.normal(0, 0.04)),
                           brow_diff=float(rng.normal(0, 0.006)),
                           gaze_h=round(self.gaze(el, rng, cue), 4), blink=0.05)
        if cue and cue[0] == "arms" and cue[1] > 1.5:  # following "close your eyes"
            ff.gaze_h, ff.blink, ff.eye_ratio = None, 0.9, None
        if self._in(self.turns, el) and not cue:
            ff.valid, ff.reason, ff.yaw, ff.mouth_diff, ff.gaze_h = False, "head turned", 35.0, 0.2, None
        elif self._in(self.talks, el) and not cue:
            ff.valid, ff.reason, ff.mouth_diff = False, "talking / mouth open", 0.1
        elif int(el * 10) % 40 == 0:
            ff.eye_ratio, ff.gaze_h, ff.blink = None, None, 0.8
            ff.reason = "blinking (eye feature skipped)"
        return ff

    def pose(self, t: float, el: float, cue: tuple[str, float] | None,
             rng: np.random.Generator) -> PoseFeatures:
        """Arm heights (shoulder widths, wrist above shoulder)."""
        key, ce = cue if cue else (None, 0.0)
        heights = {}
        for s, sign in (("l", 1), ("r", -1)):
            h = -1.6  # hands resting in the lap
            if key == "arms" and ce > 2.0:
                lift = min(1.0, (ce - 2.0) / 2.0)
                weak = self._after(self.arm_weak_at, el) and sign == self.side
                top = -0.15 if weak else 0.05
                h = -1.6 + lift * (top + 1.6)
                if weak and ce > 8.0:  # pronator drift: slowly sinks with eyes closed
                    h = max(-1.4, top - 0.07 * (ce - 8.0))
            heights[s] = h + float(rng.normal(0, 0.03))
        self._pose = PoseFeatures(t=t, arm_l=round(heights["l"], 4), arm_r=round(heights["r"], 4),
                                  shoulders=True)
        return self._pose

    def draw(self, ff: FrameFeatures, el: float, cue: tuple[str, float] | None = None
             ) -> np.ndarray:
        img = np.full((480, 640, 3), 40, np.uint8)
        cx = 320 + int((ff.yaw or 0) * 3)
        cv2.ellipse(img, (cx, 240), (130, 170), 0, 0, 360, (180, 200, 220), -1)
        # image right = patient's left; + gaze moves the pupils toward image right
        gx = int((ff.gaze_h or 0.0) * 70)
        for dx in (-50, 50):
            if ff.blink is not None and ff.blink > 0.5:
                cv2.line(img, (cx + dx - 14, 200), (cx + dx + 14, 200), (50, 50, 50), 3, cv2.LINE_AA)
            else:
                cv2.ellipse(img, (cx + dx, 200), (20, 11), 0, 0, 360, (245, 245, 245), -1)
                cv2.circle(img, (cx + dx + gx, 200), 8, (50, 50, 50), -1)
        if cue and cue[0] == "arms" and self._pose is not None:
            # hands: image right = patient's left; y = shoulder line - height
            for h, x in ((self._pose.arm_l, cx + 175), (self._pose.arm_r, cx - 175)):
                if h is not None:
                    y = int(330 - h * 110)
                    cv2.line(img, (x, 470), (x, y), (150, 170, 190), 18, cv2.LINE_AA)
                    cv2.circle(img, (x, y), 18, (180, 200, 220), -1, cv2.LINE_AA)
        # image right = patient's left; a +droop pulls that mouth corner down
        mouth_y, d = 310, int(self.droop(el) * 600)
        if ff.reason and ff.reason.startswith("talking"):
            cv2.ellipse(img, (cx, mouth_y), (45, 22), 0, 0, 360, (40, 40, 120), -1)
        else:
            cv2.line(img, (cx - 50, mouth_y + max(-d, 0)), (cx + 50, mouth_y + max(d, 0)),
                     (40, 40, 120), 6, cv2.LINE_AA)
        cv2.putText(img, f"SIMULATED PATIENT: {self.label}", (10, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return img


def single_scenario(name: str) -> SimScenario:
    """The original one-patient mock (server --mock): droop at ~50 s -> alert at ~55 s."""
    base = 0.05 if name == "predroop" else 0.0
    return SimScenario(label=name, droop_at=50, base_droop=base, arm_weak_at=50,
                       turns=[(28, 33)], talks=[(38, 43)])


def ward_scenarios(n: int) -> list[SimScenario]:
    """n varied sim patients for the dashboard. Events are staggered so beds don't sync up."""
    presets = [
        dict(label="new left droop at 75 s", droop_at=75, side=1, arm_weak_at=75),
        dict(label="stable"),
        dict(label="old droop, worsens at 140 s", droop_at=140, side=-1, base_droop=0.05,
             new_droop=0.05),
        dict(label="stable, talkative", talk_every=22, talk_len=9),
        dict(label="stable, camera drops out", offline=[(60, 90), (400, 420)]),
        dict(label="new right droop at 200 s", droop_at=200, side=-1, arm_weak_at=200,
             gaze_dev_at=200),
        dict(label="eyes to right + left arm weak at 110 s, face normal", side=1,
             gaze_dev_at=110, arm_weak_at=110),
        dict(label="stable, restless, won't follow the eye check", turn_every=18, turn_len=5,
             ignores_eye_check=True),
    ]
    out = []
    for i in range(n):
        p = dict(presets[i % len(presets)])
        lap = i // len(presets)  # beyond 8 beds: same presets, later droops
        if p.get("droop_at") is not None:
            p["droop_at"] += 90 * lap
        talk_every, talk_len = p.pop("talk_every", 60), p.pop("talk_len", 6)
        turn_every, turn_len = p.pop("turn_every", 45), p.pop("turn_len", 4)
        phase = 7.0 * i
        # first events after the 20 s baseline would normally finish, so baselines build fast
        p["talks"] = periodic(30 + phase % talk_every, talk_every, talk_len)
        p["turns"] = periodic(25 + (phase * 1.7) % turn_every, turn_every, turn_len)
        out.append(SimScenario(**p))
    return out


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "--mock")
    from .server import main

    main()
