"""Simulated patients ("sim beds") for building/demoing the dashboard without cameras.

Synthetic FrameFeatures go through the REAL Detector (see beds.SimBed), so the states are
realistic. Each scenario is a timeline in seconds since the server started:
  - head turns and talking at intervals (frames get skipped, as with a real patient)
  - optionally a pre-existing droop (absorbed into the baseline)
  - optionally a NEW droop starting at `droop_at` (-> ALERT a few seconds later)
  - optionally a camera dropout window (bed shows "camera offline")

    python -m strokewatch.mock [--scenario predroop]     (same as server --mock)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .vision import FrameFeatures


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

    def features(self, t: float, el: float, rng: np.random.Generator) -> FrameFeatures:
        droop = self.droop(el)
        ff = FrameFeatures(t=t, yaw=float(rng.normal(0, 3)), pitch=float(rng.normal(-5, 2)),
                           roll=float(rng.normal(0, 2)), valid=True,
                           mouth_diff=droop + float(rng.normal(0, 0.006)),
                           eye_ratio=float(rng.normal(0, 0.04)),
                           brow_diff=float(rng.normal(0, 0.006)))
        if self._in(self.turns, el):
            ff.valid, ff.reason, ff.yaw, ff.mouth_diff = False, "head turned", 35.0, 0.2
        elif self._in(self.talks, el):
            ff.valid, ff.reason, ff.mouth_diff = False, "talking / mouth open", 0.1
        elif int(el * 10) % 40 == 0:
            ff.eye_ratio, ff.reason = None, "blinking (eye feature skipped)"
        return ff

    def draw(self, ff: FrameFeatures, el: float) -> np.ndarray:
        img = np.full((480, 640, 3), 40, np.uint8)
        cx = 320 + int((ff.yaw or 0) * 3)
        cv2.ellipse(img, (cx, 240), (130, 170), 0, 0, 360, (180, 200, 220), -1)
        for dx in (-50, 50):
            cv2.circle(img, (cx + dx, 200), 12, (50, 50, 50), -1)
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
    return SimScenario(label=name, droop_at=50, base_droop=base,
                       turns=[(28, 33)], talks=[(38, 43)])


def ward_scenarios(n: int) -> list[SimScenario]:
    """n varied sim patients for the dashboard. Events are staggered so beds don't sync up."""
    presets = [
        dict(label="new left droop at 75 s", droop_at=75, side=1),
        dict(label="stable"),
        dict(label="old droop, worsens at 140 s", droop_at=140, side=-1, base_droop=0.05,
             new_droop=0.05),
        dict(label="stable, talkative", talk_every=22, talk_len=9),
        dict(label="stable, camera drops out", offline=[(60, 90), (400, 420)]),
        dict(label="new right droop at 200 s", droop_at=200, side=-1),
        dict(label="stable"),
        dict(label="stable, restless", turn_every=18, turn_len=5),
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
