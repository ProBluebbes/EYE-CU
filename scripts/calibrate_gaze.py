"""Guided gaze calibration on a webcam: measures what YOUR eyes read as (gaze_h) when looking
straight, fully left and fully right with the head still, and suggests gaze thresholds.

    .venv/bin/python scripts/calibrate_gaze.py [--source 0] [--reps 2] [--no-voice]

Speaks the instructions (macOS `say`) and shows a preview window. Writes every frame to
clips/gaze_calibration.csv and prints a summary + suggested config values:
  gaze.deviation           ~ half of the smaller full-sideways reading
  exam.gaze_min_excursion  ~ a quarter of it
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strokewatch import ROOT, load_config  # noqa: E402
from strokewatch.vision import FaceAnalyzer, VideoSource, draw_landmarks  # noqa: E402

# The first run (Nov 14 2026) showed the eyes moving ~2 s after the prompt ENDED (prompts take
# ~1.3 s to speak), so wait 3.5 s before recording and judge each phase by its steady part.
SETTLE_S = 3.5
PHASE_S = 6.5
PROMPTS = {"center": "Look straight at the camera.", "left": "Look all the way to your left.",
           "right": "Look all the way to your right."}


class Voice:
    def __init__(self, on: bool):
        self.cmd = shutil.which("say") if on else None
        self.proc = None

    def say(self, text: str):
        print(f">> {text}")
        if self.cmd:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = subprocess.Popen([self.cmd, text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--wait", type=float, default=90, help="max seconds to wait for a face")
    args = ap.parse_args()

    cfg = load_config()
    src = VideoSource(args.source, cfg["camera"]["width"], cfg["camera"]["height"],
                      rotate=cfg["camera"].get("rotate", 0))
    an = FaceAnalyzer(cfg)
    voice = Voice(not args.no_voice)
    rows: list[dict] = []
    label, text = "waiting", "Sit facing the camera"

    def frame_step(phase: str, record: bool):
        ok, frame, t = src.read()
        if not ok:
            raise SystemExit("camera stopped delivering frames")
        ff, px = an.process(frame, t)
        if record:
            rows.append({"phase": phase, "t": round(t, 3), "gaze_h": ff.gaze_h, "yaw": ff.yaw,
                         "pitch": ff.pitch, "blink": ff.blink, "reason": ff.reason})
        if not args.no_window:
            draw_landmarks(frame, px, ff)
            g = "-" if ff.gaze_h is None else f"{ff.gaze_h:+.3f}"
            for i, line in enumerate([text, f"gaze {g}  yaw {ff.yaw if ff.yaw is not None else '-'}"]):
                cv2.putText(frame, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("StrokeWatch gaze calibration", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                raise SystemExit("cancelled")
        return ff

    # 1. wait until a face with irises is tracked steadily for 2 s
    voice.say("Gaze calibration. Please sit facing the camera.")
    t0, seen_since = time.monotonic(), None
    while True:
        ff = frame_step("wait", False)
        now = time.monotonic()
        if ff.gaze_h is not None and ff.yaw is not None and abs(ff.yaw) < 15:
            seen_since = seen_since or now
            if now - seen_since >= 2.0:
                break
        else:
            seen_since = None
        if now - t0 > args.wait:
            raise SystemExit("no face with visible eyes found in front of the camera")

    voice.say("Keep your head still, and move only your eyes.")
    end = time.monotonic() + 3.5
    while time.monotonic() < end:
        frame_step("intro", False)

    # 2. center / left / center / right, repeated
    order = []
    for _ in range(args.reps):
        order += ["center", "left", "center", "right"]
    for phase in order:
        text = PROMPTS[phase]
        voice.say(text)
        start = time.monotonic()
        while time.monotonic() - start < PHASE_S:
            frame_step(phase, time.monotonic() - start >= SETTLE_S)
    text = "Done, thank you"
    voice.say("Done. Thank you.")
    src.release()
    an.close()
    if not args.no_window:
        cv2.destroyAllWindows()

    out = ROOT / "clips" / "gaze_calibration.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows, cfg, out)


def summarize(rows: list[dict], cfg: dict, out: Path):
    print(f"\nframes: {len(rows)} -> {out}")
    vals, yaws = {}, {}
    for p in ("center", "left", "right"):
        g = [r["gaze_h"] for r in rows if r["phase"] == p and r["gaze_h"] is not None]
        y = [r["yaw"] for r in rows if r["phase"] == p and r["yaw"] is not None]
        lost = sum(1 for r in rows if r["phase"] == p and r["gaze_h"] is None)
        vals[p], yaws[p] = np.array(g), np.array(y)
        if len(g):
            print(f"{p:6s} n={len(g):3d} lost={lost:3d}  gaze median {np.median(g):+.3f}  "
                  f"p10..p90 {np.percentile(g, 10):+.3f}..{np.percentile(g, 90):+.3f}  "
                  f"sd {np.std(g):.3f}  head yaw median {np.median(y):+.1f}")
        else:
            print(f"{p:6s} no gaze samples (lost={lost})")
    if any(len(v) < 5 for v in vals.values()):
        print("\nNot enough samples in some phase: no suggestion.")
        return
    # steady extreme of each sideways phase (robust to a late or early eye movement)
    c = float(np.median(vals["center"]))
    full_l = float(np.percentile(vals["left"], 85)) - c
    full_r = c - float(np.percentile(vals["right"], 15))
    print(f"\nfull left  = {full_l:+.3f}   (should be positive)")
    print(f"full right = {full_r:+.3f}   (should be positive)")
    print(f"straight-ahead spread (p25..p75) = {np.percentile(vals['center'], 25):+.3f}.."
          f"{np.percentile(vals['center'], 75):+.3f}")
    head = max(abs(np.median(yaws['left']) - np.median(yaws['center'])),
               abs(np.median(yaws['right']) - np.median(yaws['center'])))
    if head > 8:
        print(f"WARNING: the head turned ~{head:.0f} deg with the eyes; numbers may be off.")
    if full_l <= 0 or full_r <= 0:
        print("WARNING: a direction has the wrong sign: check camera mirroring/rotation.")
        return
    small = min(full_l, full_r)
    print(f"left/right balance = {small / max(full_l, full_r):.2f} "
          f"(exam.gaze_ratio is {cfg['exam']['gaze_ratio']}; healthy should be well above it)")
    print("\nSuggested config.yaml values:")
    print(f"  gaze.deviation:          {round(small * 0.5, 3)}   (now {cfg['gaze']['deviation']})")
    print(f"  exam.gaze_min_excursion: {round(small * 0.25, 3)}   (now {cfg['exam']['gaze_min_excursion']})")


if __name__ == "__main__":
    main()
