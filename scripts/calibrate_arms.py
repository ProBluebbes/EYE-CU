"""Arm-drift calibration on a webcam: runs the real BE-FAST arm step (same prompt, timing and
scoring as a check) for a few spoken trials and records every frame.

    .venv/bin/python scripts/calibrate_arms.py [--source 0] [--trials steady,sink_left]

Trials: steady (hold both arms still), sink_left / sink_right (let that arm slowly sink
during the hold). Writes clips/arm_calibration.csv and prints, per trial, the check's verdict
plus the arm heights (shoulder widths, + = wrist above shoulder) and eye-closure readings.
Sit so your shoulders and hands are in the picture.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strokewatch import ROOT, load_config  # noqa: E402
from strokewatch.befast import STEPS, ExamRunner, ServerVoice  # noqa: E402
from strokewatch.vision import (FaceAnalyzer, PoseAnalyzer, VideoSource, draw_landmarks,  # noqa: E402
                                draw_pose)

INTROS = {
    "steady": "Trial: steady. When asked, close your eyes and hold both arms out as still as "
              "you can, until I say lower your arms.",
    "sink_left": "Trial: weak left arm. When asked, raise both arms, then let your LEFT arm "
                 "slowly sink. Keep your right arm up until I say lower your arms.",
    "sink_right": "Trial: weak right arm. When asked, raise both arms, then let your RIGHT arm "
                  "slowly sink. Keep your left arm up until I say lower your arms.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--trials", default="steady,sink_left")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()
    trials = [t.strip() for t in args.trials.split(",") if t.strip()]
    for t in trials:
        if t not in INTROS:
            raise SystemExit(f"unknown trial {t!r}; choose from {', '.join(INTROS)}")

    cfg = load_config()
    src = VideoSource(args.source, cfg["camera"]["width"], cfg["camera"]["height"],
                      realtime=True, loop=True, rotate=cfg["camera"].get("rotate", 0))
    face, pose = FaceAnalyzer(cfg), PoseAnalyzer(cfg)
    voice = ServerVoice(not args.no_voice)
    rows, results = [], {}
    caption = ""

    def frame(trial: str | None, ex: ExamRunner | None):
        ok, img, t = src.read()
        if not ok:
            raise SystemExit("camera stopped delivering frames")
        ff, fpx = face.process(img, t)
        pf, ppx = pose.process(img, t)
        said = []
        if ex is not None:
            step = ex.step.key if ex.step else None
            el = t - ex.step_t0 if ex.step else 0.0
            said = ex.update(ff, pf if ex.needs_pose or step is None else None)
            rows.append({"trial": trial, "step": step, "el": round(el, 2), "t": round(t, 3),
                         "arm_l": pf.arm_l, "arm_r": pf.arm_r, "shoulders": pf.shoulders,
                         "blink": ff.blink, "face": ff.reason or "ok"})
        if not args.no_window:
            draw_landmarks(img, fpx, ff)
            draw_pose(img, ppx)
            lines = [caption,
                     f"L {pf.arm_l if pf.arm_l is not None else '-'}   R {pf.arm_r if pf.arm_r is not None else '-'}"
                     f"   blink {ff.blink if ff.blink is not None else '-'}"]
            for i, line in enumerate(lines):
                for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                    cv2.putText(img, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col,
                                th, cv2.LINE_AA)
            cv2.imshow("StrokeWatch arm calibration", img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                raise SystemExit("cancelled")
        return said, pf

    # wait for both shoulders in view
    caption = "Sit so shoulders and hands are visible"
    voice.say("Arm calibration. Sit facing the camera, with your shoulders and hands in view.")
    t0, since = time.monotonic(), None
    while True:
        _, pf = frame(None, None)
        now = time.monotonic()
        since = (since or now) if pf.shoulders else None
        if since and now - since > 2.0:
            break
        if now - t0 > 90:
            raise SystemExit("shoulders never came into view")

    arm_steps = [s for s in STEPS if s.key in ("arms", "arms_rest")]
    for trial in trials:
        caption = trial
        voice.say(INTROS[trial])
        end = time.monotonic() + 7.0
        while time.monotonic() < end:
            frame(None, None)
        ex = ExamRunner(cfg, f"calibration:{trial}", gaze_center=0.0)
        ex.steps = list(arm_steps)
        while not ex.finished:
            said, _ = frame(trial, ex)
            for text in said:
                if not text.startswith("That's all"):
                    voice.say(text)
        results[trial] = ex.result["components"]["arms"]
    voice.say("Done. Thank you.")
    src.release()
    face.close()
    pose.close()
    if not args.no_window:
        cv2.destroyAllWindows()

    out = ROOT / "clips" / "arm_calibration.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nframes: {len(rows)} -> {out}")
    hold = next(s for s in STEPS if s.key == "arms")
    for trial, comp in results.items():
        print(f"\n== {trial}: {comp['status']} - {comp['detail']}")
        print("   ", {k: v for k, v in comp["values"].items()})
        tr = [r for r in rows if r["trial"] == trial and r["step"] == "arms"]
        meas = [r for r in tr if r["el"] >= hold.settle]
        for side in ("l", "r"):
            v = [r[f"arm_{side}"] for r in meas if r[f"arm_{side}"] is not None]
            if v:
                print(f"    arm_{side}: n={len(v)}/{len(meas)} min {min(v):+.2f} max {max(v):+.2f} "
                      f"sd {np.std(v):.3f}")
        b = [r["blink"] for r in meas if r["blink"] is not None]
        nf = sum(1 for r in meas if r["face"] == "no face")
        if b:
            print(f"    blink during hold: median {np.median(b):.2f}, >0.5 in "
                  f"{np.mean(np.array(b) > 0.5):.0%} of {len(b)} frames; no face in {nf} frames")
        else:
            print(f"    blink: no readings (no face in {nf} frames)")


if __name__ == "__main__":
    main()
