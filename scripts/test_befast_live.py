"""Live end-to-end test of the BE-FAST eyes + arms parts on a webcam, with spoken trials.

    .venv/bin/python scripts/test_befast_live.py [--source 0] [--trials normal,weak_left,hold_right]

Trials (each uses the real GazeMonitor / ExamRunner, same prompts, timing and scoring):
  normal      follow the whole check normally (expect: eyes normal, arms normal)
  weak_left   act out a left-sided stroke: barely look left, let the left arm sink
              (expect: eyes abnormal + left arm drift, weak side LEFT)
  weak_right  the same on the right
  hold_right  passive gaze watch: hold the eyes far right, head still (expect: flagged ~30 s)
  hold_left   the same to the left
Writes clips/befast_live.csv (every frame) and prints each trial's verdict.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strokewatch import ROOT, load_config  # noqa: E402
from strokewatch.befast import ExamRunner, GazeMonitor, ServerVoice  # noqa: E402
from strokewatch.vision import (FaceAnalyzer, PoseAnalyzer, VideoSource, draw_landmarks,  # noqa: E402
                                draw_pose)

INTROS = {
    "normal": "Trial: normal. Follow the check normally. Close your eyes when asked, and "
              "hold both arms as still as you can until I say lower your arms.",
    "weak_left": "Trial: weak left side. When asked to look left, move your eyes only a "
                 "little. When holding your arms, let your LEFT arm slowly sink, and keep "
                 "your right arm up until I say lower your arms.",
    "weak_right": "Trial: weak right side. When asked to look right, move your eyes only a "
                  "little. When holding your arms, let your RIGHT arm slowly sink, and keep "
                  "your left arm up until I say lower your arms.",
    "hold_right": "Trial: held gaze. Look at something far to your right, mostly with your "
                  "eyes. Keep looking there, without glancing back at the screen, until I say "
                  "stop. This takes up to a minute.",
    "hold_left": "Trial: held gaze. Look at something far to your left, mostly with your "
                 "eyes. Keep looking there, without glancing back at the screen, until I say "
                 "stop. This takes up to a minute.",
}
HOLD_TIMEOUT_S = 70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--trials", default="normal,weak_left,hold_right")
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
    gm = GazeMonitor(cfg)
    rows, results = [], {}
    caption = ""

    def say(text: str):
        print(f">> {text}")
        voice.say(text)

    def frame(trial=None, ex: ExamRunner | None = None, use_pose: bool = False):
        ok, img, t = src.read()
        if not ok:
            raise SystemExit("camera stopped delivering frames")
        ff, fpx = face.process(img, t)
        pf, ppx = pose.process(img, t) if use_pose or (ex and ex.needs_pose) else (None, None)
        said = []
        step = None
        if ex is not None:
            step = ex.step.key if ex.step else None
            said = ex.update(ff, pf if ex.needs_pose else None)
        if trial:
            rows.append({"trial": trial, "step": step, "t": round(t, 3), "gaze_h": ff.gaze_h,
                         "yaw": ff.yaw, "blink": ff.blink, "face": ff.reason or "ok",
                         "arm_l": pf.arm_l if pf else None, "arm_r": pf.arm_r if pf else None,
                         "gaze_dev": gm.deviation, "gaze_value": gm.value})
        if not args.no_window:
            draw_landmarks(img, fpx, ff)
            draw_pose(img, ppx)
            g = "-" if ff.gaze_h is None else f"{ff.gaze_h:+.2f}"
            lines = [caption, f"gaze {g}  held: {gm.deviation or '-'}"]
            if ex and ex.step and not ex.finished:
                lines.append(ex.step.label)
            for i, line in enumerate(lines):
                for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                    cv2.putText(img, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                col, th, cv2.LINE_AA)
            cv2.imshow("StrokeWatch live test", img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                raise SystemExit("cancelled")
        return ff, pf, said

    def idle(seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            frame()

    # 1. face + shoulders in view
    caption = "Sit with face, shoulders and hands in view"
    say("Live test. Sit facing the camera, with your face, shoulders and hands in view.")
    t0, since = time.monotonic(), None
    while True:
        ff, pf, _ = frame(use_pose=True)
        now = time.monotonic()
        ok = ff.gaze_h is not None and pf is not None and pf.shoulders
        since = (since or now) if ok else None
        if since and now - since > 2.0:
            break
        if now - t0 > 90:
            raise SystemExit("face and shoulders never both came into view")

    # 2. learn the gaze centre, as a bed does after start
    caption = "Learning your gaze centre"
    say("First, just look at the screen normally for a few seconds.")
    t0 = time.monotonic()
    while gm.center is None:
        ff, _, _ = frame()
        gm.update(ff)
        if time.monotonic() - t0 > 60:
            raise SystemExit("could not learn the gaze centre (eyes not tracked)")
    print(f"gaze centre: {gm.center:+.3f}")

    for trial in trials:
        caption = trial
        say(INTROS[trial])
        idle(9.0 if trial.startswith("hold") else 8.0)
        if trial.startswith("hold"):
            gm.clear_window()
            start, flagged_at = time.monotonic(), None
            while time.monotonic() - start < HOLD_TIMEOUT_S:
                ff, _, _ = frame(trial)
                gm.update(ff)
                if gm.deviation:
                    flagged_at = time.monotonic() - start
                    break
            say("Stop. You can look back at the screen.")
            results[trial] = {"flagged": gm.deviation, "after_s": flagged_at,
                              "value": gm.value, "center": gm.center}
            gm.clear_window()
            idle(3.0)
        else:
            ex = ExamRunner(cfg, f"live test: {trial}", gaze_center=gm.center)
            while not ex.finished:
                _, _, said = frame(trial, ex)
                for text in said:
                    say(text)
            results[trial] = ex.result
            idle(4.0)
    say("Test finished. Thank you.")
    src.release()
    face.close()
    pose.close()
    if not args.no_window:
        cv2.destroyAllWindows()

    out = ROOT / "clips" / "befast_live.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nframes: {len(rows)} -> {out}\ngaze centre {gm.center:+.3f}")
    for trial, r in results.items():
        if trial.startswith("hold"):
            want = trial.split("_")[1].upper()
            got = r["flagged"]
            ok = "OK" if got == want else "MISSED" if got is None else "WRONG SIDE"
            when = f"after {r['after_s']:.0f} s" if r["after_s"] else f"not within {HOLD_TIMEOUT_S} s"
            print(f"\n== {trial}: held gaze flagged: {got} {when} [{ok}] "
                  f"(last median {r['value'] if r['value'] is None else round(r['value'], 3)})")
        else:
            print(f"\n== {trial}: {r['summary']}  | weak side: {r['weak_side']}")
            for k, c in r["components"].items():
                print(f"   {k}: {c['status']} - {c['detail']}")
                if c["values"]:
                    print(f"      {c['values']}")


if __name__ == "__main__":
    main()
