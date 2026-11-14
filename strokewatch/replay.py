"""Run the full pipeline on a recorded clip and save a score-vs-time plot.

    python -m strokewatch.replay clips/droop.mp4                 # -> clips/droop.png + .csv
    python -m strokewatch.replay clips/droop.mp4 --show          # also show annotated video
    python -m strokewatch.replay clips/*.mp4                     # batch
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import apply_overrides, load_config  # noqa: E402
from .detector import Detector  # noqa: E402
from .vision import FaceAnalyzer, VideoSource, draw_overlay  # noqa: E402


def replay(path: str, cfg: dict, show: bool = False) -> dict:
    src = VideoSource(path, clock="video", rotate=cfg["camera"].get("rotate", 0))
    an, det = FaceAnalyzer(cfg), Detector(cfg)
    rows = []
    while True:
        ok, frame, t = src.read()
        if not ok:
            break
        ff, px = an.process(frame, t)
        st = det.update(ff)
        rows.append({**{k: v for k, v in ff.to_dict().items()},
                     "state": st.state, "score": st.score, "side": st.side,
                     "onset_time": st.onset_time})
        if show:
            cv2.imshow("replay", draw_overlay(frame, px, ff, st.to_dict()))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                show = False
                cv2.destroyAllWindows()
    src.release()
    an.close()
    if show:
        cv2.destroyAllWindows()
    if not rows:
        raise SystemExit(f"no frames read from {path}")

    out = Path(path).with_suffix("")
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dc = cfg["detector"]
    ts = [r["t"] for r in rows]
    score = [r["score"] if r["valid"] and r["state"] != "baselining" else float("nan") for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(ts, score, color="#1f77b4", lw=1.5, label="score (smoothed max |z|)")
    ax1.axhline(dc["threshold"], color="#d62728", ls="--", lw=1, label="threshold")
    ax1.axhline(dc["pre_threshold"], color="#ff7f0e", ls=":", lw=1, label="pre-threshold")
    # grey = skipped frames
    for r in rows:
        if not r["valid"]:
            ax1.axvspan(r["t"], r["t"] + 1 / 30, color="0.85", lw=0)
    base_end = next((r["t"] for r in rows if r["state"] != "baselining"), None)
    if base_end is not None:
        ax1.axvline(base_end, color="0.4", lw=1)
        ax1.text(base_end, ax1.get_ylim()[1] * 0.95, " baseline done", color="0.3", va="top")
    alert = next((r for r in rows if r["state"] == "ALERT"), None)
    if alert:
        ax1.axvline(alert["t"], color="#d62728", lw=2)
        ax1.axvline(alert["onset_time"], color="#d62728", lw=1, ls=":")
        ax1.text(alert["t"], ax1.get_ylim()[1] * 0.8,
                 f" ALERT {alert['side']}\n onset {alert['onset_time']:.1f}s", color="#d62728", va="top")
    ax1.set_ylabel("score")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{Path(path).name}  (grey = skipped frames)")
    for key, col in [("mouth_diff", "#2ca02c"), ("eye_ratio", "#9467bd"), ("brow_diff", "#8c564b")]:
        ax2.plot(ts, [r[key] if r[key] is not None else float("nan") for r in rows],
                 color=col, lw=1, label=key)
    ax2.set_ylabel("raw feature (+ = left lower)")
    ax2.set_xlabel("video time (s)")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=110)
    plt.close(fig)

    n_valid = sum(r["valid"] for r in rows)
    summary = {"clip": path, "frames": len(rows), "valid_frac": round(n_valid / len(rows), 2),
               "baseline_done_at": base_end,
               "alert_at": alert["t"] if alert else None,
               "onset": alert["onset_time"] if alert else None,
               "side": alert["side"] if alert else None,
               "max_score": max((s for s in score if s == s), default=0.0),
               "plot": f"{out}.png"}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--set", action="append", metavar="KEY=VAL", help="config override, e.g. detector.threshold=3.5")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    for c in args.clips:
        s = replay(c, cfg, args.show)
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
