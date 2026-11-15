"""Aggregate test results of a cross-validation directory (runs/cv_*).

Reports mean ± std of per-fold video-level metrics, plus pooled metrics over all
out-of-fold test videos (each video is tested exactly once across the folds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import numpy as np
import pandas as pd
from palsynet.engine import _metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cv_dir", type=Path)
    args = p.parse_args()

    runs = sorted(args.cv_dir.glob("*_fold*_*/test_metrics.json"))
    if not runs:
        raise SystemExit(f"No test_metrics.json under {args.cv_dir}")
    keys = ["accuracy", "balanced_accuracy", "f1", "auc"]
    per_fold = {k: [json.load(open(r))["video"][k] for r in runs] for k in keys}
    preds = pd.concat([pd.read_csv(r.parent / "test_predictions.csv") for r in runs], ignore_index=True)
    y = (preds["label"] == "affected").astype(int).to_numpy()
    # thresholded metrics use the saved "pred" column (decided on the unrounded probability; the CSV rounds
    # probabilities to 4 decimals, so e.g. 0.49996 would otherwise be read back as 0.5000 and counted >= 0.5)
    prob = preds["prob_affected"].to_numpy().astype(float)
    pred = (preds["pred"] == "affected").astype(int).to_numpy()
    prob = np.where(pred == 1, np.maximum(prob, 0.5), np.minimum(prob, np.nextafter(0.5, 0)))
    pooled = _metrics(y, prob)

    lines = [f"{len(runs)} folds, {len(preds)} test videos (video-level)", ""]
    lines.append(f"{'metric':<20}{'mean ± std over folds':>24}{'pooled':>10}")
    for k in keys:
        v = np.array(per_fold[k], dtype=float)
        lines.append(f"{k:<20}{np.nanmean(v):>14.3f} ± {np.nanstd(v):.3f}{pooled[k]:>10.3f}")
    tn, fp, fn, tp = np.array(pooled["confusion_matrix"]).ravel()
    lines += ["", f"pooled confusion (rows=true unaffected/affected): [[{tn} {fp}] [{fn} {tp}]]"]
    wrong = preds[pred != y]
    if len(wrong):
        lines += ["", "misclassified videos:"] + [f"  {r.video_id:<16} p(affected)={r.prob_affected:.3f}" for r in wrong.itertuples()]
    text = "\n".join(lines)
    print(text)
    (args.cv_dir / "summary.txt").write_text(text + "\n")
    preds.to_csv(args.cv_dir / "oof_predictions.csv", index=False)


if __name__ == "__main__":
    main()
