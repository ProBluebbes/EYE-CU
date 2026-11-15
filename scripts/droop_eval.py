"""Score the synthetic-droop benchmark (scripts/droop_benchmark.py extract): StrokeWatch's detector, the learned
palsy models, and combinations, on the same frames.

Per run (control or droop) and per 0.5 s step we record StrokeWatch's score / threshold and state, and each
learned model's p(affected) for the last 2 s (same smoothing / 3D frontalisation / window as strokewatch/palsy.py).
Saved to <bench>/scores.pkl for analysis.

    uv run python scripts/droop_eval.py score runs/droop_bench --models runs/cv_landmark_tcn_... runs/cv_landmark_geom_... runs/cv_landmark_stgcn_...
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

HOP_S, SMOOTH, BATCH = 0.5, 3, 32  # batch 32: the 3D-CNN at 256 needs several GB per process on CPU


def stream_windows(run, fps: float, num_frames: int, sample_fps: float):
    """Offline copy of strokewatch.palsy.PalsyStream sampling: (times, windows (K, T, 152, 3), |yaw| max)."""
    from palsynet.landmarks import StreamingLandmarkSmoother

    sm, size = None, None
    buf, yaw, times, wins, wyaw = [], [], [], [], []
    next_sample = next_pred = None
    for i, (ff, lms, shape) in enumerate(run):
        h, w = shape
        t = i / fps
        if sm is None or size != (w, h):
            sm, size = StreamingLandmarkSmoother(SMOOTH, w, h, max_gap=round(0.5 * fps)), (w, h)
            buf, yaw, next_sample = [], [], None
        for lf in sm.push(None, lms, t):
            if lf.features is None:
                buf, yaw, next_sample = [], [], None
                continue
            if next_sample is None:
                next_sample = lf.timestamp
            if lf.timestamp + 1e-6 >= next_sample:
                buf.append(lf.features)
                yaw.append(abs(float(lf.pose[0])))
                buf, yaw = buf[-num_frames:], yaw[-num_frames:]
                next_sample += 1.0 / sample_fps
            if len(buf) == num_frames and (next_pred is None or lf.timestamp + 1e-6 >= next_pred):
                times.append(lf.timestamp)
                wins.append(np.stack(buf))
                wyaw.append(max(yaw))
                next_pred = lf.timestamp + HOP_S
    return np.array(times), (np.stack(wins) if wins else np.zeros((0, num_frames, 152, 3), np.float32)), np.array(wyaw)


_W = {}


def _init_worker(model_dirs, device, threads):
    import torch

    torch.set_num_threads(threads)
    from strokewatch import load_config, palsy

    _W["M"] = palsy.PalsyModels(model_dirs, device=device)
    _W["cfg"] = load_config()


def _score_run(job):
    """One (clip, run): StrokeWatch's detector frame by frame + every learned model on the same landmarks."""
    import torch

    from strokewatch.detector import Detector

    pkl, name, dst = job
    M, cfg = _W["M"], _W["cfg"]
    rec = pickle.load(open(pkl, "rb"))
    fps, run, clip = rec["fps"], rec["runs"][name], rec["clip"]
    del rec  # keep only this run in memory
    det = Detector(cfg)
    sw = []
    for ff, _, _ in run:
        st = det.update(ff)
        sw.append((ff.t, st.score / st.threshold, st.state, st.valid, st.state == "baselining"))
    sw_t = np.array([s[0] for s in sw])
    times, wins, wyaw = stream_windows(run, fps, M.num_frames, M.sample_fps)
    probs = []
    with torch.no_grad():
        for g in M.groups:
            ps = []
            for b in range(0, len(wins), BATCH):
                x = torch.from_numpy(wins[b:b + BATCH]).float().to(M.device)
                ps.append(np.mean([m(x).float().softmax(-1)[:, 1].cpu().numpy() for m in g["models"]], 0))
            probs.append(np.concatenate(ps) if ps else np.zeros(0))
    j = np.clip(np.searchsorted(sw_t, times, side="right") - 1, 0, len(sw) - 1)
    res = {"t": times, "p": np.stack(probs, 1) if len(times) else np.zeros((0, len(M.groups))), "yaw": wyaw,
           "sw_rel": np.array([sw[k][1] for k in j]), "sw_state": np.array([sw[k][2] for k in j]),
           "sw_valid": np.array([sw[k][3] for k in j]), "sw_baselining": np.array([sw[k][4] for k in j]),
           "sw_t": sw_t, "sw_state_all": np.array([s[2] for s in sw])}
    with open(dst, "wb") as f:
        pickle.dump(res, f)
    return f"{clip} {name}: {len(times)} windows"


def score(bench: Path, model_dirs: list[str], device: str = "cpu", workers: int = 4):
    """Every (clip, run) is its own job, run in parallel and saved on its own (re-running skips done jobs)."""
    from multiprocessing import Pool

    from strokewatch import load_config

    cache = bench / "scored"
    cache.mkdir(exist_ok=True)
    clips = [p for p in sorted(bench.glob("*.pkl")) if p.name != "scores.pkl"]
    # run names are fixed by droop_benchmark.py; listing them needs no clip loaded (memory: ~3 GB per worker)
    sys.path.insert(0, str(ROOT / "scripts"))
    from droop_benchmark import FPS, SIDES, STRENGTHS

    run_names = ["control"] + [f"{sd}_{st}" for sd in SIDES for st in STRENGTHS]
    jobs = []
    for pkl in clips:
        if (cache / pkl.name).exists():  # whole clip scored by an older (one-process) run
            continue
        for name in run_names:
            dst = cache / f"{pkl.stem}__{name}.pkl"
            if not dst.exists():
                jobs.append((pkl, name, dst))
    # long clips first, so they don't end up last on one worker
    jobs.sort(key=lambda j: -j[0].stat().st_size)
    threads = max(1, (os.cpu_count() or 4) // workers)
    print(f"{len(jobs)} jobs on {workers} workers x {threads} threads", flush=True)
    if jobs:
        with Pool(workers, initializer=_init_worker, initargs=(model_dirs, device, threads)) as pool:
            for k, msg in enumerate(pool.imap_unordered(_score_run, jobs), 1):
                print(f"[{k}/{len(jobs)}] {msg}", flush=True)
    import torch  # noqa: F401  (names from a model load, for the summary)
    from strokewatch import palsy

    names = palsy.PalsyModels(model_dirs, device="cpu").names()
    out = {"models": names, "threshold_sw": load_config()["detector"]["threshold"], "clips": {}}
    for pkl in clips:
        if (cache / pkl.name).exists():
            r = pickle.load(open(cache / pkl.name, "rb"))
            out["clips"][r["clip"]] = r["res"]
            continue
        runs = {n: pickle.load(open(cache / f"{pkl.stem}__{n}.pkl", "rb")) for n in run_names}
        onset_t = (len(runs["control"]["sw_t"]) // 2) / FPS  # droop_benchmark: onset = middle of the timeline
        out["clips"][pkl.stem] = {"onset_t": onset_t, "runs": runs}
    with open(bench / "scores.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"saved {bench / 'scores.pkl'} ({len(out['clips'])} clips)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("bench", type=Path)
    sc.add_argument("--models", nargs="+", required=True)
    sc.add_argument("--device", default="cpu")
    sc.add_argument("--workers", type=int, default=2)  # ~3 GB each (a whole clip is loaded per job)
    an = sub.add_parser("analyze")
    an.add_argument("bench", type=Path)
    an.add_argument("--fpr", type=float, default=0.01)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.bench, a.models, a.device, a.workers)
    else:
        analyze(a.bench, a.fpr)
        alerts_tuned(a.bench, ["StrokeWatch", "TCN", "3D-CNN", "ensemble(3)", "SW, else 3D-CNN",
                               "TCN vs own baseline", "3D-CNN vs own baseline", "ensemble vs own baseline",
                               "SW + TCN vs baseline", "SW + 3D-CNN vs baseline", "SW + ensemble vs baseline"], a.fpr)



# ----------------------------------------------------------------------------- analysis
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _roll(x, k=4):
    """Trailing mean of the last k steps (the live --avg / palsy.py average)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)


def build_signals(run, names, base_s=20.0):
    """All candidate scores for one run, on its 0.5 s steps. Every score is scaled so that 1.0 = its default
    flag line (StrokeWatch: its threshold; models: p 0.5), except the personal ones (threshold tuned)."""
    t, p = run["t"], run["p"]
    sw_ok = ~run["sw_baselining"] & run["sw_valid"]  # StrokeWatch has a verdict right now
    S = {"StrokeWatch": np.where(run["sw_baselining"], 0.0, run["sw_rel"])}
    pav = np.stack([_roll(p[:, k]) for k in range(p.shape[1])], 1) if len(t) else p
    for k, n in enumerate(names):
        S[n] = pav[:, k] / 0.5
    S["ensemble(3)"] = pav.mean(1) / 0.5
    # personal baseline for the learned models, built like StrokeWatch's: their scores in the first 20 s
    base = t < (t[0] + base_s) if len(t) else np.zeros(0, bool)
    lg = _logit(pav)
    d = lg - (np.median(lg[base], 0) if base.any() else 0.0)
    for k, n in enumerate(names):
        S[f"{n} vs own baseline"] = d[:, k]
    S["ensemble vs own baseline"] = d.mean(1)
    sw = S["StrokeWatch"]
    for k, n in enumerate(names + ["ensemble(3)"]):
        q = pav[:, k] if k < len(names) else pav.mean(1)
        S[f"SW AND {n}"] = np.minimum(sw, q / 0.5)         # both over their lines
        S[f"SW OR {n}"] = np.maximum(sw, q / 0.5)
        S[f"SW, else {n}"] = np.where(sw_ok, sw, q / 0.5)  # StrokeWatch when it can judge, model otherwise
    for k, n in enumerate(names + ["ensemble"]):
        dk = d[:, k] if k < len(names) else d.mean(1)
        S[f"SW + {n} vs baseline"] = sw + dk               # weight fixed at 1 (no tuning on the test clips)
    return S


def _auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def analyze(bench: Path, fpr_target: float = 0.01, persist_s: float = 6.0):
    D = pickle.load(open(bench / "scores.pkl", "rb"))
    names = D["models"]
    clips = sorted(D["clips"])
    # collect (clip, label, strength, score dict) for steps after the droop has fully ramped in
    rows = {}  # signal -> list of (clip, run, values in region)
    region = {}
    for c in clips:
        res = D["clips"][c]
        for rn, run in res["runs"].items():
            S = build_signals(run, names)
            m = run["t"] >= res["onset_t"] + 3.0
            region[(c, rn)] = (run["t"], m, res["onset_t"])
            for k, v in S.items():
                rows.setdefault(k, []).append((c, rn, v))
    sig_names = list(rows)

    def split(k, clip_filter, strength=None):
        pos, neg = [], []
        for c, rn, v in rows[k]:
            if not clip_filter(c):
                continue
            m = region[(c, rn)][1]
            if rn == "control":
                neg.append(v[m])
            elif strength is None or rn.endswith(strength):
                pos.append(v[m])
        return (np.concatenate(pos) if pos else np.zeros(0)), (np.concatenate(neg) if neg else np.zeros(0))

    # window level: AUC (pooled), and detection at <= fpr_target false positives with the threshold chosen on
    # the OTHER clips (leave-one-clip-out), pooled over held-out clips
    table = []
    for k in sig_names:
        pos, neg = split(k, lambda c: True)
        auc = _auc(pos, neg)
        aucs = {s: _auc(*split(k, lambda c: True, s)) for s in ("mild", "moderate", "strong")}
        tp = fp = npos = nneg = 0
        for held in clips:
            ptr, ntr = split(k, lambda c: c != held)
            thr = np.quantile(ntr, 1 - fpr_target) if len(ntr) else np.inf
            pte, nte = split(k, lambda c: c == held)
            tp += (pte > thr).sum(); npos += len(pte)
            fp += (nte > thr).sum(); nneg += len(nte)
        # at the default flag line (1.0) where it exists
        dflt = None
        if not k.endswith("baseline") and "+" not in k:  # signals with a fixed flag line at 1.0
            dflt = ((pos >= 1).mean(), (neg >= 1).mean())
        table.append((k, auc, aucs, tp / max(npos, 1), fp / max(nneg, 1), dflt))

    print(f"\n{len(clips)} clips; droop runs: left/right x mild/moderate/strong; scored after onset + 3 s ramp\n")
    print(f"{'signal':34s} {'AUC':>6s} {'mild':>6s} {'mod.':>6s} {'strong':>6s}   "
          f"{'held-out @'+str(int(fpr_target*100))+'% FPR: det / FP':>30s}   {'default line: det / FP':>24s}")
    for k, auc, aucs, tpr, fpr, dflt in sorted(table, key=lambda r: -np.nan_to_num(r[1])):
        d = f"{dflt[0]:6.0%} / {dflt[1]:5.1%}" if dflt else ""
        print(f"{k:34s} {auc:6.3f} {aucs['mild']:6.3f} {aucs['moderate']:6.3f} {aucs['strong']:6.3f}   "
              f"{tpr:20.0%} / {fpr:5.1%}   {d:>24s}")

    # per clip, at the default flag line: share of droop steps flagged / share of control steps flagged
    show = ["StrokeWatch"] + names + ["ensemble(3)"] + [f"SW, else {n}" for n in names]
    print("\nper clip at the default line (droop flagged / control flagged); StrokeWatch coverage = share of "
          "post-onset steps where it had a baseline and a usable frame:")
    print(f"{'clip':22s}{'SW coverage':>12s}" + "".join(f"{n[:16]:>18s}" for n in show))
    for c in clips:
        ctl = D["clips"][c]["runs"]["control"]
        mm = region[(c, "control")][1]
        cov = (~ctl["sw_baselining"][mm] & ctl["sw_valid"][mm]).mean() if mm.any() else np.nan
        cells = []
        for k in show:
            pos, neg = split(k, lambda x: x == c)
            cells.append(f"{(pos >= 1).mean():6.0%} / {(neg >= 1).mean():4.0%}" if len(pos) and len(neg) else "-")
        print(f"{c:22s}{cov:>12.0%}" + "".join(f"{x:>18s}" for x in cells))

    # alert level (StrokeWatch's own rule: above the line continuously for 6 s); StrokeWatch native = its ALERT
    print(f"\nalerts (>= line for {persist_s:.0f} s; StrokeWatch = its own ALERT state):")
    print(f"{'signal':34s} {'droops caught':>14s} {'median delay':>13s} {'false alerts (controls)':>24s}")
    steps = int(round(persist_s / HOP_S))
    for k in ["StrokeWatch"] + names + ["ensemble(3)"] + [f"SW AND {n}" for n in names + ['ensemble(3)']] + \
             [f"SW OR {n}" for n in names] + [f"SW, else {n}" for n in names + ['ensemble(3)']]:
        caught, delays, false = 0, [], 0
        nd = nc = 0
        for c, rn, v in rows[k]:
            res = D["clips"][c]
            run = res["runs"][rn]
            t, m, onset = region[(c, rn)]
            if k == "StrokeWatch":
                st_t, st = run["sw_t"], run["sw_state_all"]
                post = st_t >= onset
                hits = st_t[post & (st == "ALERT")]
                first = hits[0] if len(hits) else None
                pre_false = (st[(st_t < onset)] == "ALERT").any()
            else:
                above = v >= 1
                run_len = np.zeros(len(v), int)
                for i in range(len(v)):
                    run_len[i] = run_len[i - 1] + 1 if above[i] and i > 0 else int(above[i])
                fire = run_len >= steps
                first = t[(t >= onset) & fire][0] if ((t >= onset) & fire).any() else None
                pre_false = (fire & (t < onset)).any()
            if rn == "control":
                nc += 1
                false += (first is not None) or pre_false
            else:
                nd += 1
                if first is not None:
                    caught += 1
                    delays.append(first - onset)
        md = f"{np.median(delays):.1f} s" if delays else "-"
        print(f"{k:34s} {caught:>6d}/{nd:<7d} {md:>13s} {false:>14d}/{nc}")



def alerts_tuned(bench: Path, signals: list[str], fpr_target: float = 0.01, persist_s: float = 6.0):
    """Alert-level results for signals without a fixed flag line: for each held-out clip, the line is the
    (1 - fpr_target) quantile of the OTHER clips' control steps; alert = above it for persist_s."""
    D = pickle.load(open(bench / "scores.pkl", "rb"))
    names, clips = D["models"], sorted(D["clips"])
    sig = {c: {rn: build_signals(run, names) for rn, run in D["clips"][c]["runs"].items()} for c in clips}
    steps = int(round(persist_s / HOP_S))
    print(f"\nalerts with the line tuned on the other clips (<= {fpr_target:.0%} of their no-droop steps), "
          f">= line for {persist_s:.0f} s:")
    print(f"{'signal':34s} {'droops caught':>14s} {'median delay':>13s} {'false alerts':>13s}   per clip caught (of 6)")
    for k in signals:
        caught, delays, false, nd, nc, per = 0, [], 0, 0, 0, []
        for held in clips:
            onset = D["clips"][held]["onset_t"]
            neg = np.concatenate([sig[c]["control"][k][D["clips"][c]["runs"]["control"]["t"] >= D["clips"][c]["onset_t"] + 3]
                                  for c in clips if c != held])
            line = np.quantile(neg, 1 - fpr_target)
            pc = 0
            for rn, S in sig[held].items():
                v, t = S[k], D["clips"][held]["runs"][rn]["t"]
                above = v > line
                run_len = np.zeros(len(v), int)
                for i in range(len(v)):
                    run_len[i] = run_len[i - 1] + 1 if (above[i] and i > 0) else int(above[i])
                fire = run_len >= steps
                post = (t >= onset) & fire
                if rn == "control":
                    nc += 1
                    false += bool(fire[t >= t[0] + 20].any())  # after the 20 s personal baseline
                else:
                    nd += 1
                    if post.any():
                        caught += 1; pc += 1
                        delays.append(t[post][0] - onset)
            per.append(f"{held[:10]} {pc}")
        md = f"{np.median(delays):.1f} s" if delays else "-"
        print(f"{k:34s} {caught:>6d}/{nd:<7d} {md:>13s} {false:>9d}/{nc}   " + ", ".join(per))


if __name__ == "__main__":
    main()
