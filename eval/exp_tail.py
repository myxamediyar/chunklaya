#!/usr/bin/env python3
"""Step 3c: can the detector's false-positive tail be shrunk?

exp_scale_semantic.py shows that with the relabelled pool -- real sports stories out, Laya's own false
positives in -- full-scan `max` over the 4-way choice decays with m (0.985 at 70 passages, 0.785 at
5600). The reason is 36 passages in 5659 that the detector rates as sports, 16 of them above 0.9 and
one at 0.9998. This scores the pool with other detectors and simulates, on the same nested haystacks:

    single     max over one detector's P(sports)      4w (baseline) | 2w | 5w (a distractor class) | 4w-strict
    mean       max over the mean of two detectors     the tail shrinks if their mistakes differ
    min        max over the min of two detectors      both must agree (the stronger AND)
    two-stage  4w full scan, re-verify its top-20 with a second detector, score = max of min(4w, second)
               -- same as `min` on the top-20, but only 20 extra forward passes per document

Everything is per-passage, so the arms are simulated from cached scores exactly as in
exp_scale_semantic.py (which verified that simulation against ChunkLaya.ask).

    .venv/bin/python eval/exp_tail.py --n 100
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import PredictionCache, predict_many
from metrics import auc_ci, recall_at_spec
from needle_data import CHOICE_Q, NeedleData

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--sizes", default="70,300,1000,3000,5600")
ap.add_argument("--top", type=int, default=20, help="passages the two-stage arm re-verifies")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--relabels", default=str(HERE / "data" / "ag_news_full_relabels.json"))
ap.add_argument("--cache", default=str(HERE.parent / "results" / "cache" / "predictions.jsonl"))
ap.add_argument("--out", default=str(HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-tail")))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
SIZES = [int(x) for x in args.sizes.split(",")]

DETECTORS = {
    "4w": (CHOICE_Q, "Sports"),
    "2w": ({"type": "choice", "instructions": "Which category does this text belong to?",
            "criteria": {"sports": "sports and athletics", "other": "anything else"}}, "sports"),
    "5w": ({"type": "choice", "instructions": "Which news category does this text belong to?",
            "criteria": {"World": "international and political news", "Sports": "sports and athletics: a game, match, race, "
                         "tournament, season, athlete's performance or team's results",
                         "Business": "business, markets, and finance", "Sci/Tech": "science and technology",
                         "Adjacent": "a crime, court, celebrity, business or entertainment story that only mentions an athlete, "
                                     "a team, a sports brand or a video game"}}, "Sports"),
    "4w-strict": ({"type": "choice", "instructions": "Which news category does this text belong to?",
                   "criteria": {"World": "international and political news, crime, courts and accidents",
                                "Sports": "the result or preview of a game, match, race or tournament, or an athlete's or team's "
                                          "performance, transfer, injury or record",
                                "Business": "business, markets, finance and company news", "Sci/Tech": "science, technology and video games"}},
                  "Sports"),
}

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
cache = PredictionCache(args.cache)
needles = NeedleData(agent.tok, HERE / "data" / "ag_news.json").exp2_needles(args.n)
full = json.load(open(HERE / "data" / "ag_news_full.json"))
pool = [r["text"] for r in full["rows"] if full["labels"][r["label"]] != "Sports"]
rl = json.load(open(args.relabels))
row_of = {r["text"]: i for i, r in enumerate(full["rows"])}
sports_rows = {r["row"] for r in rl["rows"] if r["is_sports"]}
keep = [j for j, t in enumerate(pool) if row_of[t] not in sports_rows]
pool = [pool[j] for j in keep]
print(f"loaded in {time.time()-t0:.0f}s; relabelled pool {len(pool)}, {args.n} needles", flush=True)

S = {}
for name, (q, key) in DETECTORS.items():
    t = time.time()
    res = predict_many(agent, [(x, q) for x in pool + needles], 16, cache=cache)
    p = np.array([r["probabilities"][key] for r in res])
    S[name] = (p[:len(pool)], p[len(pool):])
    sp, sn = S[name]
    print(f"  {name:10} {time.time()-t:5.0f}s  pool: >0.5 {int((sp>0.5).sum()):3} >0.9 {int((sp>0.9).sum()):3} max {sp.max():.4f}   "
          f"needles: q50 {np.median(sn):.3f} q10 {np.quantile(sn, .1):.3f} min {sn.min():.3f}", flush=True)

perm = {i: np.array(random.Random(args.seed * 1000 + 500 + i).sample(range(len(pool)), len(pool))) for i in range(args.n)}
ARMS = {"4w": lambda P: P["4w"], "2w": lambda P: P["2w"], "5w": lambda P: P["5w"], "4w-strict": lambda P: P["4w-strict"],
        "mean(4w,2w)": lambda P: (P["4w"] + P["2w"]) / 2, "mean(4w,5w)": lambda P: (P["4w"] + P["5w"]) / 2,
        "min(4w,2w)": lambda P: np.minimum(P["4w"], P["2w"]), "min(4w,5w)": lambda P: np.minimum(P["4w"], P["5w"]),
        "min(4w,strict)": lambda P: np.minimum(P["4w"], P["4w-strict"])}
TWO_STAGE = {"2stage(4w→5w)": "5w", "2stage(4w→2w)": "2w", "2stage(4w→strict)": "4w-strict"}


def two_stage(P, second, top):
    """4w over everything; the `top` passages by 4w are re-verified by `second`; a passage counts only if both say so."""
    a = P["4w"]
    if len(a) <= top:
        return float(np.minimum(a, P[second]).max())
    idx = np.argpartition(-a, top)[:top]
    return float(np.minimum(a[idx], P[second][idx]).max())


R = {"n": args.n, "sizes": SIZES, "detectors": {k: v[0] for k, v in DETECTORS.items()}, "summary": {}}
print(f"\n  {'m':>5}{'arm':>20}{'AUC [95% CI]':>22}{'r@spec90':>10}{'none q50':>10}{'none q90':>10}{'needle q50':>12}")
for m in SIZES:
    m_eff = min(m, len(pool))
    for arm in list(ARMS) + list(TWO_STAGE):
        pos, neg = [], []
        for i in range(args.n):
            hay = perm[i][:m_eff]
            Ph = {k: S[k][0][hay] for k in DETECTORS}
            Pp = {k: np.append(S[k][0][hay], S[k][1][i]) for k in DETECTORS}
            if arm in ARMS:
                neg.append(float(ARMS[arm](Ph).max())); pos.append(float(ARMS[arm](Pp).max()))
            else:
                neg.append(two_stage(Ph, TWO_STAGE[arm], args.top)); pos.append(two_stage(Pp, TWO_STAGE[arm], args.top))
        auc, lo, hi = auc_ci(pos, neg)
        summ = {"auc": auc, "ci": [lo, hi], "recall_spec90": recall_at_spec(pos, neg), "none_q50": statistics.median(neg),
                "none_q90": float(np.quantile(neg, .9)), "needle_q50": statistics.median(pos)}
        R["summary"][f"{m_eff}/{arm}"] = summ
        print(f"  {m_eff:>5}{arm:>20}{auc:>8.3f} [{lo:.2f},{hi:.2f}]{summ['recall_spec90']:>10.2f}{summ['none_q50']:>10.3f}"
              f"{summ['none_q90']:>10.3f}{summ['needle_q50']:>12.3f}", flush=True)
    print()

cache.close()
json.dump(R, open(OUT / "results.json", "w"), indent=1)
print(f"written {OUT}/results.json | total {time.time()-t0:.0f}s")
