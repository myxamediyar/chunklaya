#!/usr/bin/env python3
"""Step 3 (semantic): AUC against haystack size m for the sports needle.

Arms, all in paragraph mode with the 4-way choice detector and `max` aggregation:

    full      Laya scores every passage                      -- chunklaya today, no prefilter
    dense@k   bge-small ranks the passages against "sports and athletics" (the best query in
              exp_ranker.py); Laya scores the top k          -- the locate -> verify design

Paragraph mode scores each passage on its own, at offset 0 of its own forward pass, so a document's
score depends only on its passages' scores. Every pool passage and needle is scored once (the row
cache has them), and each arm is then evaluated over sampled haystacks of those scores. --verify
runs a few m=70 documents through ChunkLaya.ask and checks the `full` arm matches to 1e-6.

Haystacks are nested: one shuffle of the pool per needle, and the m-passage haystack is its first m
entries, so the curve over m is paired within each needle. The pool is the AG News test split's 5700
non-sports rows (eval/data/ag_news_full.json). It carries AG News's ~1% label noise -- sports stories
filed under World or Sci/Tech -- so every arm is also run on the pool with rows Laya itself scores
P(Sports) > --clean removed, the same (model-judged) cleaning as the README's clean table -- which is
circular for the `full` arm (nothing left in the pool can outscore --clean). The `relabelled` pool is the
honest one: eval/data/ag_news_full_relabels.json hand-labels the 77 flagged rows, the real sports stories
come out and Laya's own false positives stay in.

    .venv/bin/python eval/exp_scale_semantic.py --n 100
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import ChunkLaya, PredictionCache, predict_many
from dense_ranker import DenseRanker
from metrics import auc_ci, recall_at_spec
from needle_data import CHOICE_Q, NOUL_Q, NeedleData

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--sizes", default="70,300,1000,3000,5600")
ap.add_argument("--ks", default="5,10,20,50,100")
ap.add_argument("--query", default="sports and athletics")
ap.add_argument("--clean", type=float, default=0.2)
ap.add_argument("--relabels", default=str(HERE / "data" / "ag_news_full_relabels.json"),
                help="hand labels for the passages Laya flags; 'none' to skip the relabelled pool")
ap.add_argument("--verify", type=int, default=5, help="documents to check against ChunkLaya.ask at m=70")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cache", default=str(HERE.parent / "results" / "cache" / "predictions.jsonl"))
ap.add_argument("--out", default=str(HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-scale-semantic")))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
SIZES = [int(x) for x in args.sizes.split(",")]
KS = [int(x) for x in args.ks.split(",")]

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
cache = PredictionCache(args.cache)
needles = NeedleData(agent.tok, HERE / "data" / "ag_news.json").exp2_needles(args.n)
full = json.load(open(HERE / "data" / "ag_news_full.json"))
pool = [r["text"] for r in full["rows"] if full["labels"][r["label"]] != "Sports"]
assert not set(needles) & set(pool)

# per-passage Laya scores (cached) and dense similarities
res = predict_many(agent, [(t, CHOICE_Q) for t in pool + needles], 16, cache=cache)
S = np.array([r["probabilities"]["Sports"] for r in res])
S_pool, S_needle = S[:len(pool)], S[len(pool):]
dense = DenseRanker("BAAI/bge-small-en-v1.5", str(agent.device))
q = dense.embed([DenseRanker.QUERY_PREFIX + args.query])[0]
sim_pool, sim_needle = dense.embed(pool) @ q, dense.embed(needles) @ q
print(f"loaded in {time.time()-t0:.0f}s; pool {len(pool)} passages, {args.n} needles; {cache.stats()}; "
      f"dense {dense.encoded} passages in {dense.seconds:.1f}s")

POOLS = {"raw": np.arange(len(pool)), "clean": np.flatnonzero(S_pool <= args.clean)}
if args.relabels != "none":
    # the honest pool: real sports stories out (hand-labelled), Laya's own false positives kept in
    rl = json.load(open(args.relabels))
    row_of = {r["text"]: i for i, r in enumerate(full["rows"])}
    sports_rows = {r["row"] for r in rl["rows"] if r["is_sports"]}
    POOLS["relabelled"] = np.array([j for j, t in enumerate(pool) if row_of[t] not in sports_rows])
    kept_fp = [r for r in rl["rows"] if not r["is_sports"]]
    print(f"relabelled pool: {len(POOLS['relabelled'])} of {len(pool)}: {len(sports_rows)} hand-labelled sports stories removed, "
          f"{len(kept_fp)} detector false positives kept (max P(Sports) {max(r['p_sports'] for r in kept_fp):.3f})")
print(f"clean pool: {len(POOLS['clean'])} of {len(pool)} (dropped {len(pool)-len(POOLS['clean'])} with P(Sports) > {args.clean})")
perms = {i: random.Random(args.seed * 1000 + 500 + i) for i in range(args.n)}



R = {"n": args.n, "sizes": SIZES, "ks": KS, "query": args.query, "clean": args.clean, "summary": {}, "rows": []}
for pool_name, idx in POOLS.items():
    order = {i: perms[i].sample(list(idx), len(idx)) for i in range(args.n)}
    print(f"\n=== pool: {pool_name} ({len(idx)} passages) ===")
    print(f"  {'m':>5}{'arm':>10}{'AUC [95% CI]':>22}{'r@spec90':>10}{'none q50':>10}{'needle q50':>12}{'R@k':>7}{'laya/doc':>10}{'dense/doc':>10}")
    for m in SIZES:
        m_eff = min(m, len(idx))
        arms = {"full": None, **{f"dense@{k}": k for k in KS if k < m_eff}}
        per = {a: {"pos": [], "neg": [], "hit": []} for a in arms}
        for i in range(args.n):
            hay = np.array(order[i][:m_eff])
            hs, hsim = S_pool[hay], sim_pool[hay]
            ps, psim = np.append(hs, S_needle[i]), np.append(hsim, sim_needle[i])
            for a, k in arms.items():
                if k is None:
                    neg, pos, hit = float(hs.max()), float(ps.max()), True
                else:
                    neg = float(hs[np.argpartition(-hsim, k)[:k]].max())
                    top = np.argpartition(-psim, k)[:k]
                    pos, hit = float(ps[top].max()), bool(len(ps) - 1 in top)
                per[a]["pos"].append(pos); per[a]["neg"].append(neg); per[a]["hit"].append(hit)
                R["rows"].append({"pool": pool_name, "m": m_eff, "arm": a, "i": i, "present": pos, "absent": neg, "ranker_hit": hit})
        for a, k in arms.items():
            pos, neg = per[a]["pos"], per[a]["neg"]
            auc, lo, hi = auc_ci(pos, neg)
            summ = {"auc": auc, "ci": [lo, hi], "recall_spec90": recall_at_spec(pos, neg),
                    "none_q50": statistics.median(neg), "needle_q50": statistics.median(pos),
                    "ranker_recall": statistics.mean(per[a]["hit"]),
                    "laya_per_doc": m_eff if k is None else k, "dense_per_doc": 0 if k is None else m_eff}
            R["summary"][f"{pool_name}/{m_eff}/{a}"] = summ
            print(f"  {m_eff:>5}{a:>10}{auc:>8.3f} [{lo:.2f},{hi:.2f}]{summ['recall_spec90']:>10.2f}"
                  f"{summ['none_q50']:>10.3f}{summ['needle_q50']:>12.3f}{summ['ranker_recall']:>7.2f}"
                  f"{summ['laya_per_doc']:>10}{summ['dense_per_doc']:>10}")

if args.verify:
    cj = ChunkLaya(agent, mode="paragraphs", cache=cache)
    idx = POOLS["raw"]; worst = 0.0
    for i in range(args.verify):
        hay = [pool[j] for j in random.Random(args.seed * 1000 + 500 + i).sample(list(idx), len(idx))[:70]]
        doc = "\n\n".join(hay[:35] + [needles[i]] + hay[35:])
        r = cj.ask(doc, {"q": NOUL_Q}, detectors={"q": (CHOICE_Q, "Sports")})
        sim = max(float(S_pool[[pool.index(h) for h in hay]].max()), float(S_needle[i]))
        worst = max(worst, abs(r["answers"]["q"]["noul"] - sim))
        assert r["n_chunks"] == 71, r["n_chunks"]
    print(f"\nverify: {args.verify} documents at m=70 through ChunkLaya.ask, max |ask - simulated| = {worst:.2e}")
    assert worst < 1e-6

cache.close()
json.dump(R, open(OUT / "results.json", "w"), indent=1)
print(f"written {OUT}/results.json | total {time.time()-t0:.0f}s")
