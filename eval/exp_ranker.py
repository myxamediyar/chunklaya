#!/usr/bin/env python3
"""Step 2 of the arbitrary-length plan: can a ranker locate a *semantic* needle?

The BM25 locate step puts the lexical Jev needle first in 180/180 documents. The sports needle has no
key phrase, so this asks recall@k for three rankers on the Exp 2 needles (eval/needle_data.py):

    bm25    chunklaya.prefilter.bm25_rank against the question text (what prefilter="bm25" does now)
    dense   BAAI/bge-small-en-v1.5 (33M params), CLS pooling, cosine — the query carries bge's
            retrieval instruction prefix
    hybrid  reciprocal-rank fusion of the two

at two haystack sizes: the ~70-passage Exp 2 haystacks, and the whole 877-passage non-sports pool.
The full pool is also scored with the 14 mislabeled sports stories removed (Laya's own P(Sports) >
--clean, from the row cache), since a ranker that puts a real sports story ahead of the needle is right.

Output: recall@k and the needle's median rank per ranker; every row goes to results.json.

    .venv/bin/python eval/exp_ranker.py --n 100
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import PredictionCache, predict_many
from chunklaya.prefilter import bm25_rank, question_query
from needle_data import CHOICE_Q, NOUL_Q, NeedleData

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--dense", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--clean", type=float, default=0.2, help="drop pool articles Laya scores P(Sports) above this")
ap.add_argument("--cache", default=str(HERE.parent / "results" / "cache" / "predictions.jsonl"))
ap.add_argument("--out", default=str(HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-ranker")))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
KS = (1, 3, 5, 10, 20, 50)


from dense_ranker import DenseRanker


def rrf(*rankings, k: int = 60):
    n = len(rankings[0]); s = np.zeros(n)
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] += 1.0 / (k + pos)
    return list(np.argsort(-s, kind="stable"))


def rank_of(order, target):
    return int(list(order).index(target))


t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
data = NeedleData(agent.tok, HERE / "data" / "ag_news.json")
cache = PredictionCache(args.cache)
dense = DenseRanker(args.dense, str(agent.device))
print(f"loaded in {time.time()-t0:.0f}s on {agent.device}; dense={args.dense}")

needles = data.exp2_needles(args.n)
QUERIES = {"noul": question_query(NOUL_Q), "choice": question_query(CHOICE_Q), "option": "sports and athletics"}
RANKERS = {"bm25": lambda T, q: bm25_rank(T, q), "dense": lambda T, q: dense.rank(T, q),
           "hybrid": lambda T, q: rrf(bm25_rank(T, q), dense.rank(T, q))}

# pools: exp2 haystacks (~70 passages, per item), the full non-sports pool, and the cleaned pool
sc = predict_many(agent, [(a, CHOICE_Q) for a in data.non_sports], 16, cache=cache)
clean_pool = [a for a, r in zip(data.non_sports, sc) if r["probabilities"]["Sports"] <= args.clean]
print(f"pool: {len(data.non_sports)} non-sports; clean pool: {len(clean_pool)} (dropped {len(data.non_sports)-len(clean_pool)} "
      f"with Laya P(Sports) > {args.clean})  [{cache.stats()}]")
POOLS = {"exp2 (~70)": lambda i: data.exp2_haystack(i, needles[i]),
         f"full ({len(data.non_sports)})": lambda i: data.non_sports,
         f"clean ({len(clean_pool)})": lambda i: clean_pool}

R = {"n": args.n, "dense": args.dense, "rows": []}
for pool_name, pool_fn in POOLS.items():
    print(f"\n=== pool: {pool_name} ===")
    print(f"  {'ranker':8}{'query':8}" + "".join(f"{'R@'+str(k):>7}" for k in KS) + f"{'med rank':>10}{'m':>6}")
    for qname, query in QUERIES.items():
        for rname, rfn in RANKERS.items():
            ranks, ms = [], []
            for i, nd in enumerate(needles):
                hay = pool_fn(i)
                # needle goes in the middle; rankers are order-invariant, this just keeps index bookkeeping honest
                texts = hay[:len(hay) // 2] + [nd] + hay[len(hay) // 2:]
                tgt = len(hay) // 2
                r = rank_of(rfn(texts, query), tgt)
                ranks.append(r); ms.append(len(texts))
                R["rows"].append({"pool": pool_name, "query": qname, "ranker": rname, "i": i, "rank": r, "m": len(texts)})
            rec = {k: sum(r < k for r in ranks) / len(ranks) for k in KS}
            print(f"  {rname:8}{qname:8}" + "".join(f"{rec[k]:>7.2f}" for k in KS)
                  + f"{statistics.median(ranks):>10.0f}{statistics.median(ms):>6.0f}")

print(f"\ndense encoder: {dense.encoded} passages in {dense.seconds:.1f}s = {dense.seconds/max(1,dense.encoded)*1000:.1f} ms/passage")
cache.close()
json.dump(R, open(OUT / "results.json", "w"), indent=1)
print(f"written {OUT}/results.json | total {time.time()-t0:.0f}s")
