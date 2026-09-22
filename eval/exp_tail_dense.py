#!/usr/bin/env python3
"""Step 3c, second opinion from a different model.

exp_tail.py: every Laya phrasing shares the false-positive tail (same weights). bge-small is a different
model, so: 4-way choice full scan, and a passage only counts if its cosine to "sports and athletics"
clears a gate set on AG News *train* sports articles (their 5th percentile -- nothing is tuned on the
test needles). Simulated on the same nested haystacks as exp_tail.py.

    .venv/bin/python eval/exp_tail_dense.py --n 100
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import PredictionCache, predict_many
from dense_ranker import DenseRanker
from metrics import auc_ci, recall_at_spec
from needle_data import CHOICE_Q, NeedleData

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--sizes", default="70,300,1000,3000,5600")
ap.add_argument("--query", default="sports and athletics")
ap.add_argument("--gate-pct", type=float, default=5.0, help="percentile of train sports articles' cosine that sets the gate")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--relabels", default=str(HERE / "data" / "ag_news_full_relabels.json"))
ap.add_argument("--cache", default=str(HERE.parent / "results" / "cache" / "predictions.jsonl"))
ap.add_argument("--out", default=str(HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-tail-dense")))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
SIZES = [int(x) for x in args.sizes.split(",")]

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
cache = PredictionCache(args.cache)
needles = NeedleData(agent.tok, HERE / "data" / "ag_news.json").exp2_needles(args.n)
full = json.load(open(HERE / "data" / "ag_news_full.json"))
pool_all = [r["text"] for r in full["rows"] if full["labels"][r["label"]] != "Sports"]
rl = json.load(open(args.relabels))
row_of = {r["text"]: i for i, r in enumerate(full["rows"])}
sports_rows = {r["row"] for r in rl["rows"] if r["is_sports"]}
fp_rows = {r["row"] for r in rl["rows"] if not r["is_sports"]}
pool = [t for t in pool_all if row_of[t] not in sports_rows]
is_fp = np.array([row_of[t] in fp_rows for t in pool])

# gate from the train split's sports articles (independent of every test item)
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
tr = pq.read_table(hf_hub_download("fancyzhx/ag_news", "data/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
train_sports = [r["text"] for r in tr if r["label"] == 1]
train_sports = random.Random(1).sample(train_sports, 500)

dense = DenseRanker("BAAI/bge-small-en-v1.5", str(agent.device))
q = dense.embed([DenseRanker.QUERY_PREFIX + args.query])[0]
sim_pool, sim_needle, sim_train = dense.embed(pool) @ q, dense.embed(needles) @ q, dense.embed(train_sports) @ q
gate = float(np.percentile(sim_train, args.gate_pct))
res = predict_many(agent, [(x, CHOICE_Q) for x in pool + needles], 16, cache=cache)
p = np.array([r["probabilities"]["Sports"] for r in res]); P_pool, P_needle = p[:len(pool)], p[len(pool):]
print(f"loaded in {time.time()-t0:.0f}s; pool {len(pool)} ({int(is_fp.sum())} Laya false positives), {args.n} needles, {len(train_sports)} train sports articles")
print(f"cosine to '{args.query}':  train sports q05 {gate:.3f} q50 {np.median(sim_train):.3f} | test needles q05 {np.percentile(sim_needle,5):.3f} "
      f"q50 {np.median(sim_needle):.3f} | pool q50 {np.median(sim_pool):.3f} q99 {np.percentile(sim_pool,99):.3f}")
fp_hi = is_fp & (P_pool > 0.9)
print(f"Laya FPs > 0.9 ({int(fp_hi.sum())}): cosine q50 {np.median(sim_pool[fp_hi]):.3f} max {sim_pool[fp_hi].max():.3f}; "
      f"{int((sim_pool[fp_hi] >= gate).sum())} of them pass the gate at {gate:.3f}; "
      f"needles passing: {int((sim_needle >= gate).sum())}/{args.n}; pool passing: {int((sim_pool >= gate).sum())}/{len(pool)}")
for row in sorted([(P_pool[j], sim_pool[j], pool[j][:60]) for j in np.flatnonzero(fp_hi)], key=lambda x: -x[0]):
    print(f"   P={row[0]:.4f} cos={row[1]:.3f} {'PASS' if row[1] >= gate else 'cut '} {row[2]}")

perm = {i: np.array(random.Random(args.seed * 1000 + 500 + i).sample(range(len(pool)), len(pool))) for i in range(args.n)}
gated_pool = np.where(sim_pool >= gate, P_pool, 0.0); gated_needle = np.where(sim_needle >= gate, P_needle, 0.0)
R = {"gate": gate, "summary": {}}
print(f"\n  {'m':>5}{'arm':>16}{'AUC [95% CI]':>22}{'r@spec90':>10}{'none q50':>10}{'none q90':>10}{'needle q50':>12}")
for m in SIZES:
    m_eff = min(m, len(pool))
    for arm, (Sp, Sn) in {"4w": (P_pool, P_needle), "4w+dense gate": (gated_pool, gated_needle)}.items():
        pos, neg = [], []
        for i in range(args.n):
            hay = perm[i][:m_eff]
            neg.append(float(Sp[hay].max())); pos.append(max(float(Sp[hay].max()), float(Sn[i])))
        auc, lo, hi = auc_ci(pos, neg)
        summ = {"auc": auc, "ci": [lo, hi], "recall_spec90": recall_at_spec(pos, neg), "none_q50": statistics.median(neg),
                "none_q90": float(np.quantile(neg, .9)), "needle_q50": statistics.median(pos)}
        R["summary"][f"{m_eff}/{arm}"] = summ
        print(f"  {m_eff:>5}{arm:>16}{auc:>8.3f} [{lo:.2f},{hi:.2f}]{summ['recall_spec90']:>10.2f}{summ['none_q50']:>10.3f}"
              f"{summ['none_q90']:>10.3f}{summ['needle_q50']:>12.3f}")
cache.close()
json.dump(R, open(OUT / "results.json", "w"), indent=1)
print(f"written {OUT}/results.json | total {time.time()-t0:.0f}s")
