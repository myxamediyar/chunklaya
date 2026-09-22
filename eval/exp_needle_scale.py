#!/usr/bin/env python3
"""Step 3 (lexical): the Jev needle at 24k, 100k, 300k and 1M tokens.

jev-decision-bench stops at ~22k tokens, the most a 32k-context Jev call can hold. The harness has no
such ceiling: with prefilter="bm25", top_k=1 Laya scores one passage whatever the length, and the only
cost that grows with the document is chunking plus a stdlib BM25 pass. This builds the benchmark's
own 60 needles (third_party/probes_long_context.py: same seed, colors, objects, codes and depths)
into haystacks of unique SQuAD *train* contexts (eval/data/squad_contexts.json; 17,864 of them, so a
1M-token haystack repeats nothing) and runs:

    bm25@1   prefilter="bm25", top_k=1, choice detector        -- the recipe from the README
    bm25@5   same with top_k=5, the false-alarm term times 5
    full     every chunk scored (--full-n items, up to --full-max tokens): the O(m) arm, for the collapse

The 24k point here is built from the train pool, so it is not byte-identical to the benchmark's
items (those use the validation pool); results/2026-09-20-needle-vs-jev-prefilter has that run.

    .venv/bin/python eval/exp_needle_scale.py --sizes 24k,100k,300k,1m
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))
import laya
from chunklaya import ChunkLaya, PredictionCache
from chunklaya.prefilter import bm25_rank, question_query
from jev_questions import choice_q, noul_q
from metrics import auc_ci, recall_at_spec
from probes_long_context import CHARS_PER_TOKEN, build_haystack, make_needles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="24k,100k,300k,1m")
ap.add_argument("--n", type=int, default=60, help="items per size (max 60)")
ap.add_argument("--full-n", type=int, default=10, help="items per size for the every-chunk arm (0 = skip)")
ap.add_argument("--full-max", type=int, default=300_000, help="largest size (tokens) the every-chunk arm runs at")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "needle_jev.jsonl"))
ap.add_argument("--out", default=str(ROOT / "results" / (time.strftime("%Y-%m-%d") + "-needle-scale")))
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)


def tokens(s: str) -> int:
    return int(float(s[:-1]) * (1_000_000 if s[-1] == "m" else 1_000))


SIZES = [(s, tokens(s)) for s in args.sizes.split(",")]
t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
CACHE = PredictionCache(args.cache)
pool = json.load(open(HERE / "data" / "squad_contexts.json"))["train"]
needles = make_needles()[: args.n]
print(f"loaded on {agent.device} in {time.time()-t0:.0f}s; filler pool {len(pool)} unique contexts, "
      f"{sum(map(len, pool))/CHARS_PER_TOKEN/1e6:.2f}M tokens", flush=True)
ARMS = {"bm25@1": ChunkLaya(agent, mode="paragraphs", batch_size=args.batch, cache=CACHE, prefilter="bm25", top_k=1),
        "bm25@5": ChunkLaya(agent, mode="paragraphs", batch_size=args.batch, cache=CACHE, prefilter="bm25", top_k=5),
        "full": ChunkLaya(agent, mode="paragraphs", batch_size=args.batch, cache=CACHE)}

REPORT = {"sizes": {}, "pool": "squad train, unique contexts"}
for name, size in SIZES:
    rows, breakdown = [], None
    print(f"\n=== {name} (~{size:,} tokens) ===", flush=True)
    for k, nd in enumerate(needles):
        docs, depth = build_haystack(pool, nd, size * CHARS_PER_TOKEN, 10**12)
        code = nd["asked_code"]
        nq, cq = noul_q(nd["color"], nd["object"], code), choice_q(nd["color"], nd["object"], code)
        row = {"i": nd["i"], "gold": nd["gold"], "depth": depth, "approx_tokens": len(docs) // CHARS_PER_TOKEN}
        if breakdown is None:   # where the time goes, once per size: chunking, ranking, Laya
            t = time.time(); chunks = ARMS["bm25@1"].chunks(docs); t_chunk = time.time() - t
            t = time.time(); bm25_rank([c.text for c in chunks], question_query(nq)); t_rank = time.time() - t
            breakdown = {"n_chunks": len(chunks), "tokens": chunks[-1].tok_end, "s_chunk": t_chunk, "s_rank": t_rank}
        for arm, cj in ARMS.items():
            if arm == "full" and (k >= args.full_n or size > args.full_max):
                continue
            t = time.time()
            a = cj.ask(docs, {"q": nq}, detectors={"q": (cq, "yes")})
            row[arm] = {"p": float(a["answers"]["q"]["noul"]), "s": time.time() - t,
                        "n_chunks": a["n_chunks"], "n_scored": a["answers"]["q"]["n_scored"]}
        rows.append(row)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(needles)}  {time.time()-t0:.0f}s elapsed  "
                  f"({row['bm25@1']['s']:.1f}s/doc bm25@1, {row['bm25@1']['n_chunks']} chunks)", flush=True)
    stats = {"breakdown": breakdown}
    print(f"  one document: {breakdown['n_chunks']} chunks, {breakdown['tokens']:,} tokens; "
          f"chunking {breakdown['s_chunk']:.1f}s, BM25 {breakdown['s_rank']:.1f}s")
    for arm in ARMS:
        sub = [r for r in rows if arm in r]
        if not sub:
            continue
        pos = [r[arm]["p"] for r in sub if r["gold"]]; neg = [r[arm]["p"] for r in sub if not r["gold"]]
        a, lo, hi = auc_ci(pos, neg)
        acc = sum((r[arm]["p"] >= 0.5) == r["gold"] for r in sub) / len(sub)
        s = {"n": len(sub), "acc@0.5": acc, "auc": a, "auc_lo": lo, "auc_hi": hi,
             "recall@spec90": recall_at_spec(pos, neg) if pos and neg else float("nan"),
             "s_per_doc": statistics.mean(r[arm]["s"] for r in sub),
             "n_chunks": statistics.mean(r[arm]["n_chunks"] for r in sub),
             "n_scored": statistics.mean(r[arm]["n_scored"] for r in sub)}
        stats[arm] = s
        print(f"  {arm:7} n={s['n']:2}  acc@0.5={acc:.3f}  AUC={a:.3f} [{lo:.3f},{hi:.3f}]  rec@spec90={s['recall@spec90']:.3f}"
              f"  chunks={s['n_chunks']:.0f} scored={s['n_scored']:.0f}  {s['s_per_doc']:.2f}s/doc", flush=True)
    REPORT["sizes"][name] = {"stats": stats, "rows": rows}

CACHE.close()
(OUT / "results.json").write_text(json.dumps(REPORT, indent=1))
print(f"\nwrote {OUT/'results.json'} | total {time.time()-t0:.0f}s")
