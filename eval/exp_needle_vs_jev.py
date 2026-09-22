#!/usr/bin/env python3
"""chunklaya on jev-decision-bench's needle tasks, item for item against Jev.

The needle tasks regenerate deterministically (seed 7, SQuAD filler), so the 60 items at each size
are byte-identical to the ones behind the published per-item Jev output in
results/jev-decision-bench-needle/. That makes this a paired comparison on the same documents --
not our own Jev run, but the same questions on the same haystacks.

Three arms, all reading the same `documents` string:

    raw        one Laya forward pass over the truncated head (max_len 1024) -- no harness
    noul       paragraphs + the benchmark's own noul asked per passage, aggregated max
    choice     paragraphs + a choice detector per passage, aggregated max  (README's recipe)

--prefilter bm25 --top-k 1 puts the harness's locate step in front of the noul and choice arms:
BM25 picks the chunk, Laya verifies it. That is the configuration that matches Jev on these tasks
(results/README.md); the flag-less run scores every chunk and is there for the comparison.

The benchmark's state is a dict {documents, color, object, code}; ChunkLaya takes a string, so the
color/object/code are inlined into each item's question and only `documents` is chunked. That is the
same information Jev got, arranged the way a chunked harness has to arrange it.

    .venv/bin/python eval/exp_needle_vs_jev.py --sizes 1k,12k,24k
    .venv/bin/python eval/exp_needle_vs_jev.py --sizes 1k,12k,24k --arms noul,choice --prefilter bm25 --top-k 1
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import ChunkLaya, PredictionCache, predict_many
from metrics import auc, auc_ci, recall_at_spec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="1k,12k,24k", help="comma list; only sizes with Jev output are comparable")
ap.add_argument("--arms", default="raw,noul,choice")
ap.add_argument("--n", type=int, default=0, help="limit items per size (0 = all 60)")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--chunk-tokens", type=int, default=750)
ap.add_argument("--agg", default="max", choices=["max", "any", "mean"])
ap.add_argument("--prefilter", default=None, choices=[None, "bm25"])
ap.add_argument("--top-k", type=int, default=1)
ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "needle_jev.jsonl"))
ap.add_argument("--out", default=None)
args = ap.parse_args()

SIZES = args.sizes.split(",")
ARMS = args.arms.split(",")
OUT = Path(args.out) if args.out else ROOT / "results" / (time.strftime("%Y-%m-%d") + "-needle-vs-jev")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = None if args.cache == "none" else PredictionCache(args.cache)
JEV = ROOT / "results" / "jev-decision-bench-needle"


from jev_questions import choice_q, noul_q


t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
print(f"loaded on {agent.device} in {time.time()-t0:.0f}s; "
      f"max_len={agent.cfg['max_len']} head_max_len={agent.cfg['head_max_len']}", flush=True)
predict_many(agent, [("warm up", noul_q("amber", "anchor", "1234"))])

cj = ChunkLaya(agent, chunk_tokens=args.chunk_tokens, batch_size=args.batch,
               mode="paragraphs", cache=CACHE, prefilter=args.prefilter, top_k=args.top_k)

REPORT = {"sizes": {}, "config": {"chunk_tokens": args.chunk_tokens, "agg": args.agg,
                                  "mode": "paragraphs", "device": str(agent.device),
                                  "prefilter": args.prefilter, "top_k": args.top_k}}

for size in SIZES:
    task = json.load(open(ROOT / "tasks" / f"needle_{size}.json"))
    items = task["items"][: args.n] if args.n else task["items"]

    jev_rows = {}
    jf = JEV / f"jev_needle_{size}.jsonl"
    if jf.exists():
        for line in open(jf):
            r = json.loads(line)
            jev_rows[r["id"]] = r

    print(f"\n=== needle_{size}: {len(items)} items, "
          f"{'jev rows ' + str(len(jev_rows)) if jev_rows else 'NO jev output'} ===", flush=True)

    rows = []
    for k, it in enumerate(items):
        st = it["state"]
        docs, color, obj, code = st["documents"], st["color"], st["object"], st["code"]
        nq, cq = noul_q(color, obj, code), choice_q(color, obj, code)
        row = {"id": it["id"], "gold": it["gold"], "depth": it["meta"]["depth_fraction"],
               "approx_tokens": it["meta"]["approx_tokens"]}

        if "raw" in ARMS:
            t = time.time()
            r = predict_many(agent, [(docs, nq)], max_len=agent.cfg["max_len"], cache=CACHE)[0]
            row["raw"] = {"p": float(r["noul"]), "s": time.time() - t}

        if "noul" in ARMS:
            t = time.time()
            a = cj.ask(docs, {"q": nq}, agg={"q": args.agg})
            row["noul"] = {"p": float(a["answers"]["q"]["noul"]), "s": time.time() - t,
                           "n_chunks": a["n_chunks"], "n_scored": a["answers"]["q"]["n_scored"],
                           "in_tok": a["usage"]["input_tokens"]}

        if "choice" in ARMS:
            t = time.time()
            a = cj.ask(docs, {"q": nq}, agg={"q": args.agg}, detectors={"q": (cq, "yes")})
            row["choice"] = {"p": float(a["answers"]["q"]["noul"]), "s": time.time() - t,
                             "n_chunks": a["n_chunks"], "n_scored": a["answers"]["q"]["n_scored"],
                             "in_tok": a["usage"]["input_tokens"]}

        if it["id"] in jev_rows:
            j = jev_rows[it["id"]]
            row["jev"] = {"p": j["preds"]["q"]["p"], "wall_ms": j.get("wall_ms"),
                          "server_ms": j.get("server_ms"), "in_tok": j.get("in_tok")}
        rows.append(row)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(items)}  {time.time()-t0:.0f}s elapsed", flush=True)

    # ---- scoring -------------------------------------------------------------
    stats = {}
    arms = [a for a in ARMS] + (["jev"] if jev_rows else [])
    for arm in arms:
        sub = [r for r in rows if arm in r]
        if not sub:
            continue
        pos = [r[arm]["p"] for r in sub if r["gold"]]
        neg = [r[arm]["p"] for r in sub if not r["gold"]]
        a, lo, hi = auc_ci(pos, neg)
        acc = sum((r[arm]["p"] >= 0.5) == r["gold"] for r in sub) / len(sub)
        secs = [r[arm]["s"] for r in sub if "s" in r[arm]]
        stats[arm] = {
            "n": len(sub), "acc@0.5": acc, "auc": a, "auc_lo": lo, "auc_hi": hi,
            "recall@spec90": recall_at_spec(pos, neg) if pos and neg else float("nan"),
            "mean_p_pos": sum(pos) / len(pos) if pos else float("nan"),
            "mean_p_neg": sum(neg) / len(neg) if neg else float("nan"),
            "s_per_doc": (sum(secs) / len(secs)) if secs else None,
            "n_chunks": (sum(r[arm]["n_chunks"] for r in sub) / len(sub)) if "n_chunks" in sub[0][arm] else None,
            "n_scored": (sum(r[arm]["n_scored"] for r in sub) / len(sub)) if "n_scored" in sub[0][arm] else None,
        }
        s = stats[arm]
        chunks = f" chunks={s['n_chunks']:.1f}" if s["n_chunks"] else ""
        if s["n_scored"] is not None and s["n_scored"] != s["n_chunks"]:
            chunks += f" scored={s['n_scored']:.1f}"
        secs_s = f" {s['s_per_doc']:.2f}s/doc" if s["s_per_doc"] else ""
        print(f"  {arm:7} acc@0.5={acc:.3f}  AUC={a:.3f} [{lo:.3f},{hi:.3f}]  "
              f"rec@spec90={s['recall@spec90']:.3f}  P(yes|true)={s['mean_p_pos']:.3f} "
              f"P(yes|false)={s['mean_p_neg']:.3f}{chunks}{secs_s}", flush=True)

    REPORT["sizes"][size] = {"stats": stats, "rows": rows}

if CACHE:
    CACHE.flush()
    print(f"\ncache: {CACHE.stats()}")

(OUT / "results.json").write_text(json.dumps(REPORT, indent=1))
print(f"\nwrote {OUT/'results.json'}")
