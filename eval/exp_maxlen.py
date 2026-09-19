#!/usr/bin/env python3
"""Does laya-multilingual degrade when max_len is raised past its 1024 training length?

Exp 1 (length only): AG News 4-way choice. State = target article + same-class filler,
   built to ~700 / ~1800 / ~3800 tokens. Every part of the state points to the same label,
   so any accuracy drop at 2048/4096 is the decision head failing at length, not missing evidence.
Exp 2 (needle): noul "does the text contain a sports item?". Haystack = non-sports articles
   (~3800 tokens), one sports article at start/middle/end, or absent. At max_len=1024 the
   state is truncated to the first ~990 tokens, so middle/end needles are invisible; at 4096
   they are visible iff extrapolation works.
"""
import argparse, json, random, statistics, time
from pathlib import Path
import torch
import laya
from laya.common import build_sequence

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30, help="items per class (exp1) / per condition (exp2)")
ap.add_argument("--exp", choices=["1", "2", "both"], default="both")
ap.add_argument("--device", default=None)
ap.add_argument("--data", default=str(HERE / "data" / "ag_news.json"))
ap.add_argument("--out", default=None, help="results dir; default results/<date>-maxlen/")
args = ap.parse_args()
OUT = Path(args.out) if args.out else HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-maxlen")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)
D = json.load(open(args.data))
LABELS = D["labels"]
by_cls = {l: [r["text"] for r in D["rows"] if LABELS[r["label"]] == l] for l in LABELS}

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=args.device)
tok = agent.tok
print(f"loaded in {time.time()-t0:.1f}s on {agent.device}, dtype {agent.dtype}, cfg max_len={agent.cfg['max_len']} head_max_len={agent.cfg['head_max_len']}")

def ntok(s): return len(tok(s, add_special_tokens=False)["input_ids"])

def fill_to(parts, pool, target_tokens, exclude):
    """Append articles from pool until the joined text reaches target_tokens."""
    pool = [p for p in pool if p not in exclude]; random.shuffle(pool)
    out = list(parts); n = ntok("\n\n".join(out))
    for p in pool:
        if n >= target_tokens: break
        out.append(p); n += ntok(p) + 2
    return out

def run(state, q, max_len):
    agent.cfg["max_len"] = max_len
    ids, markers = build_sequence(tok, state, agent._to_internal(q), max_len, agent.cfg["head_max_len"])
    t = time.time()
    r = agent.predict(state, {"q": q})["answers"]["q"]
    return r, (time.time() - t) * 1000, len(ids)

def mem():
    try: return f"{torch.mps.driver_allocated_memory()/2**30:.2f} GB mps"
    except Exception: return "n/a"

CHOICE_Q = {"type": "choice", "instructions": "Which news category does this text belong to?",
            "criteria": {"World": "international and political news", "Sports": "sports and athletics",
                         "Business": "business, markets, and finance", "Sci/Tech": "science and technology"}}
NOUL_Q = {"type": "noul", "instructions": "Does the text contain a news item about sports or athletics?"}

# warm-up
run("warm up", NOUL_Q, 1024)
results = {"device": str(agent.device), "exp1": [], "exp2": []}

if args.exp in ("1", "both"):
    print("\n=== Exp 1: length-only (same-class filler), 4-way choice ===")
    items = []
    for cls in LABELS:
        for tgt in random.sample(by_cls[cls], args.n):
            items.append((cls, tgt))
    # (name, state_target_tokens, max_len)
    conds = [("S@1024", 700, 1024), ("L@1024(trunc)", 3800, 1024), ("M@2048", 1800, 2048), ("L@4096", 3800, 4096)]
    for name, tt, ml in conds:
        rows = []
        for cls, tgt in items:
            state = "\n\n".join(fill_to([tgt], by_cls[cls], tt, {tgt}))
            r, ms, L = run(state, CHOICE_Q, ml)
            rows.append({"cond": name, "gold": cls, "pred": r["choice"], "p_gold": r["probabilities"][cls],
                         "conf": r["confidence"], "ms": ms, "seq_len": L})
        acc = sum(x["pred"] == x["gold"] for x in rows) / len(rows)
        print(f"  {name:14} n={len(rows):3}  acc={acc:.3f}  P(gold)={statistics.mean(x['p_gold'] for x in rows):.3f}  "
              f"conf={statistics.mean(x['conf'] for x in rows):.3f}  seq_len≈{int(statistics.median(x['seq_len'] for x in rows))}  "
              f"ms/item={statistics.median(x['ms'] for x in rows):.0f}  [{mem()}]")
        results["exp1"] += rows

if args.exp in ("2", "both"):
    print("\n=== Exp 2: needle (sports article in non-sports haystack), noul ===")
    non_sports = [t for l in LABELS if l != "Sports" for t in by_cls[l]]
    needles = random.sample(by_cls["Sports"], args.n)
    def build(needle, pos):
        hay = fill_to([], non_sports, 3800 - (ntok(needle) if needle else 0), set())
        if needle is None: parts = hay
        elif pos == "start": parts = [needle] + hay
        elif pos == "end": parts = hay + [needle]
        else: parts = hay[:len(hay)//2] + [needle] + hay[len(hay)//2:]
        return "\n\n".join(parts)
    for ml in (1024, 4096):
        for pos in ("start", "middle", "end", "none"):
            rows = []
            for i in range(args.n):
                state = build(None if pos == "none" else needles[i], pos)
                r, ms, L = run(state, NOUL_Q, ml)
                rows.append({"max_len": ml, "pos": pos, "gold": pos != "none", "p_true": r["noul"], "ms": ms, "seq_len": L})
            pt = [x["p_true"] for x in rows]
            hit = sum((p >= 0.5) == rows[0]["gold"] for p in pt) / len(pt)
            print(f"  max_len={ml}  needle={pos:6}  n={len(rows)}  mean P(true)={statistics.mean(pt):.3f}  "
                  f"median={statistics.median(pt):.3f}  correct@0.5={hit:.3f}  seq_len≈{int(statistics.median(x['seq_len'] for x in rows))}  "
                  f"ms/item={statistics.median(x['ms'] for x in rows):.0f}")
            results["exp2"] += rows

json.dump(results, open(OUT / "results.json", "w"), indent=1)
print(f"\nwritten {OUT}/results.json  |  total {time.time()-t0:.0f}s  |  peak {mem()}")
