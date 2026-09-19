#!/usr/bin/env python3
"""Rejoin jev-decision-bench's published per-item needle results against needle depth.

Their RESULTS.md reports only the aggregate (1.000 at every size). This regenerates the task files
(deterministic: seed 7, SQuAD filler) via third_party/probes_long_context.py and bins by depth, so the
numbers are comparable to eval/analyze_maxlen.py's start/middle/end breakdown for Laya.

    python eval/jev_needle_by_depth.py          # regenerates tasks/ on first run (~40 HF requests)
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS, RES = ROOT / "tasks", ROOT / "results" / "jev-decision-bench-needle"

if not (TASKS / "needle_24k.json").exists():
    subprocess.run([sys.executable, "probes_long_context.py"], cwd=ROOT / "third_party", check=True)

def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")

BINS = [(0.0, 0.34, "early"), (0.34, 0.67, "middle"), (0.67, 1.01, "late")]
for size in ("1k", "12k", "24k"):
    items = {it["id"]: it for it in json.load(open(TASKS / f"needle_{size}.json"))["items"]}
    rows = []
    for line in open(RES / f"jev_needle_{size}.jsonl"):
        p = json.loads(line); it = items[p["id"]]
        rows.append({"depth": it["meta"]["depth_fraction"], "gold": it["gold"], "p": p["preds"]["q"]["p"], "tok": it["meta"]["approx_tokens"]})
    n = len(rows); tok = sorted(r["tok"] for r in rows)[n // 2]
    pos, neg = [r["p"] for r in rows if r["gold"]], [r["p"] for r in rows if not r["gold"]]
    print(f"needle_{size}  n={n}  ~{tok} tok  acc@0.5={sum((r['p']>=.5)==r['gold'] for r in rows)/n:.3f}  AUC={auc(pos, neg):.3f}")
    for lo, hi, name in BINS:
        sub = [r for r in rows if lo <= r["depth"] < hi]
        sp, sn = [r["p"] for r in sub if r["gold"]], [r["p"] for r in sub if not r["gold"]]
        print(f"    {name:6} n={len(sub):2}  acc={sum((r['p']>=.5)==r['gold'] for r in sub)/len(sub):.3f}  "
              f"P(true|yes)={sum(sp)/len(sp):.3f}  P(true|no)={sum(sn)/len(sn):.3f}")
