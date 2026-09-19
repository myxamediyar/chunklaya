#!/usr/bin/env python3
"""Depth-stratified summary of an exp_maxlen.py run: AUC and recall@spec90 per needle position.

    python eval/analyze_maxlen.py results/2026-09-19-maxlen/results.json
"""
import json, statistics, sys

def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def q(xs):
    s = sorted(xs); n = len(s)
    return f"q25={s[n//4]:.3f} q50={s[n//2]:.3f} q75={s[3*n//4]:.3f}"

R = json.load(open(sys.argv[1]))
if R["exp1"]:
    print("Exp 1 — length only, 4-way choice, same-class filler")
    for cond in dict.fromkeys(r["cond"] for r in R["exp1"]):
        rows = [r for r in R["exp1"] if r["cond"] == cond]
        acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
        print(f"  {cond:14} n={len(rows):3}  acc={acc:.3f}  P(gold)={statistics.mean(r['p_gold'] for r in rows):.3f}  "
              f"seq≈{int(statistics.median(r['seq_len'] for r in rows))}  ms={statistics.median(r['ms'] for r in rows):.0f}")
if R["exp2"]:
    print("\nExp 2 — needle, noul; AUC of needle-present vs needle-absent (0.5 = chance)")
    for ml in sorted({r["max_len"] for r in R["exp2"]}):
        neg = [r["p_true"] for r in R["exp2"] if r["max_len"] == ml and r["pos"] == "none"]
        thr = sorted(neg)[int(0.9 * len(neg))]
        print(f"  max_len={ml}   none: {q(neg)}")
        for pos in ("start", "middle", "end"):
            p = [r["p_true"] for r in R["exp2"] if r["max_len"] == ml and r["pos"] == pos]
            if not p: continue
            print(f"    {pos:6}  AUC={auc(p, neg):.3f}   {q(p)}   recall@spec90(thr={thr:.3f})={sum(x >= thr for x in p)/len(p):.2f}")
