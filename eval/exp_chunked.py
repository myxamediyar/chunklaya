#!/usr/bin/env python3
"""Does chunk → gate → aggregate recover what raw long-context Laya loses?

Every arm sees byte-identical states (eval/needle_data.py), so comparisons are paired.

Exp 1  whole-state 4-way choice, same-class filler, ~3800 tokens
         raw@4096  vs  chunked 750/375: mixture (gated), loglinear (gated), stack
Exp 2  sports needle in ~3800 tokens of non-sports, at start / middle / end / absent
         raw@4096  vs  chunked configs from --configs: token windows "750/375" etc., or "para" (one
         passage per chunk); per-chunk detector from --detector: the noul itself, or the 4-way
         choice's P(Sports). --clean-haystack P drops pool articles the model rates P(Sports) > P.
         Reports AUC with bootstrap 95% CI, recall@spec90, tokens/doc, and the share of documents
         in the ambiguous [0.3, 0.7] band.
Exp 3  needle at token offset 0 / 250 / 500 / 700 inside ONE ~850-token window, raw@1024
         → the shape of the positional decay inside the first 1000 tokens

Per-row predictions are cached (--cache, default results/cache/predictions.jsonl) so reruns with a
different pool, threshold, or aggregation are close to free.

    .venv/bin/python eval/exp_chunked.py --n 30                                        # Exp 1-3, ~10 min on M4 Pro / MPS
    .venv/bin/python eval/exp_chunked.py --n 100 --exp 2 --configs para,750/750 --detector choice
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import ChunkLaya, PredictionCache, predict_many, noul_any, noul_max, choice_mixture, choice_loglinear
from needle_data import CHOICE_Q, NOUL_Q, LABELS, NeedleData
from metrics import auc, auc_ci, recall_at_spec, quartiles

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--exp", default="all", choices=["1", "2", "3", "all"])
ap.add_argument("--configs", default="750/375,750/750,512/256,1024/512")
ap.add_argument("--out", default=None)
ap.add_argument("--cache", default=str(HERE.parent / "results" / "cache" / "predictions.jsonl"),
                help="row-level prediction cache (chunklaya.cache); 'none' to disable")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--clean-haystack", type=float, default=None, metavar="P",
                help="exp2: drop non-sports pool articles the model itself scores P(Sports) > P in isolation. "
                     "AG News labels are noisy (Olympics stories filed under World); this isolates position "
                     "recovery from label noise.")
ap.add_argument("--detector", default="noul", choices=["noul", "choice"],
                help="exp2 per-chunk detector: the noul itself, or the 4-way choice's P(Sports)")
args = ap.parse_args()
OUT = Path(args.out) if args.out else HERE.parent / "results" / (time.strftime("%Y-%m-%d") + "-chunked")
OUT.mkdir(parents=True, exist_ok=True)
CONFIGS = [c if c == "para" else tuple(int(x) for x in c.split("/")) for c in args.configs.split(",")]
CACHE = None if args.cache == "none" else PredictionCache(args.cache)


def make_cj(cfg):
    kw = dict(batch_size=args.batch, cache=CACHE)
    return ChunkLaya(agent, mode="paragraphs", **kw) if cfg == "para" else ChunkLaya(agent, cfg[0], cfg[1], **kw)


def cfg_name(cfg):
    return "para" if cfg == "para" else f"{cfg[0]}/{cfg[1]}"

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
assert agent.cfg["max_len"] == 1024, agent.cfg["max_len"]
data = NeedleData(agent.tok, HERE / "data" / "ag_news.json")
print(f"loaded on {agent.device} in {time.time()-t0:.0f}s; cfg max_len={agent.cfg['max_len']} head_max_len={agent.cfg['head_max_len']}")
predict_many(agent, [("warm up", NOUL_Q)])
R = {"n": args.n, "configs": CONFIGS, "exp1": [], "exp2": [], "exp3": []}


def timed(fn):
    t = time.time(); r = fn(); return r, time.time() - t


def run(name, exp): return args.exp in (exp, "all")


if run("exp1", "1"):
    print("\n=== Exp 1: whole-state choice, same-class filler ~3800 tok ===")
    items = data.exp1_items(args.n)
    states = [data.exp1_state(i, cls, tgt, 3800) for i, (cls, tgt) in enumerate(items)]
    cj = ChunkLaya(agent, 750, 375, batch_size=args.batch, cache=CACHE)
    raw, t_raw = timed(lambda: predict_many(agent, [(s, CHOICE_Q) for s in states], batch_size=2, max_len=4096, cache=CACHE))
    ch, t_ch = timed(lambda: [cj.ask(s, {"q": CHOICE_Q}) for s in states])
    st, t_st = timed(lambda: [cj.ask(s, {"q": CHOICE_Q}, agg={"q": "stack"}) for s in states])
    for (cls, _), s, r, c, k in zip(items, states, raw, ch, st):
        d = c["answers"]["q"]["chunks"]; D = [x["p"] for x in d]; w = [x["gate"] for x in d]
        keys = list(CHOICE_Q["criteria"]); gi = keys.index(cls)
        ll = choice_loglinear(D, w)
        R["exp1"].append({"gold": cls, "n_chunks": c["n_chunks"],
                          "raw": {"pred": r["choice"], "p_gold": r["probabilities"][cls]},
                          "mixture": {"pred": c["answers"]["q"]["choice"], "p_gold": c["answers"]["q"]["probabilities"][cls]},
                          "loglinear": {"pred": keys[int(ll.argmax())], "p_gold": float(ll[gi])},
                          "stack": {"pred": k["answers"]["q"]["choice"], "p_gold": k["answers"]["q"]["probabilities"][cls]},
                          "gates": w})
    for arm, T in (("raw", t_raw), ("mixture", t_ch), ("loglinear", None), ("stack", t_st)):
        acc = statistics.mean(x[arm]["pred"] == x["gold"] for x in R["exp1"])
        pg = statistics.mean(x[arm]["p_gold"] for x in R["exp1"])
        print(f"  {arm:10} acc={acc:.3f}  P(gold)={pg:.3f}" + (f"  {T/len(states)*1000:.0f} ms/state" if T else "  (from mixture's chunks)"))
    print(f"  chunks/state≈{statistics.median(x['n_chunks'] for x in R['exp1'])}  gate q50={statistics.median(g for x in R['exp1'] for g in x['gates']):.2f}")

if run("exp2", "2"):
    print("\n=== Exp 2: needle noul, ~3800 tok haystack ===")
    needles = data.exp2_needles(args.n)
    if args.clean_haystack is not None:
        sc = predict_many(agent, [(a, CHOICE_Q) for a in data.non_sports], args.batch, cache=CACHE)
        keep = [a for a, r in zip(data.non_sports, sc) if r["probabilities"]["Sports"] <= args.clean_haystack]
        print(f"  clean haystack: dropped {len(data.non_sports) - len(keep)} of {len(data.non_sports)} "
              f"non-sports articles with P(Sports) > {args.clean_haystack}")
        data.non_sports = keep
    hays = [data.exp2_haystack(i, nd) for i, nd in enumerate(needles)]
    POS = ("start", "middle", "end", "none")
    states = {pos: [data.place(h, nd, pos) for h, nd in zip(hays, needles)] for pos in POS}
    rows = {pos: [{"pos": pos, "i": i} for i in range(args.n)] for pos in POS}
    # raw @4096
    DET = {"q": (CHOICE_Q, "Sports")} if args.detector == "choice" else None
    raw_q = CHOICE_Q if args.detector == "choice" else NOUL_Q
    score = (lambda r: r["probabilities"]["Sports"]) if args.detector == "choice" else (lambda r: r["noul"])
    flat = [(s, raw_q) for pos in POS for s in states[pos]]
    raw, t_raw = timed(lambda: predict_many(agent, flat, batch_size=2, max_len=4096, cache=CACHE))
    k = 0
    for pos in POS:
        for i in range(args.n):
            rows[pos][i]["raw4096"] = score(raw[k]); rows[pos][i]["seq_len"] = raw[k]["seq_len"]
            rows[pos][i]["tok_raw4096"] = raw[k]["seq_len"]; k += 1
    print(f"  detector: {args.detector}")
    print(f"  raw@4096: {t_raw/len(flat)*1000:.0f} ms/state, seq≈{int(statistics.median(r['seq_len'] for p in POS for r in rows[p]))}")
    # chunked
    for cfg in CONFIGS:
        cj, name = make_cj(cfg), cfg_name(cfg)
        t = time.time(); nch = []
        for pos in POS:
            for i in range(args.n):
                r = cj.ask(states[pos][i], {"q": NOUL_Q}, detectors=DET)
                ps = [x["p"][1] for x in r["answers"]["q"]["chunks"]]
                rows[pos][i][f"max_{name}"] = noul_max(ps); rows[pos][i][f"any_{name}"] = noul_any(ps)
                rows[pos][i][f"chunks_{name}"] = ps; rows[pos][i][f"tok_{name}"] = r["usage"]["input_tokens"]; nch.append(r["n_chunks"])
        print(f"  chunked {name}: {(time.time()-t)/len(flat)*1000:.0f} ms/state, {statistics.median(nch):.0f} chunks/state")
    R["exp2"] = [r for pos in POS for r in rows[pos]]
    arms = ["raw4096"] + [f"{a}_{cfg_name(cfg)}" for cfg in CONFIGS for a in ("max", "any")]
    BAND = (0.3, 0.7)
    print(f"\n  AUC needle-present vs absent, bootstrap 95% CI (n={args.n}/cell); recall@spec90 in brackets")
    print(f"  {'arm':16}" + "".join(f"{p:>30}" for p in ("start", "middle", "end")) + f"{'none q50':>10}{'tok/doc':>9}{'in band':>9}")
    summary = {}
    for arm in arms:
        neg = [r[arm] for r in rows["none"]]
        cells, summ = [], {}
        for pos in ("start", "middle", "end"):
            p = [r[arm] for r in rows[pos]]
            a, lo, hi = auc_ci(p, neg)
            summ[pos] = {"auc": a, "ci": [lo, hi], "recall_spec90": recall_at_spec(p, neg)}
            cells.append(f"{a:.3f} [{lo:.2f},{hi:.2f}] r={summ[pos]['recall_spec90']:.2f}")
        allsc = [r[arm] for pos in POS for r in rows[pos]]
        band = sum(BAND[0] <= x <= BAND[1] for x in allsc) / len(allsc)
        tok = statistics.median(r[f"tok_{arm}" if arm == "raw4096" else "tok_" + arm.split("_", 1)[1]] for pos in POS for r in rows[pos])
        summ.update(none_q50=statistics.median(neg), tok_per_doc=tok, in_band=band)
        summary[arm] = summ
        print(f"  {arm:16}" + "".join(f"{c:>30}" for c in cells) + f"{statistics.median(neg):>10.3f}{tok:>9.0f}{band:>9.1%}")
    print(f"  'in band' = share of all documents whose score falls in [{BAND[0]}, {BAND[1]}] — the fraction a classifier.dev-style smart tier would escalate")
    R["exp2_summary"] = summary

if run("exp3", "3"):
    print("\n=== Exp 3: needle offset inside one ~950-tok window, raw@1024 ===")
    needles = data.exp2_needles(args.n)
    OFFS, TOTAL = (0, 250, 500, 700), 850   # head ≈ 36 tok; fill can overshoot one article; stays < 1024
    built = {o: [data.exp3_state(i, nd, o, TOTAL) for i, nd in enumerate(needles)] for o in OFFS}
    none = [data.exp3_state(i, None, 0, TOTAL)[0] for i in range(args.n)]
    flat = [(s, NOUL_Q) for o in OFFS for s, _ in built[o]] + [(s, NOUL_Q) for s in none]
    res, t = timed(lambda: predict_many(agent, flat, batch_size=8, max_len=1024, cache=CACHE))
    k = 0
    for o in OFFS:
        for i in range(args.n):
            R["exp3"].append({"offset_req": o, "offset": built[o][i][1], "gold": True, "p": res[k]["noul"], "seq_len": res[k]["seq_len"]}); k += 1
    for i in range(args.n):
        R["exp3"].append({"offset_req": -1, "offset": -1, "gold": False, "p": res[k]["noul"], "seq_len": res[k]["seq_len"]}); k += 1
    neg = [r["p"] for r in R["exp3"] if not r["gold"]]
    mx = max(r["seq_len"] for r in R["exp3"]); assert mx < 1024, f"truncated: max seq_len {mx}"
    print(f"  {t/len(flat)*1000:.0f} ms/state, seq max={mx} (<1024: untruncated)")
    print(f"  none: {quartiles(neg)}")
    for o in OFFS:
        sub = [r for r in R["exp3"] if r["offset_req"] == o]
        p = [r["p"] for r in sub]
        print(f"  offset≈{int(statistics.median(r['offset'] for r in sub)):4}  AUC={auc(p, neg):.3f}  recall@spec90={recall_at_spec(p, neg):.2f}  {quartiles(p)}")

if CACHE is not None:
    print(f"\ncache: {CACHE.stats()}"); CACHE.close()
json.dump(R, open(OUT / "results.json", "w"), indent=1)
print(f"\nwritten {OUT}/results.json  |  total {time.time()-t0:.0f}s")
