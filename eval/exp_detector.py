#!/usr/bin/env python3
"""Which per-passage question actually detects the needle? Single articles, no haystack.

30 sports articles (the Exp 2 needles) vs 300 non-sports articles, four detectors. AUC + the
false-positive rate at a threshold that catches 80% of needles — that FP rate times ~69 passages
per document is the expected number of false alarms per `none` document under max-aggregation.
"""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya
from chunklaya import predict_many
from needle_data import CHOICE_Q, NeedleData
from metrics import auc

HERE = Path(__file__).resolve().parent
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
data = NeedleData(agent.tok, HERE / "data" / "ag_news.json")
import random
needles = data.exp2_needles(30)
neg = random.Random(7).sample(data.non_sports, 300)

DETECTORS = {
    "noul: contain a news item about sports": ({"type": "noul", "instructions": "Does the text contain a news item about sports or athletics?"}, lambda r: r["noul"]),
    "noul: is this text about sports":        ({"type": "noul", "instructions": "Is this text about sports or athletics?"}, lambda r: r["noul"]),
    "noul: statement 'This is sports news.'": ({"type": "noul", "instructions": "This is a sports news article."}, lambda r: r["noul"]),
    "noul: with criteria descriptions":       ({"type": "noul", "instructions": "Is the text about sports?",
                                                "criteria": {"true": "the text reports on a sport, athlete, team, match, or league",
                                                             "false": "the text is about something else: politics, business, science, technology"}}, lambda r: r["noul"]),
    "choice 4-way → P(Sports)":               (CHOICE_Q, lambda r: r["probabilities"]["Sports"]),
    "choice 2-way sports/other":              ({"type": "choice", "instructions": "Which category does this text belong to?",
                                                "criteria": {"sports": "sports and athletics", "other": "anything else"}}, lambda r: r["probabilities"]["sports"]),
}
out = {}
print(f"{'detector':44} {'AUC':>6} {'needle q50':>11} {'neg q50':>8} {'neg q99':>8} {'FP@80%rec':>10} {'≈FP/doc':>8}")
for name, (q, f) in DETECTORS.items():
    t = time.time()
    rp = [f(r) for r in predict_many(agent, [(a, q) for a in needles], 32)]
    rn = [f(r) for r in predict_many(agent, [(a, q) for a in neg], 32)]
    thr = sorted(rp)[len(rp) // 5]                       # catches 80% of needles
    fp = sum(x >= thr for x in rn) / len(rn)
    out[name] = {"needle": rp, "neg": rn, "auc": auc(rp, rn), "thr80": thr, "fp": fp}
    print(f"{name:44} {auc(rp, rn):6.3f} {statistics.median(rp):11.3f} {statistics.median(rn):8.3f} {sorted(rn)[int(.99*len(rn))]:8.3f} {fp:10.1%} {fp*69:8.1f}")
json.dump(out, open(HERE.parent / "results" / "2026-09-20-para" / "detectors.json", "w"), indent=1)
