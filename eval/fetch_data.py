#!/usr/bin/env python3
"""Regenerate eval/data/ag_news.json: the first 1200 rows of the AG News test split.

The file is committed so the numbers in results/ reproduce exactly; this script documents where it
came from. Rows are fetched in order from the Hugging Face datasets-server, so a rerun yields the
same file. Dataset: fancyzhx/ag_news (the standard HF mirror of Zhang, Zhao & LeCun 2015's AG News
corpus). Labels: World / Sports / Business / Sci-Tech.

    python eval/fetch_data.py            # writes eval/data/ag_news.json (~315 KB)
"""
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "ag_news.json"
N = 1200

rows, feats = [], {}
for off in range(0, N, 100):
    url = ("https://datasets-server.huggingface.co/rows?dataset=fancyzhx/ag_news&config=default"
           f"&split=test&offset={off}&length={min(100, N - off)}")
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"user-agent": "chunklaya/0.1"}), timeout=60))
    feats = {f["name"]: f["type"] for f in d["features"]}
    rows += [r["row"] for r in d["rows"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump({"labels": feats["label"]["names"], "rows": rows}, open(OUT, "w"))
print(f"wrote {OUT}: {len(rows)} rows, labels {feats['label']['names']}")
