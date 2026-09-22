#!/usr/bin/env python3
"""Regenerate the two data files behind the 2026-09-22 scale experiments (gitignored: 17 MB, derived).

eval/data/ag_news_full.json      the whole AG News test split (7,600 rows; the first 1,200 are byte-identical
                                 to eval/data/ag_news.json), for haystacks up to 5,700 non-sports passages.
                                 eval/data/ag_news_full_relabels.json (committed) indexes into its rows.
eval/data/squad_contexts.json    every distinct SQuAD v1.1 context of >= 300 chars, train and validation, as
                                 filler for the 100k-1M-token needle haystacks. The train split's 17,864
                                 contexts (~3.4M tokens) mean a 1M-token haystack repeats nothing.

Both come straight from the datasets' parquet files on the Hugging Face hub, one download each -- the
datasets-server that eval/fetch_data.py uses rate-limits at this volume.

    .venv/bin/python eval/fetch_scale_data.py
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent / "data"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

t = pq.read_table(hf_hub_download("fancyzhx/ag_news", "data/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
rows = [{"text": r["text"], "label": r["label"]} for r in t]
json.dump({"labels": LABELS, "rows": rows}, open(DATA / "ag_news_full.json", "w"))
committed = json.load(open(DATA / "ag_news.json"))["rows"]
assert rows[: len(committed)] == committed, "AG News test split changed under us"
print(f"wrote ag_news_full.json: {len(rows)} rows")

out = {}
for split in ("train", "validation"):
    t = pq.read_table(hf_hub_download("rajpurkar/squad", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset"),
                      columns=["context"]).to_pylist()
    seen, ctx = set(), []
    for r in t:
        c = " ".join(r["context"].split())
        if len(c) >= 300 and c not in seen:
            seen.add(c); ctx.append(c)
    out[split] = ctx
    print(f"squad {split}: {len(ctx)} unique contexts")
json.dump(out, open(DATA / "squad_contexts.json", "w"), ensure_ascii=False)
print("wrote squad_contexts.json")
