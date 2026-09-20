#!/usr/bin/env python3
"""Run the needle tasks against TypeSafe's Jev and write per-item output.

This is the arm the README leaves open: results/jev-decision-bench-needle/ is someone else's
published run, so it is fixed at jev-1.13.0 / 2026-09-18 and covers only 1k, 12k and 24k. This
script produces the same jsonl shape from our own key, so any size or any question can be asked.

The question and the dict state go to Jev exactly as probes_long_context.py builds them -- no
inlining, no chunking. That is the point of the comparison: Jev reads the whole haystack in one
call, chunklaya does not.

    export TYPESAFE_API_KEY=...          # or stash it at ~/.garlic/typesafe-api-key
    .venv/bin/python eval/run_jev_needle.py --sizes 1k,12k,24k --out results/jev-ours

Cost at jev-1.13 pricing ($0.042 / M input tokens): ~2.6M tokens for 1k+12k+24k at n=60, so
roughly $0.11 for the full sweep. Output tokens are billed at $0.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="1k,12k,24k")
ap.add_argument("--n", type=int, default=0, help="limit items per size (0 = all)")
ap.add_argument("--model", default="jev-latest")
ap.add_argument("--out", default=str(ROOT / "results" / "jev-ours"))
ap.add_argument("--url", default="https://api.typesafe.ai/v1/systemone")
ap.add_argument("--retries", type=int, default=5)
ap.add_argument("--dry-run", action="store_true", help="price the run, send nothing")
args = ap.parse_args()


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k.strip()
    p = Path.home() / ".garlic" / "typesafe-api-key"
    if p.exists():
        return p.read_text().strip()
    sys.exit("no TYPESAFE_API_KEY in the environment and no ~/.garlic/typesafe-api-key")


def ask(key, state, question):
    body = json.dumps({"model": args.model, "state": state,
                       "questions": {"q": question}}).encode()
    req = urllib.request.Request(args.url, data=body, method="POST", headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    last = None
    for attempt in range(args.retries):
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r), (time.time() - t) * 1000
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            # 4xx other than 429 is our bug, not the server's -- stop rather than burn retries
            if e.code != 429 and 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            last = f"HTTP {e.code}: {detail}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"gave up after {args.retries}: {last}")


OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
key = None if args.dry_run else api_key()

for size in args.sizes.split(","):
    task = json.load(open(ROOT / "tasks" / f"needle_{size}.json"))
    items = task["items"][: args.n] if args.n else task["items"]
    question = task["question"]
    dest = OUT / f"jev_needle_{size}.jsonl"

    done = set()
    if dest.exists():                      # resume: a paid run should never repeat an item
        for line in open(dest):
            done.add(json.loads(line)["id"])
    todo = [it for it in items if it["id"] not in done]

    approx = sum(it["meta"]["approx_tokens"] for it in todo)
    print(f"needle_{size}: {len(todo)} to send ({len(done)} already in {dest.name}), "
          f"~{approx/1000:.0f}k tokens, ~${approx*0.042/1e6:.3f}", flush=True)
    if args.dry_run:
        continue

    with open(dest, "a") as f:
        for k, it in enumerate(todo):
            res, wall = ask(key, it["state"], question)
            a = res["answers"]["q"]
            u = res.get("usage", {})
            f.write(json.dumps({
                "id": it["id"],
                "preds": {"q": {"p": a.get("noul", a.get("probability"))}},
                "wall_ms": round(wall),
                "server_ms": res.get("server_ms"),
                "in_tok": u.get("input_tokens"),
                "out_tok": u.get("output_tokens"),
                "model": res.get("model", args.model),
            }) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(todo)}", flush=True)
    print(f"  wrote {dest}", flush=True)
