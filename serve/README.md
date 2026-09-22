# Serving chunklaya

chunklaya as an HTTP endpoint: one document per request, up to 32 questions,
one index. It speaks the System One request and response shape that
[classifier.dev](https://classifier.dev) already sends to Beam and TypeSafe,
so that service can reach it through a `Backend` descriptor — a URL and a
bearer — rather than a new client. It exists for the documents Jev cannot
read: Jev stops at 32k tokens; the needle results in the top-level README hold
to a million.

```
serve/service.py        the endpoint
serve/test_service.py   contract tests against a fake harness; no torch, no weights
serve/Dockerfile        the image; built by .github/workflows/image.yml
```

## What it does

`POST /v1/systemone` with a bearer token:

```json
{"state": [{"id": "doc", "text": "<up to 4,000,000 characters>"}],
 "questions": {
   "dept": {"type": "choice", "instructions": "Which department should handle this?",
            "criteria": {"billing": null, "technical": null}},
   "refund": {"type": "noul", "instructions": "Does the document request a refund?"}},
 "strategy": "scan"}
```

```json
{"model": "chunklaya/multilingual", "checkpoint": "1c5edc17…", "strategy": "scan", "n_chunks": 41,
 "answers": {"dept":   {"type": "choice", "choice": "billing", "confidence": 0.91,
                        "probabilities": {"billing": 0.93, "technical": 0.07}, "agg": "mixture", "n_scored": 41},
             "refund": {"type": "noul", "noul": 0.97, "confidence": 0.97, "agg": "max", "n_scored": 41}},
 "usage": {"input_tokens": 9184, "output_tokens": 0, "document_tokens": 4211},
 "timing": {"index_ms": 38.2, "ask_ms": 412.0, "cached": false}}
```

The document is chunked by paragraph and indexed once; every question in the
request is answered off that index (`ChunkLaya.index` then `ask`). A second
request with the same text hits an in-memory cache and skips the indexing
(`timing.cached`).

Two strategies, chosen per request:

| `strategy` | What Laya scores | Right for | Cost |
|---|---|---|---|
| `scan` (default) | every passage, then aggregates | "which category is this", "does X occur anywhere" | one row per passage, plus a gate row per passage for choice/score questions |
| `locate` | the `top_k` passages BM25 ranks highest against the question (default 1) | a named fact in a long document | one forward pass at any document size |

Scan is bounded: a document with more than `MAX_SCAN_CHUNKS` passages (256)
is refused with `too_many_passages` and told to use locate, and a request
that would score more than `MAX_ROWS` rows (4,096) across its questions is
refused with `too_many_rows`. Both are 422s: the caller sees them as an
invalid request, not an outage.

Optional fields, passed through to `ChunkLaya.ask`: `top_k` (1–8, locate
only), `agg` (`{qid: mode}` — noul `max|any|mean`, choice
`mixture|loglinear|stack`, score `mixture|stack`), `detectors`
(`{noul_qid: [choice question, option]}`, the sharper per-passage detector),
and `detail: true` to return per-passage scores and token offsets under
`answers[qid].chunks`.

One difference from calling the harness directly: for `locate`, the BM25
query is the instructions plus any label *meanings*, never the label names.
`question_query` in `prefilter.py` joins `str(v)` over the criteria values,
which for `{label: null}` criteria contributes the word "None"; the service
skips those. Names are left out on purpose: the first live run put them in,
and a passage that merely contained the word "technical" was handed to Laya
as the answer, with certainty. Laya itself sees the question unchanged.

### What the first live run showed

An RTX A5000 pod, 9 vCPU, measured from a laptop through the RunPod proxy
(`timing` is the service's own clock; "client" includes the network):

| Request | index | ask | client |
|---|---|---|---|
| 3-paragraph document, scan, cold | 0.7 ms | 179 ms | 910 ms |
| same, cached | — | 25 ms | 303 ms |
| 1.09M characters (6,500 passages), locate, cold, 1.1 MB upload | 535 ms | 132 ms | 2.1 s |
| same document, cached, one question | — | 29 ms | 771 ms |

Two things about asking, both reproduced from the top-level README's results:

- **A lookup wants `detectors`.** On the exact passage holding a planted
  fact ("the access code for the teal harbor locker is 4417"), the default
  `noul` verifier answered 0.10; a described `choice` detector
  (`{"yes": "the passage gives 4417 as the locker's access code", "no": "it
  does not"}`, read as P(yes)) answered 0.98. BM25 had found the passage both
  times. Send `detectors: {"<noul qid>": [<that choice question>, "yes"]}`.
- **`locate` is not for categorization.** Asked "what is this document
  mostly about?" over one BM25-chosen passage, Laya returned exactly 0.5/0.5
  — its answer for a passage about neither label. With `top_k: 3` the
  mixture reached 0.93. Categorical questions belong to `scan`, or to
  `locate` with a `top_k` large enough to see representative passages.
  Label meanings only help retrieval when their words occur in the text
  verbatim: the harness's BM25 tokenises on `[a-z0-9]+` with no stemming
  and no stopword list, so "refunds" does not match "refund", and a query
  whose only hit is "and" ranks the one paragraph that contains "and".

Refusals carry `{error: {code, message}}`, and a `detail` string on 422.
`429` and `503` set `Retry-After`. `504` means the request passed
`REQUEST_BUDGET_S` (80 s, under the RunPod proxy's 100-second cut-off); the
work finishes on the box and only the answer is lost.

## What it keeps

Indexes are held in memory, keyed by a hash of the text, bounded by count
(`INDEX_CACHE_ITEMS`, 64) and total passages (`INDEX_CACHE_CHUNKS`, 200,000),
and expire (`INDEX_CACHE_TTL_S`, 600 s). Nothing is written to disk; the
harness's on-disk `PredictionCache` is off. Log lines carry status, strategy,
counts and durations — never request text or ids.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CHUNKLAYA_TOKEN` | required | the bearer callers must present; `CHUNKLAYA_ALLOW_ANON=1` waives it for a local run |
| `LAYA_SUBFOLDER` | `multilingual` | checkpoint: `multilingual` (1,024-token window, what the results used) or empty for English (512) |
| `LAYA_DEVICE` | `cuda` | `cuda` refuses to start if CUDA is missing; `mps` or `cpu` for a laptop |
| `CHUNK_TOKENS` | 750 | passage cap; clamped to the checkpoint's window if too large |
| `BATCH_SIZE` | 32 | rows per forward pass |
| `MAX_CHARS` | 4,000,000 | document ceiling (about a million tokens) |
| `MAX_QUESTIONS`, `MAX_LABELS` | 32, 32 | per request, per choice question |
| `MAX_SCAN_CHUNKS`, `MAX_ROWS`, `MAX_TOP_K` | 256, 4,096, 8 | the bounds above |
| `MAX_INFLIGHT`, `INDEX_WORKERS`, `QUEUE_WAIT_S` | 8, 2, 5 | admission: requests in flight, concurrent index builds, how long to wait for the GPU before 429 |
| `REQUEST_BUDGET_S` | 80 | wall-clock ceiling per request |

## Run it on a laptop first

The costly unknowns are CPU-bound and need no GPU: how fast paragraph chunking
and BM25 run on server cores, and what an index costs in memory at a million
tokens.

```sh
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e . "laya==0.3.4" "transformers==4.56.2" "fastapi==0.116.1" "uvicorn[standard]" huggingface_hub
CHUNKLAYA_ALLOW_ANON=1 LAYA_DEVICE=mps uvicorn --app-dir serve service:api --port 8000   # first start downloads the weights
```

```sh
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": [{"text": "Please refund the duplicate invoice payment.\n\nThe technical team reset the router."}],
  "questions": {"dept": {"type": "choice", "instructions": "Which department should handle this?",
                         "criteria": {"billing": null, "technical": null}}}}'
```

The contract tests need `fastapi` and `httpx` and skip themselves otherwise:

```sh
CHUNKLAYA_NO_APP=1 python -m unittest serve/test_service.py
```

## The image

`.github/workflows/image.yml` builds `serve/Dockerfile` from the repository
root on every push touching the package or `serve/`, and by hand from the
Actions tab. It pushes `ghcr.io/myxamediyar/chunklaya:<commit>`, and
`:latest` from `main`. The weights are downloaded at build time at the pinned
revision and the container runs offline.

The first push creates a **private** package. A host that pulls it needs one
of:

- the package made public — package page → Package settings → Change
  visibility. Nothing in the image is secret: Apache-2.0 code and public
  weights; the bearer is a runtime environment variable, never baked in; or
- a registry credential on the host (for RunPod: Settings → Container
  Registry Auth, with a GitHub token that has `read:packages`, passed as
  `containerRegistryAuthId` when creating the pod).

With Docker locally: `docker build --platform linux/amd64 -f serve/Dockerfile -t chunklaya:dev .`

## Deploy on RunPod

A Pod, not Serverless: the index cache needs a process that stays up, and the
GPU is nearly idle — the work is tokenizing and ranking on CPU. One
`NVIDIA RTX A5000` (24 GB) in Secure Cloud is $0.27/hour, about **$194 a
month** running continuously. `US-WA-1` and `US-CA-2` are the US-west
datacenters.

```sh
curl -s -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' -d '{
  "name": "chunklaya",
  "imageName": "ghcr.io/myxamediyar/chunklaya:<commit>",
  "cloudType": "SECURE",
  "gpuTypeIds": ["NVIDIA RTX A5000"], "gpuCount": 1,
  "dataCenterIds": ["US-WA-1", "US-CA-2"], "dataCenterPriority": "custom",
  "allowedCudaVersions": ["12.6", "12.7", "12.8", "12.9", "13.0"],
  "containerDiskInGb": 30, "volumeInGb": 0,
  "ports": ["8000/http"],
  "env": {"CHUNKLAYA_TOKEN": "<a long random secret>"}
}'
```

The response carries the pod `id`; the service answers at
`https://<id>-8000.proxy.runpod.net` once `/health` is 200 (the model loads
and warms first; allow a few minutes on the first start).

```sh
curl -s https://<id>-8000.proxy.runpod.net/health
curl -s https://<id>-8000.proxy.runpod.net/v1/systemone -H "authorization: Bearer $CHUNKLAYA_TOKEN" \
  -H 'content-type: application/json' -d @request.json
```

What to know about the pod:

- The proxy URL is public; the bearer is the only thing between it and the
  internet. Rotate it by editing the pod's environment and restarting.
- The proxy cuts a connection at 100 seconds (524). `REQUEST_BUDGET_S` stays
  under it.
- **Stopping a pod keeps billing its volume; terminate when done:**
  `curl -X DELETE https://rest.runpod.io/v1/pods/<id> -H "Authorization: Bearer $RUNPOD_API_KEY"`.
- A pod does not scale. One box is one box; the second one is a decision, not
  an autoscaler.
- `OCI runtime create failed` at start means the host's driver is older than
  CUDA 12.6; `allowedCudaVersions` above should prevent it.

## Not yet measured

The numbers in the top-level README are fp32 on Apple MPS. On CUDA,
`predict_many` autocasts to fp16. The warmup at startup catches a crash, not a
numerical drift — run one paired comparison on real documents before trusting
the scores from a GPU.
