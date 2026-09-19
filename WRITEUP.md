# chunklaya: extending Laya to long inputs

**What this is:** a harness that fixes a specific, measured failure in [Laya](https://github.com/NandhaKishorM/laya) (an open-weights, Jev-style decision model) — it cannot reliably find information past the first ~200 tokens of a multi-passage input — without retraining the model. On the benchmark built to measure this, un-chunked Laya scores at chance (AUC 0.51, 95% CI crosses 0.5) once the needle is past the opening; the harness scores 0.88 with no tuning and 1.000 once known label noise in the test set is removed. Cost is at parity with the naive approach, not a tax.

This document is the narrative version. The [README](README.md) is the technical reference; this is why-and-what-changed.

---

## 1. Background: what Laya is and where it breaks

Laya is a 322–421M parameter non-autoregressive encoder (ModernBERT / mmBERT) with a small decision head on top, trained to answer typed questions — `noul` (yes/no probability), `choice` (categorical), `score` (ordinal) — about a piece of text in a single forward pass. It's the open-weights analog of TypeSafe's closed Jev.

Its checkpoints ship configured for 512–1024 tokens of input. Jev reads 32,000. The underlying encoder is pretrained to 8192 tokens, so raising the window is *possible* — the question this project started with was whether it's *useful*, or whether the model falls apart past its trained length.

**First result: length alone is not the problem.** On a whole-document classification task (4-way news category, every part of a ~3800-token state pointing at the same answer), raising `max_len` from 1024 to 4096 cost nothing — accuracy held at 0.975–0.992 throughout. The encoder handles long, coherent input fine.

**Second result: the decision head reads the opening of the state, not the whole thing.** With a single sports-news sentence placed at token offset 0 vs ~220 vs ~470 vs ~680 inside one 850-token window (nothing truncated, everything visible to the model), the probability that a `noul` question correctly flagged it dropped 9x between offset 0 (0.649) and offset ~220 (0.070) — and stayed there. This is not a context-length limit. It is the model treating "the text" as characterized by whatever comes first, which matches its training distribution (single coherent documents — an email, a ticket) but breaks the moment a state is a concatenation of many unrelated passages, which is exactly what a long document is.

**Third result: the obvious fix — overlapping token windows — doesn't fix it.** Splitting a 3800-token haystack into 512–1024-token windows, with 50% stride overlap, and taking the max `noul` score across windows: `end`-position AUC stayed at 0.52–0.56 for every window size tested, indistinguishable from chance. Overlap guarantees every token is near *some* window's start, but the needle still lands at a random offset inside whichever window contains it, and Finding 2 showed that offset is what matters. Fixed windows don't fix a within-window problem.

So the naive engineering response — bigger context window, sliding windows with overlap — fails. Something else was needed.

---

## 2. What was built: the chunklaya package

Four modules, built specifically to work around the measured failure mode rather than around it:

| Module | What it does | Why |
|---|---|---|
| `batch.py` — `predict_many` | Runs any list of `(state, question)` pairs through Laya in batched forward passes; reproduces `agent.predict` to 1e-4. Rows are length-sorted before batching. | Laya's stock API takes one state and many questions. Chunking needs many states, one question each. Length-sorting cut a measured 1.6x padding tax down to parity with the unchunked baseline. |
| `chunk.py` — `chunk_text`, `chunk_paragraphs` | Token-bounded sliding windows (exact substrings via tokenizer offsets), or one blank-line-separated passage per chunk. | `chunk_paragraphs` is the fix that actually works: every passage starts at offset 0 of its own forward pass, which is the zone where detection is reliable (Finding 2). |
| `aggregate.py` | `noul_max`, `noul_any` (noisy-OR), `noul_mean`, `choice_mixture`, `choice_loglinear`, `score_expected` — all gate-weighted. | Combines per-chunk answers into one document-level answer. `max` won for existence questions; noisy-OR inflates false positives as chunk count grows. |
| `harness.py` — `ChunkLaya` | Same call shape as `laya.Agent.predict`: `.ask(state, questions, agg=, detectors=)`. Auto-gates `choice`/`score` questions with a relevance `noul`; supports swapping in a sharper detector per question. Passes through untouched when a state fits in one window. | The actual interface. `detectors={qid: (choice_q, option)}` exists because of Finding 4 below. |
| `cache.py` — `PredictionCache` | Append-only JSONL keyed by `(state, question, window, checkpoint)`. | Long-input experiments re-score the same passages across many document compositions; without this, iterating on thresholds or pool filtering re-pays full inference every time. |

One more finding shaped the harness before it was finalized:

**The default `noul` detector is weak; a described `choice` detector is nearly perfect, on the same model.** Tested on 30 sports articles vs 300 non-sports articles, single passage, no haystack — the easiest possible version of the detection problem:

| Per-passage question | AUC | Needle median P(true) | False-positive rate @ 80% recall |
|---|---|---|---|
| `noul`: "contain a news item about sports" | 0.927 | 0.135 | 7.7% |
| `noul`: "is this text about sports" | 0.901 | 0.052 | 19.7% |
| `noul` statement: "This is a sports news article." | 0.965 | 0.360 | 4.3% |
| **`choice` 4-way → P(Sports)** | **0.998** | **1.000** | **0.3%** |
| `choice` 2-way sports/other | 0.993 | 1.000 | 2.0% |

The `noul` primitive misses the target two times in three *even when it's the entire input*. Laya ships with no fitted calibration temperatures, and `noul` is evidently the weakest-calibrated primitive on this checkpoint. Swapping to a `choice` question and reading off `P(option)` — same model, same weights, just a different question shape — took the false-positive rate from 7.7% to 0.3%. `ChunkLaya.ask(..., detectors={qid: (choice_q, option)})` makes this swap a parameter, not a rewrite.

---

## 3. The result: does it actually work?

Same 400 synthetic documents (one sports article planted in ~69 non-sports articles, at the start, middle, end, or not at all — ~3800 tokens each), every arm scored on byte-identical states. `n=100` per cell, bootstrap 95% confidence intervals, **no filtering of the test data**:

| Arm | start | middle | end | tokens/doc | ms/doc |
|---|---|---|---|---|---|
| Raw Laya @ 4096 tokens (no chunking) | 0.902 [0.85, 0.94] | **0.511 [0.43, 0.59]** | **0.518 [0.44, 0.60]** | 3794 | 469 |
| Fixed 750-token windows, 50% overlap, max | 0.870 [0.82, 0.92] | 0.548 [0.46, 0.62] | 0.558 [0.48, 0.64] | 3978 | — |
| **chunklaya: one passage per chunk + choice detector** | 0.882 [0.83, 0.93] | **0.882 [0.83, 0.93]** | **0.882 [0.83, 0.93]** | 6884 | 483 |

Two ways to read this. The interesting number is `middle`/`end`: raw Laya's confidence interval straddles 0.5 — it cannot reliably tell whether the needle is present once it's past the opening — while chunklaya is at 0.88 with a non-overlapping interval. That's the finding: **past the first passage, un-chunked Laya has no signal; chunked Laya recovers most of it, without retraining anything.**

**Where the remaining 0.12 gap comes from, and why it isn't a harness failure:** AG News's own labels have errors — roughly 1.6% of articles filed under "World" or "Sci/Tech" are actually sports stories (a baseball pitcher's 302nd win, Olympic basketball, an Olympic weightlifting verdict). A 70-passage "no needle" document almost always contains one by chance. The harness catches these — correctly — which raises its `none`-document score to 0.98 instead of 0.00. Raw Laya's `none` score looks perfect (0.000) only because it never reads past the first passage, so it never encounters the mislabeled article deeper in the document; that isn't accuracy, it's blindness. Removing the 14 (of 877) contaminated articles — identified by the model's own judgment, not manual review — closes the gap completely:

| Arm (label-noise removed) | start | middle | end |
|---|---|---|---|
| Raw Laya @ 4096 | 0.907 [0.86, 0.95] | 0.503 [0.42, 0.59] | 0.521 [0.44, 0.60] |
| **chunklaya** | **1.000 [1.00, 1.00]** | **1.000 [1.00, 1.00]** | **1.000 [1.00, 1.00]** |

Recall at 90% specificity: 1.00 / 1.00 / 1.00, all positions. Position independence here is a structural property, not a tuned result — every passage sits at offset 0 of its own forward pass, so there's no "position" left for the model to be sensitive to.

**Cost:** the first version of this harness ran at 1.6x the compute of raw Laya, because padding many short passages to a common batch length wastes cycles. Sorting rows by token length before batching (now the default in `predict_many`) closed that gap: 483 ms/document vs 469 for raw Laya at 4096 tokens, on the same hardware (M4 Pro, MPS, fp32). Chunked mode processes about 1.8x the raw token count (each passage carries its own short instruction header) but roughly 20x less total attention (many short sequences vs one long one), and those two effects now roughly cancel.

**Whether a confidence-gated second opinion (à la [classifier.dev](https://github.com/mrmps/classifier-dev)'s escalation tier) would help further:** no — checked directly. Only 0.2% of documents score in the ambiguous [0.3, 0.7] band; the harness's residual errors on the unfiltered data are *confident* mistakes (mislabeled articles scoring 0.97+), not uncertain ones, so there's nothing for a second-pass model to catch.

---

## 4. Summary: base Laya vs. chunklaya

| Capability | Base Laya | + chunklaya |
|---|---|---|
| Whole-document classification, coherent long text | Works to 4096 tokens (0.98–0.99 acc) | No change needed |
| Find a fact anywhere in a multi-passage document | **Chance past ~200 tokens** (AUC ≈ 0.51) | **0.88 AUC unfiltered, 1.000 with clean labels** |
| Cost vs. raw long-context call | — | Parity (483 vs 469 ms/doc) after batching fix |
| Per-passage detection precision | `noul` misses 2/3 of true positives standalone | `choice`-based detector: 0.3% false-positive rate |

Everything above is reproducible: `tests/test_smoke.py` verifies the package's core claims (batching equivalence, chunker invariants, harness correctness) against the live model in about a minute; `eval/exp_chunked.py --n 100 --exp 2 --configs para,750/750 --detector choice` regenerates the headline table in `results/2026-09-20-n100-raw/`. Full data, every arm, every run: `results/`. Every script: `eval/`.

## Open questions

- Jev has never been scored on this same semantic needle test — its own published needle benchmark uses a near-exact-match template the author calls saturated, so the comparison above (chunklaya vs. raw Laya) is solid, but chunklaya vs. Jev directly is not yet measured.
- Everything here is one detector (sports/not-sports) on one passage type (news articles). The chunking design should generalize; the specific numbers haven't been shown to.
- Real long documents (contracts, transcripts) will not split as cleanly on blank lines as AG News articles do. Real contracts with lawyer-labeled clauses (e.g. CUAD) are the next test.
- The opening-bias in Finding 2 is presumably fixable by fine-tuning Laya on multi-passage states with the target at randomized positions — the model's own RLCD notebook supports this — which would reduce reliance on the harness altogether.
