# chunklaya

Long-input harness for [Laya](https://github.com/NandhaKishorM/laya), the open-weights System 1
decision model: typed questions — `noul`, `choice`, `score` — answered over one piece of text in a
single forward pass, no autoregression. It's the open answer to TypeSafe's closed Jev.

Laya's checkpoints are configured for 512–1024 tokens, and Jev reads 32k. So what happens if you just
feed Laya more text? Not what you'd guess. The encoder handles long input fine — it's pretrained to
8192 tokens. The problem is the decision head sitting on top of it, and it's a weirder failure than
"runs out of context": past roughly the first 200 tokens, it stops paying attention to anything you
give it. A fact buried in the middle of a long document is effectively invisible to it.

This repo measures that failure precisely, then fixes it: chunk the input into passages, screen each
one, combine the results. Tested on 400 documents with real confidence intervals, at no extra cost
over just cranking up the context window.

The narrative version is in [WRITEUP.md](WRITEUP.md), if you want the story instead of the reference.

## What you get over plain Laya

| | Plain Laya | + chunklaya |
|---|---|---|
| Classify or rate a long, coherent document | Fine up to 4096 tokens once you raise `max_len` (0.98–0.99 acc) | No change needed |
| Find whether something occurs anywhere in a long, multi-part input | Chance past the first ~200 tokens (AUC 0.51, CI includes 0.5) | 0.88 at every position, no filtering; 1.000 once known label noise is removed |
| How precise the per-passage detector is | The default `noul` question misses 2 of 3 true positives, even on one clean passage | Swap to a described `choice` question: 0.3% false positives |
| Cost vs. one long raw call | — | About the same (483 vs 469 ms/doc) |

What's actually in the package:

- **`predict_many`** (`batch.py`) — runs a batch of `(state, question)` pairs through Laya at once.
  Laya's own API takes one state and many questions; chunking needs the opposite. Matches
  `agent.predict` to 1e-4. Sorting rows by length before batching turned a 1.6x cost penalty into no
  penalty at all.
- **`chunk_paragraphs`** (`chunk.py`) — splits on blank lines, one passage per chunk. This is the fix
  that actually works, not `chunk_text`'s token windows: every passage lands at position 0 of its own
  forward pass, which is the only place the model reads reliably. `chunk_text` is still there for a
  document that's genuinely one long piece of text and doesn't fit even 4096 tokens.
- **`detectors=`** (`harness.py`) — swap in a sharper per-chunk question and read a probability off it
  instead of using the default `noul`. Necessary because the default one is bad at this — see below.
- **`aggregate.py`** — `max`, noisy-OR, and mean for combining `noul` scores across chunks; mixture and
  log-linear pooling for `choice`/`score`.
- **`PredictionCache`** (`cache.py`) — caches every prediction to disk, keyed by the exact input.
  Passages repeat across documents in testing, so this made iterating on thresholds and filtering
  nearly free.
- **`ChunkLaya.ask`** — the actual interface. Same shape as `laya.Agent.predict`, plus the knobs above.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```python
import laya
from chunklaya import ChunkLaya

agent = laya.load("convaiinnovations/laya", subfolder="multilingual")

# Whole-document question: just raise the window, don't chunk.
agent.cfg["max_len"] = 4096
agent.predict(long_document, {"tone": {"type": "score", "instructions": "How formal is this document?",
                                       "criteria": ["casual", "neutral", "formal"]}})

# "Does X occur anywhere in it": chunk by passage, use a sharper detector.
CATEGORY = {"type": "choice", "instructions": "Which news category does this text belong to?",
            "criteria": {"World": "international and political news", "Sports": "sports and athletics",
                         "Business": "business, markets, and finance", "Sci/Tech": "science and technology"}}
cj = ChunkLaya(agent, mode="paragraphs")
out = cj.ask(long_document,
             {"has_sports": {"type": "noul", "instructions": "Does the document contain a sports story?"}},
             detectors={"has_sports": (CATEGORY, "Sports")})
out["answers"]["has_sports"]["noul"]      # document-level P(true)
out["answers"]["has_sports"]["chunks"]    # per-passage scores and offsets
```

## Results

Everything below is `laya-multilingual` (mmBERT-base, 322M) in fp32 on an M4 Pro via MPS, using AG
News test articles as passages. Every arm in a given experiment scores the exact same states, so the
comparisons are paired. Scripts are in `eval/`; raw output for every run is in `results/` (see
[`results/README.md`](results/README.md) for what's what).

### The headline result

A sports article, planted among ~69 non-sports articles (about 3800 tokens total), at the start,
middle, end, or not at all. n=100 per cell, bootstrap 95% confidence intervals, no filtering of the
test data.

| | start | middle | end | `none` (median) | tokens/doc | in band |
|---|---|---|---|---|---|---|
| Raw Laya @ 4096 tokens | 0.902 [0.85, 0.94] | 0.511 [0.43, 0.59] | 0.518 [0.44, 0.60] | 0.000 | 3794 | 2.0% |
| Fixed 750-token windows, overlapping, max | 0.870 [0.82, 0.92] | 0.548 [0.46, 0.62] | 0.558 [0.48, 0.64] | 0.001 | 3978 | 6.8% |
| **chunklaya: one passage per chunk, choice detector** | 0.882 [0.83, 0.93] | **0.882** [0.83, 0.93] | **0.882** [0.83, 0.93] | 0.982 | 6884 | 0.2% |

Look at middle and end. Raw Laya's confidence interval straddles 0.5 — it genuinely can't tell you
whether the needle is there once it's past the opening. chunklaya sits at 0.88 with an interval
nowhere near that. That's the whole claim, and it holds without touching the test data.

**So why not 1.0?** Turns out AG News has label errors — about 1.6% of the articles filed under
"World" or "Sci/Tech" are actually sports stories (a pitcher's 302nd win, an Olympic basketball game,
a doping verdict). A 70-passage document almost always has one somewhere. The harness finds them,
correctly, which pushes `none`-document scores up to 0.98. Raw Laya's `none` score looks clean at
0.000 only because it never reads far enough into the document to hit the mislabeled article — that's
blindness, not accuracy. Pull the 14 (of 877) contaminated articles out of the pool, using the model's
own judgment rather than manual review, and:

| | start | middle | end |
|---|---|---|---|
| Raw Laya @ 4096 | 0.907 [0.86, 0.95] | 0.503 [0.42, 0.59] | 0.521 [0.44, 0.60] |
| **chunklaya** | **1.000** [1.00, 1.00] | **1.000** [1.00, 1.00] | **1.000** [1.00, 1.00] |

Recall at 90% specificity is 1.00 across the board. Position stops mattering because every passage
gets its own forward pass starting at position 0 — there's no "position" left for the model to be
sensitive to.

**Cost.** The first version of this ran 1.6x slower than raw Laya, purely from padding — a batch of
many short passages pads everything to the longest one. Sorting rows by length before batching fixed
that: 483 ms/doc chunked vs 469 raw, same hardware. Chunked mode pushes about 1.8x the tokens through
(each passage carries its own short instruction header) but way less total attention, and the two
roughly cancel out.

**"In band"** is the fraction of documents scoring in an ambiguous middle range, 0.3 to 0.7 — the ones
a confidence-gated second pass would need to re-check. It's 0.2% for chunklaya. The mistakes that
remain are confident mistakes (mislabeled articles scoring 0.97+), not uncertain ones, so a second
opinion wouldn't catch them anyway.

This whole way of evaluating the harness — cache every prediction, batch aggressively, and put cost
right next to accuracy instead of reporting accuracy alone — is borrowed from
[classifier.dev](https://github.com/mrmps/classifier-dev), a free HTTP classifier built on Jev. Its
"smart tier" escalates to a bigger model whenever Jev is under 0.7 confident, which is where the "in
band" idea above comes from too.

### How we got there

**1. Raising `max_len` alone works fine for whole-document questions.** 4-way classification,
same-class filler padded to ~3800 tokens, n=120: 0.983 accuracy at 762 tokens, 0.975–0.992 at 3801
tokens with `max_len=4096`. The encoder and head are both fine with long, coherent input. (An
experimental "stack" mode, where Laya reads back its own per-chunk answers, doesn't work zero-shot —
0.625 accuracy. It's left in the code but not recommended.)

**2. The decision head reads the opening of a state and basically nothing else.** Put a single needle
sentence at a chosen offset inside one 850-token window — nothing truncated, everything visible — and
ask a yes/no question about it:

| offset (tokens) | 0 | ~220 | ~470 | ~680 |
|---|---|---|---|---|
| AUC | 0.878 | 0.629 | 0.593 | 0.570 |
| median P(true) | 0.649 | 0.070 | 0.078 | 0.071 |

The probability drops 9x between offset 0 and offset ~220, and doesn't recover. This isn't a length
limit — Finding 1 already showed the model is fine with long input. It's that the model treats "the
text" as whatever comes first, which tracks with how it was trained: on single coherent documents, not
concatenated fragments.

**3. Sliding windows with overlap don't fix this.** Same needle test, chunked into overlapping token
windows instead of one long pass, taking the max score across windows:

| | start | middle | end | `none` (median) |
|---|---|---|---|---|
| Raw @4096 | 0.880 | 0.576 | 0.552 | 0.115 |
| 750-token windows, 375 stride | 0.788 | 0.642 | 0.521 | 0.273 |
| 750-token windows, no overlap | 0.811 | 0.547 | 0.521 | 0.184 |
| 512-token windows, 256 stride | 0.629 | 0.592 | 0.517 | 0.276 |
| 1024-token windows, 512 stride | 0.739 | 0.561 | 0.552 | 0.290 |

`end` stays at chance no matter how you tune it. The needle still lands at some random offset inside
whichever window it falls in, and Finding 2 already told us offset is what matters. More windows also
raises the false-positive floor faster than it raises the signal. This negative result is what pointed
toward paragraph chunking instead.

**4. The default `noul` question is a weak detector. A described `choice` question is nearly
perfect.** Tested on 30 sports articles and 300 non-sports articles, one at a time, no haystack — as
easy as this gets:

| question | AUC | needle median | false positives @ 80% recall |
|---|---|---|---|
| `noul`: "contain a news item about sports" | 0.927 | 0.135 | 7.7% |
| `noul` statement: "This is a sports news article." | 0.965 | 0.360 | 4.3% |
| **`choice` (4-way) → P(Sports)** | **0.998** | **1.000** | **0.3%** |
| `choice` (2-way, sports vs. other) | 0.993 | 1.000 | 2.0% |

The `noul` question gets it wrong two times out of three, on an article that's entirely about sports
and nothing else. Laya ships without fitted calibration for this primitive, and it shows. Switching to
a `choice` question — same model, same weights — dropped the false-positive rate by 25x. (AG News is
in Laya's training data, which helps the 4-way version specifically; the 2-way one, which isn't a
training label, still holds up at 0.993.)

**5.** Put 2 and 4 together — one passage per chunk, a `choice`-based detector — and you get the
headline table above.

### Does Jev handle the same needle test any better?

Honestly, we don't know — not on this exact question. There's public per-item data for Jev on a
needle-in-haystack test, and it's a flat 1.000 at every position out to ~22k tokens. But look at the
needle: it's a fixed template ("The access code for the {color} {object} locker is {4 digits}"), and
the question restates the exact fields being asked about — that's closer to an exact-text-match lookup
than the kind of semantic judgment our test uses, and the benchmark's own author calls it saturated.
TypeSafe's documentation does say plainly that Jev's accuracy falls as the input fills up with
irrelevant content, and recommends filtering with a relevance question first — the same idea this
harness is built around — but there's no public number for Jev on a semantic needle test like ours.
That comparison is still open.

## How to use it, in practice

```
Whole-document question (classify it, rate it)
    → just raise max_len to 4096. Don't chunk.

"Does X occur anywhere in this?" over a long, multi-part input
    → ChunkLaya(mode="paragraphs")
    → detectors={qid: (choice_question, option)}   — use a choice question, not the default noul
    → aggregate with "max" — noisy-OR gets worse as chunk count grows
    → the false-positive rate of your detector, times the number of passages, is roughly your
      false-alarm rate per document — that's the number to optimize, not the needle detection itself

Whole-document question over something that doesn't fit even 4096 tokens
    → ChunkLaya(mode="tokens", chunk_tokens=750, stride=375), aggregate with "mixture" or "loglinear"
      (works where chunks agree; untested where they disagree with each other)
```

## Layout

```
chunklaya/
  batch.py       predict_many — batched (state, question) pairs, matches agent.predict to 1e-4
  cache.py       PredictionCache — on-disk cache keyed by (state, question, window, checkpoint)
  chunk.py       chunk_text (overlapping token windows) and chunk_paragraphs (one passage per chunk)
  aggregate.py   noul_max / noul_any / noul_mean, choice_mixture / choice_loglinear, score_expected
  harness.py     ChunkLaya.ask(state, questions, agg=, detectors=) — same shape as laya.Agent.predict
eval/
  needle_data.py     deterministic test-state builders
  fetch_data.py      how eval/data/ag_news.json was built
  exp_maxlen.py      Finding 1, plus the first raw-window needle test
  exp_chunked.py     the main experiments: chunked classification, needle detection, offset sweep
  exp_detector.py    Finding 4 — comparing detector questions
  analyze_maxlen.py, jev_needle_by_depth.py, metrics.py
results/         one folder per run — see results/README.md
tests/test_smoke.py   equivalence and correctness checks against the live model, ~1 min
third_party/     the needle-task builders behind the Jev comparison — someone else's code, their
                 license, kept as-is (see results/README.md and the license note below)
```

## Running it

```bash
.venv/bin/python tests/test_smoke.py
.venv/bin/python eval/exp_chunked.py --n 30                                                    # ~10 min
.venv/bin/python eval/exp_chunked.py --n 100 --exp 2 --configs para,750/750 --detector choice   # the headline table, ~5 min
.venv/bin/python eval/exp_chunked.py --n 100 --exp 2 --configs para --detector choice --clean-haystack 0.2
.venv/bin/python eval/exp_detector.py                                                          # ~1 min
```

Timings are for an M4 Pro on MPS in fp32. If an MPS op isn't implemented, set
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

## What's still open

- We don't have Jev's number on our actual needle test, only on an easier one.
- Real long documents (contracts, transcripts) won't split as cleanly on blank lines as news articles
  do. That's the next thing worth testing.
- The mixture/log-linear aggregation was only tested where every chunk agrees. Disagreeing chunks are
  untested.
- Everything here uses one detector (sports/not-sports) on one kind of passage (news articles). The
  approach should generalize, but that hasn't been shown yet.
- The opening-bias in Finding 2 is probably fixable by fine-tuning on multi-passage input with the
  target at random positions — Laya's own training notebook supports this.
- `noul` isn't calibrated on this checkpoint. Fit a temperature before trusting the raw probability as
  a threshold.

## License

Apache 2.0 (see [LICENSE](LICENSE)) — same as Laya. `third_party/` is someone else's code under its
own MIT license (see `third_party/LICENSE-jev-decision-bench`); Apache 2.0 doesn't extend to those
files. `eval/data/ag_news.json` is a 1200-row sample of the AG News test split from the Hugging Face
mirror `fancyzhx/ag_news`. `results/jev-decision-bench-needle/` is someone else's published output, kept
here under its original MIT license so the comparisons above can be checked.
