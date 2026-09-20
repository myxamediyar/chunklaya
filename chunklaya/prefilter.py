"""Locate, then let Laya verify.

Two different jobs: finding the passage a question is about, and deciding what that passage says.
Laya is built for the second. For the first, a lexical ranker over the chunks is fast, needs no
forward pass, and on the jev-decision-bench needle tasks puts the right passage first every time
(recall@1 1.000 at 1k, 12k and 24k). Handing Laya only that passage is what gets the harness to
Jev's numbers on that benchmark; see results/README.md.

BM25 against the question text. Stdlib only; the tokenizer is a lowercase [a-z0-9]+ split.
"""
import math
import re
from collections import Counter
from typing import List, Sequence

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOK.findall(text.lower())


def bm25_rank(texts: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75) -> List[int]:
    """Indices of `texts` ordered best-first by BM25 score against `query`. Ties keep input order."""
    docs = [_tokens(t) for t in texts]
    n = len(docs)
    if n == 0:
        return []
    avg = sum(map(len, docs)) / n
    df = Counter(t for d in docs for t in set(d))
    q = _tokens(query)
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1.0) / (tf[t] + k1 * (1.0 - b + b * len(d) / avg))
        scored.append((-s, i))
    scored.sort()
    return [i for _, i in scored]


def question_query(qdef: dict) -> str:
    """The text a question carries: instructions plus whatever the criteria say."""
    crit = qdef.get("criteria")
    if isinstance(crit, dict):
        parts = [str(v) for v in crit.values()]
    elif isinstance(crit, (list, tuple)):
        parts = [str(v) for v in crit]
    else:
        parts = []
    return " ".join([qdef.get("instructions", "")] + parts)
