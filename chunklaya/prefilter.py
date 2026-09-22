"""Locate, then let Laya verify.

Two different jobs: finding the passage a question is about, and deciding what that passage says.
Laya is built for the second. For the first, a lexical ranker over the chunks is fast, needs no
forward pass, and on the jev-decision-bench needle tasks puts the right passage first every time
(recall@1 1.000 at 1k, 12k and 24k -- and at 100k, 300k and 1M tokens, results/2026-09-22-needle-scale).
Handing Laya only that passage is what gets the harness to Jev's numbers on that benchmark; see
results/README.md.

BM25 against the question text. Stdlib only; the tokenizer is a lowercase [a-z0-9]+ split. BM25Index
builds the postings once so that one long input can take many questions; bm25_rank is the one-shot
form. Lexical means lexical: a question with nothing to search for (a category, "any sports story")
ranks no better than chance here (results/2026-09-22-ranker), and should score every chunk instead.
"""
import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOK.findall(text.lower())


class BM25Index:
    """BM25 over a fixed list of texts: tokenised and counted once, queried many times.

    A query walks only the postings of its own terms, so its cost is independent of how many chunks
    say nothing about it. Ties keep input order, and a repeated query term counts each time, exactly
    as bm25_rank did before the index existed."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        docs = [_tokens(t) for t in texts]
        self.n, self.k1, self.b = len(docs), k1, b
        self.avg = sum(map(len, docs)) / self.n if self.n else 0.0
        self.len = [len(d) for d in docs]
        self.df: Counter = Counter(t for d in docs for t in set(d))
        self.post: Dict[str, List[tuple]] = {}          # term -> [(doc id, term frequency)]
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                self.post.setdefault(t, []).append((i, tf))

    def scores(self, query: str) -> List[float]:
        s = [0.0] * self.n
        for t in _tokens(query):
            if t not in self.post:
                continue
            idf = math.log(1.0 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in self.post[t]:
                s[i] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * self.len[i] / self.avg))
        return s

    def rank(self, query: str) -> List[int]:
        """Indices of the texts ordered best-first against `query`."""
        return [i for _, i in sorted((-v, i) for i, v in enumerate(self.scores(query)))]


def bm25_rank(texts: Sequence[str], query: str, k1: float = 1.5, b: float = 0.75) -> List[int]:
    """Indices of `texts` ordered best-first by BM25 score against `query`. Ties keep input order."""
    return BM25Index(texts, k1, b).rank(query)


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
