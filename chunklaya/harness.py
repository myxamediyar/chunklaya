"""ChunkLaya: chunk → gate → aggregate, with the same call shape as laya.Agent.predict."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from laya.common import confidence_from_probs

from .aggregate import choice_loglinear, choice_mixture, noul_any, noul_max, noul_mean, score_expected
from .batch import predict_many
from .chunk import Chunk, chunk_paragraphs, chunk_text
from .prefilter import BM25Index, question_query

GATE_TEMPLATE = ("Does this text contain the information needed to answer the following question? "
                 "Question: {instructions}")
DEFAULT_AGG = {"noul": "max", "choice": "mixture", "score": "mixture"}


class ChunkIndex:
    """One state, chunked once, with its BM25 index built on first use.

    ChunkLaya.index(text) makes one; ChunkLaya.ask accepts it in place of the text. Chunking and
    indexing are the only per-question costs that grow with the input once a prefilter is on (at
    ~1M tokens, 0.9 of the 0.92 s a question takes -- results/2026-09-22-needle-scale), so a state
    that will take more than one question should be indexed once."""

    def __init__(self, text: str, chunks: List[Chunk], params: tuple):
        self.text, self.chunks, self.params = text, chunks, params
        self._bm25: Optional[BM25Index] = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index([c.text for c in self.chunks])
        return self._bm25

    def __len__(self) -> int:
        return len(self.chunks)


class ChunkLaya:
    """
    agent          a loaded laya.Agent
    chunk_tokens   window size; keep it inside the head's reliable zone (README: decay past ~1000)
    stride         window step; <= chunk_tokens/2 puts every token near some window's start
    gate           "auto"  → relevance noul per chunk for choice/score; none for noul (it is its own gate)
                   True/False → force on/off;  str → custom template with {instructions}
    prefilter      None → Laya scores every chunk;  "bm25" → rank chunks against the question text and
                   score only the top `top_k`. Recommended for "does X occur anywhere" over long input:
                   locating is a retrieval job, verifying is Laya's, and splitting them is what matches
                   Jev on the needle benchmark (README). Keep top_k small -- with `max` aggregation each
                   extra chunk is another chance for a false positive.
    """

    def __init__(self, agent, chunk_tokens: int = 750, stride: Optional[int] = None,
                 batch_size: int = 16, gate: Union[str, bool] = "auto", mode: str = "tokens", cache=None,
                 prefilter: Optional[str] = None, top_k: int = 1):
        """mode: "tokens" → fixed windows of chunk_tokens every stride;  "paragraphs" → one blank-line-
        separated unit per chunk (chunk_tokens is then only the fallback cap for an oversized paragraph)."""
        if mode not in ("tokens", "paragraphs"):
            raise ValueError(mode)
        self.agent, self.chunk_tokens, self.stride = agent, chunk_tokens, stride or chunk_tokens // 2
        self.batch_size, self.gate, self.mode, self.cache = batch_size, gate, mode, cache
        if prefilter not in (None, "bm25"):
            raise ValueError(prefilter)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.prefilter, self.top_k = prefilter, top_k

    def chunks(self, text: str) -> List[Chunk]:
        if self.mode == "paragraphs":
            return chunk_paragraphs(self.agent.tok, text, self.chunk_tokens)
        return chunk_text(self.agent.tok, text, self.chunk_tokens, self.stride)

    @property
    def _params(self) -> tuple:
        return (self.mode, self.chunk_tokens, self.stride)

    def index(self, text: str) -> ChunkIndex:
        """Chunk `text` once (and, on demand, index it) so it can take many questions via `ask`."""
        if not isinstance(text, str):
            raise TypeError("ChunkLaya.index takes a string state; serialize dict/list states first")
        return ChunkIndex(text, self.chunks(text), self._params)

    def _select(self, idx: ChunkIndex, qdef: Dict) -> List[Chunk]:
        """The chunks Laya will score for this question, in document order."""
        chunks = idx.chunks
        if self.prefilter is None or len(chunks) <= self.top_k:
            return chunks
        keep = sorted(idx.bm25.rank(question_query(qdef))[: self.top_k])
        return [chunks[i] for i in keep]

    def _use_gate(self, qdef: Dict) -> bool:
        if self.gate == "auto":
            return qdef["type"] != "noul"
        return bool(self.gate)

    def _gate_q(self, qdef: Dict) -> Dict:
        tpl = self.gate if isinstance(self.gate, str) and self.gate != "auto" else GATE_TEMPLATE
        return {"type": "noul", "instructions": tpl.format(instructions=qdef["instructions"])}

    def ask(self, state: Union[str, ChunkIndex], questions: Dict[str, Dict], agg: Optional[Dict[str, str]] = None,
            max_len: Optional[int] = None, detectors: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
        """
        state      the text, or a ChunkIndex from `index` when the same text takes several calls
        agg        per-question override of the aggregation rule:
                   noul: max | any | mean      choice: mixture | loglinear | stack      score: mixture | stack
        max_len    window passed to Laya for each chunk (default agent.cfg["max_len"])
        detectors  {qid: (question, option_key)} — for a noul qid, ask `question` per chunk instead and use
                   P(option_key) as that chunk's P(true). On laya-multilingual a described choice is a far
                   sharper per-passage detector than a noul (results/2026-09-20-para/detectors.json:
                   AUC 0.998 vs 0.93, 0.3% vs 7.7% false positives at 80% recall).
        Returns laya-shaped {"answers": {qid: {...}}} plus n_chunks and per-chunk detail under
        answers[qid]["chunks"] = [{"index", "p", "gate", "tok_start", "tok_end"}].
        """
        if isinstance(state, ChunkIndex):
            if state.params != self._params:
                raise ValueError(f"index was built with {state.params}, this harness chunks with {self._params}")
            idx = state
        else:
            idx = self.index(state)
        chunks = idx.chunks
        agg = {**DEFAULT_AGG, **{k: v for k, v in (agg or {}).items()}}
        m = len(chunks)

        # one batched pass: (answer row, optional gate row) per selected chunk per question.
        # The prefilter selects per question, so each question has its own chunk list.
        pairs, index, sel = [], [], {}
        detectors = detectors or {}
        for qid, qdef in questions.items():
            C = sel[qid] = self._select(idx, qdef)
            mq = len(C)
            g = self._use_gate(qdef) and mq > 1
            det = detectors.get(qid)
            if det is not None and qdef["type"] != "noul":
                raise ValueError("detectors apply to noul questions only")
            # Passthrough is for a state that fit in one window. A prefilter keeping one chunk of many is
            # not that: the chunk is a passage, so the per-passage detector still applies.
            row_q = det[0] if (det is not None and m > 1) else qdef
            for j, c in enumerate(C):
                pairs.append((c.text, row_q)); index.append((qid, j, "ans"))
                if g:
                    pairs.append((c.text, self._gate_q(qdef))); index.append((qid, j, "gate"))
        res = predict_many(self.agent, pairs, self.batch_size, max_len=max_len, cache=self.cache)
        per: Dict[str, Dict[str, list]] = {qid: {"ans": [None] * len(sel[qid]), "gate": [None] * len(sel[qid])}
                                           for qid in questions}
        for (qid, j, kind), r in zip(index, res):
            per[qid][kind][j] = r

        answers, total_tokens = {}, sum(r["seq_len"] for r in res)
        for qid, qdef in questions.items():
            C, mq = sel[qid], len(sel[qid])
            A, G, t = per[qid]["ans"], per[qid]["gate"], qdef["type"]
            w = [g["noul"] for g in G] if G[0] is not None else None
            mode = agg.get(qid, agg[t])
            detail = [{"index": c.index, "tok_start": c.tok_start, "tok_end": c.tok_end,
                       "p": [float(x) for x in a["p"]], "gate": (None if w is None else float(w[i]))}
                      for i, (c, a) in enumerate(zip(C, A))]
            if m == 1:                                   # passthrough: the state fit in one window
                ans = {k: v for k, v in A[0].items() if k not in ("p", "keys", "seq_len")}
            elif t == "noul":
                det = detectors.get(qid)
                ps = [a["probabilities"][det[1]] for a in A] if det is not None else [a["noul"] for a in A]
                if det is not None:
                    for dd, pp in zip(detail, ps):
                        dd["p"] = [1.0 - pp, pp]
                v = {"max": noul_max, "any": noul_any}.get(mode, lambda x: noul_mean(x, w))(ps)
                ans = {"type": "noul", "noul": v, "confidence": max(v, 1.0 - v)}
            elif mode == "stack":
                ans = self._stack(qdef, A, w, max_len)
            else:
                D = [a["p"] for a in A]
                p = choice_loglinear(D, w) if mode == "loglinear" else choice_mixture(D, w)
                keys = A[0]["keys"]
                ans = {"type": t, "confidence": float(confidence_from_probs(p, len(p))),
                       "probabilities": {k: float(v) for k, v in zip(keys, p)}}
                if t == "choice":
                    ans["choice"] = keys[int(p.argmax())]
                else:
                    ans["score"], ans["legend"] = score_expected(D, w)[0], A[0]["legend"]
            ans.update(agg=mode if m > 1 else "passthrough", n_scored=mq, chunks=detail)
            answers[qid] = ans

        return {"model": "chunklaya", "answers": answers, "n_chunks": m, "mode": self.mode,
                "chunk_tokens": self.chunk_tokens, "stride": self.stride, "prefilter": self.prefilter,
                "usage": {"input_tokens": int(total_tokens), "output_tokens": 0}}

    def _stack(self, qdef: Dict, A: list, w: Optional[list], max_len: Optional[int]) -> Dict[str, Any]:
        """Second-order decision: Laya reads its own per-chunk outputs as a compact state. Experimental."""
        ev = []
        for i, a in enumerate(A):
            row = {"part": i + 1}
            if w is not None:
                row["relevance"] = round(float(w[i]), 2)
            if qdef["type"] == "choice":
                row["answer"], row["p"] = a["choice"], round(float(a["p"].max()), 2)
            else:
                row["score"] = round(float(a["score"]), 2)
            ev.append(row)
        order = np.argsort([-r.get("relevance", 1.0) for r in ev])
        state = {"note": "Each part is one section of a longer document, judged independently.",
                 "parts": [ev[i] for i in order]}
        r = predict_many(self.agent, [(state, qdef)], 1, max_len=max_len, cache=self.cache)[0]
        return {k: v for k, v in r.items() if k not in ("p", "keys", "seq_len")}
