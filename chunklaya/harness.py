"""ChunkLaya: chunk → gate → aggregate, with the same call shape as laya.Agent.predict."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from laya.common import confidence_from_probs

from .aggregate import choice_loglinear, choice_mixture, noul_any, noul_max, noul_mean, score_expected
from .batch import predict_many
from .chunk import Chunk, chunk_paragraphs, chunk_text

GATE_TEMPLATE = ("Does this text contain the information needed to answer the following question? "
                 "Question: {instructions}")
DEFAULT_AGG = {"noul": "max", "choice": "mixture", "score": "mixture"}


class ChunkLaya:
    """
    agent          a loaded laya.Agent
    chunk_tokens   window size; keep it inside the head's reliable zone (README: decay past ~1000)
    stride         window step; <= chunk_tokens/2 puts every token near some window's start
    gate           "auto"  → relevance noul per chunk for choice/score; none for noul (it is its own gate)
                   True/False → force on/off;  str → custom template with {instructions}
    """

    def __init__(self, agent, chunk_tokens: int = 750, stride: Optional[int] = None,
                 batch_size: int = 16, gate: Union[str, bool] = "auto", mode: str = "tokens", cache=None):
        """mode: "tokens" → fixed windows of chunk_tokens every stride;  "paragraphs" → one blank-line-
        separated unit per chunk (chunk_tokens is then only the fallback cap for an oversized paragraph)."""
        if mode not in ("tokens", "paragraphs"):
            raise ValueError(mode)
        self.agent, self.chunk_tokens, self.stride = agent, chunk_tokens, stride or chunk_tokens // 2
        self.batch_size, self.gate, self.mode, self.cache = batch_size, gate, mode, cache

    def chunks(self, text: str) -> List[Chunk]:
        if self.mode == "paragraphs":
            return chunk_paragraphs(self.agent.tok, text, self.chunk_tokens)
        return chunk_text(self.agent.tok, text, self.chunk_tokens, self.stride)

    def _use_gate(self, qdef: Dict) -> bool:
        if self.gate == "auto":
            return qdef["type"] != "noul"
        return bool(self.gate)

    def _gate_q(self, qdef: Dict) -> Dict:
        tpl = self.gate if isinstance(self.gate, str) and self.gate != "auto" else GATE_TEMPLATE
        return {"type": "noul", "instructions": tpl.format(instructions=qdef["instructions"])}

    def ask(self, state: str, questions: Dict[str, Dict], agg: Optional[Dict[str, str]] = None,
            max_len: Optional[int] = None, detectors: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
        """
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
        if not isinstance(state, str):
            raise TypeError("ChunkLaya.ask takes a string state; serialize dict/list states first")
        chunks = self.chunks(state)
        agg = {**DEFAULT_AGG, **{k: v for k, v in (agg or {}).items()}}
        m = len(chunks)

        # one batched pass: (answer row, optional gate row) per chunk per question
        pairs, index = [], []
        detectors = detectors or {}
        for qid, qdef in questions.items():
            g = self._use_gate(qdef) and m > 1
            det = detectors.get(qid)
            if det is not None and qdef["type"] != "noul":
                raise ValueError("detectors apply to noul questions only")
            row_q = det[0] if (det is not None and m > 1) else qdef
            for c in chunks:
                pairs.append((c.text, row_q)); index.append((qid, c.index, "ans"))
                if g:
                    pairs.append((c.text, self._gate_q(qdef))); index.append((qid, c.index, "gate"))
        res = predict_many(self.agent, pairs, self.batch_size, max_len=max_len, cache=self.cache)
        per: Dict[str, Dict[str, list]] = {qid: {"ans": [None] * m, "gate": [None] * m} for qid in questions}
        for (qid, ci, kind), r in zip(index, res):
            per[qid][kind][ci] = r

        answers, total_tokens = {}, sum(r["seq_len"] for r in res)
        for qid, qdef in questions.items():
            A, G, t = per[qid]["ans"], per[qid]["gate"], qdef["type"]
            w = [g["noul"] for g in G] if G[0] is not None else None
            mode = agg.get(qid, agg[t])
            detail = [{"index": c.index, "tok_start": c.tok_start, "tok_end": c.tok_end,
                       "p": [float(x) for x in a["p"]], "gate": (None if w is None else float(w[i]))}
                      for i, (c, a) in enumerate(zip(chunks, A))]
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
            ans.update(agg=mode if m > 1 else "passthrough", chunks=detail)
            answers[qid] = ans

        return {"model": "chunklaya", "answers": answers, "n_chunks": m, "mode": self.mode,
                "chunk_tokens": self.chunk_tokens, "stride": self.stride,
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
