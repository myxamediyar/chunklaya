"""predict_many: run any list of (state, question) pairs through Laya in batched forward passes.

laya.agent.Agent.system_one takes ONE state and many questions. Nothing underneath requires that —
build_sequence and collate_items accept arbitrary rows — so this is the same loop over (state, question)
pairs. A single pair reproduces agent.predict(state, {qid: q}) to the float.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from laya.common import QTYPES, build_sequence, collate_items, confidence_from_probs, render_options, temp_bucket

Pair = Tuple[Any, Dict[str, Any]]


def _answer(q: Dict, p: np.ndarray, k: int, act_p: float, seq_len: int) -> Dict[str, Any]:
    """Laya-format answer fields plus the unrounded distribution `p` and its option `keys`."""
    out: Dict[str, Any] = {"type": q["t"], "p": p, "confidence": float(confidence_from_probs(p, k)),
                           "act_probability": act_p, "seq_len": seq_len}
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        out.update(keys=keys, choice=keys[int(p.argmax())], probabilities={kk: float(v) for kk, v in zip(keys, p)})
    elif q["t"] == "score":
        out.update(keys=[str(i) for i in range(k)], score=float((np.arange(k) * p).sum()),
                   legend={str(i): c for i, c in enumerate(q["crit"])},
                   probabilities={str(i): float(v) for i, v in enumerate(p)})
    else:
        pt = float(p[1])
        out.update(keys=["false", "true"], noul=pt, confidence=max(pt, 1.0 - pt))  # laya's noul confidence
    return out


@torch.no_grad()
def predict_many(agent, pairs: Sequence[Pair], batch_size: int = 32,
                 max_len: Optional[int] = None, head_max_len: Optional[int] = None,
                 cache=None) -> List[Dict[str, Any]]:
    """Returns one answer dict per pair, in order. `max_len` / `head_max_len` default to agent.cfg.
    `cache`: an optional chunklaya.cache.PredictionCache; hits skip the model, misses are written back."""
    max_len = max_len or agent.cfg.get("max_len", 512)
    head_max_len = head_max_len or agent.cfg.get("head_max_len", 192)
    out: List[Optional[Dict[str, Any]]] = [None] * len(pairs)

    todo = list(range(len(pairs)))
    keys: List[Optional[str]] = [None] * len(pairs)
    if cache is not None:
        from .cache import row_key
        ckpt = str(agent.cfg.get("encoder", "")) + "|" + str(getattr(agent, "checkpoint_id", ""))
        todo = []
        for i, (state, qdef) in enumerate(pairs):
            keys[i] = row_key(state, qdef, max_len, head_max_len, ckpt)
            hit = cache.get(keys[i])
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit

    items, qs = [], []
    for i in todo:
        state, qdef = pairs[i]
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError("question options exceed head_max_len=%d" % head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        qs.append(q)

    # length-sorted batching: rows in a batch are padded to the longest, so grouping similar lengths
    # cuts wasted compute (paragraph mode mixes 60- and 230-token rows). Output order is restored below.
    order = sorted(range(len(items)), key=lambda j: len(items[j]["ids"]))
    items = [items[j] for j in order]
    qs = [qs[j] for j in order]
    todo = [todo[j] for j in order]

    dev = agent.device
    use_amp = dev.type == "cuda"
    for s in range(0, len(items), batch_size):
        group = items[s:s + batch_size]
        b = collate_items([group], agent.tok.pad_token_id)
        with torch.autocast(device_type=dev.type, dtype=agent.dtype, enabled=use_amp):
            logits, act = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                      b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        for r, it in enumerate(group):
            j = s + r
            q, k, qt = qs[j], len(it["markers"]), QTYPES[qs[j]["t"]]
            t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            z = logits[r, :k] / max(1e-3, float(t_scale))
            p = np.exp(z - z.max())
            ans = _answer(q, p / p.sum(), k, float(act[r, 0]), len(it["ids"]))
            i = todo[j]
            out[i] = ans
            if cache is not None:
                cache.put(keys[i], ans)
    if cache is not None:
        cache.flush()
    return out  # type: ignore[return-value]
