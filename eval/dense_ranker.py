"""A small bi-encoder for the locate step, used by exp_ranker.py and exp_scale_semantic.py.

bge-small-en-v1.5 (33M params): CLS pooling, unit-normalised, cosine. The query carries bge's
retrieval instruction prefix. Passage embeddings are memoised by text, so a passage that appears
in many haystacks is encoded once.
"""
import time

import numpy as np
import torch

class DenseRanker:
    """bge-style bi-encoder: CLS pooling, unit-normalised, cosine. Passage embeddings are memoised by text."""
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, name: str, device: str, batch_size: int = 64, max_len: int = 512):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device, self.batch_size, self.max_len = device, batch_size, max_len
        self.mem = {}
        self.encoded = 0
        self.seconds = 0.0

    @torch.no_grad()
    def _encode(self, texts):
        out = []
        for s in range(0, len(texts), self.batch_size):
            b = self.tok(texts[s:s + self.batch_size], padding=True, truncation=True, max_length=self.max_len,
                         return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    def embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.mem]
        if todo:
            t0 = time.time()
            for t, e in zip(todo, self._encode(todo)):
                self.mem[t] = e
            self.seconds += time.time() - t0; self.encoded += len(todo)
        return np.stack([self.mem[t] for t in texts])

    def rank(self, texts, query):
        q = self.embed([self.QUERY_PREFIX + query])[0]
        s = self.embed(texts) @ q
        return list(np.argsort(-s, kind="stable"))

