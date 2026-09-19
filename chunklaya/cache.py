"""On-disk cache of per-row Laya predictions, keyed by (state, question, window, checkpoint).

Borrowed from classifier.dev's eval habit of caching every response, but keyed per *row* rather than
per document: in paragraph mode the same passage appears in many documents (and in the clean-haystack
filter), so a rerun with a different haystack composition, threshold, or aggregation costs almost
nothing. Append-only JSONL, loaded into memory on open; safe to interrupt.
"""
import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np


def row_key(state: Any, question: Dict, max_len: int, head_max_len: int, checkpoint: str) -> str:
    blob = json.dumps([state, question, max_len, head_max_len, checkpoint], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


def _to_json(ans: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ans)
    out["p"] = [float(x) for x in ans["p"]]
    return out


def _from_json(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    out["p"] = np.asarray(d["p"], dtype=float)
    return out


class PredictionCache:
    def __init__(self, path: str):
        self.path = path
        self.mem: Dict[str, Dict[str, Any]] = {}
        self.hits = self.misses = 0
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        k, v = json.loads(line)
                        self.mem[k] = v
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a")

    def __len__(self) -> int:
        return len(self.mem)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        v = self.mem.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        return _from_json(v)

    def put(self, key: str, ans: Dict[str, Any]) -> None:
        v = _to_json(ans)
        self.mem[key] = v
        self._fh.write(json.dumps([key, v], ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def stats(self) -> str:
        return f"{len(self.mem)} rows cached, {self.hits} hits / {self.misses} misses this session"
