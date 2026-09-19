"""Deterministic state builders shared by the experiments. Each item gets its own random.Random, so
one needle's haystack is identical across positions and the `none` control is that same haystack."""
import json
import random
from typing import Dict, List, Optional, Tuple

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
CHOICE_Q = {"type": "choice", "instructions": "Which news category does this text belong to?",
            "criteria": {"World": "international and political news", "Sports": "sports and athletics",
                         "Business": "business, markets, and finance", "Sci/Tech": "science and technology"}}
NOUL_Q = {"type": "noul", "instructions": "Does the text contain a news item about sports or athletics?"}


class NeedleData:
    def __init__(self, tok, path: str, seed: int = 0):
        d = json.load(open(path))
        self.tok, self.seed = tok, seed
        self.by_cls: Dict[str, List[str]] = {l: [r["text"] for r in d["rows"] if d["labels"][r["label"]] == l] for l in LABELS}
        self.non_sports = [t for l in LABELS if l != "Sports" for t in self.by_cls[l]]
        self._nt: Dict[str, int] = {}

    def ntok(self, s: str) -> int:
        if s not in self._nt:
            self._nt[s] = len(self.tok(s, add_special_tokens=False)["input_ids"])
        return self._nt[s]

    def _fill(self, rng: random.Random, pool: List[str], target_tokens: int, exclude=()) -> List[str]:
        pool = [p for p in pool if p not in exclude]
        rng.shuffle(pool)
        out, n = [], 0
        for p in pool:
            if n >= target_tokens:
                break
            out.append(p); n += self.ntok(p) + 2
        return out

    # --- Exp 1: same-class filler, whole-state choice ---------------------------------------------
    def exp1_items(self, n: int) -> List[Tuple[str, str]]:
        rng = random.Random(self.seed * 1000 + 1)
        return [(cls, t) for cls in LABELS for t in rng.sample(self.by_cls[cls], n)]

    def exp1_state(self, i: int, cls: str, target: str, target_tokens: int) -> str:
        rng = random.Random(self.seed * 1000 + 100 + i)
        return "\n\n".join([target] + self._fill(rng, self.by_cls[cls], target_tokens - self.ntok(target), {target}))

    # --- Exp 2: one sports needle in a non-sports haystack ---------------------------------------
    def exp2_needles(self, n: int) -> List[str]:
        return random.Random(self.seed * 1000 + 2).sample(self.by_cls["Sports"], n)

    def exp2_haystack(self, i: int, needle: str, total_tokens: int = 3800) -> List[str]:
        rng = random.Random(self.seed * 1000 + 200 + i)
        return self._fill(rng, self.non_sports, total_tokens - self.ntok(needle))

    @staticmethod
    def place(hay: List[str], needle: Optional[str], pos: str) -> str:
        if needle is None or pos == "none":
            parts = hay
        elif pos == "start":
            parts = [needle] + hay
        elif pos == "end":
            parts = hay + [needle]
        else:
            parts = hay[:len(hay) // 2] + [needle] + hay[len(hay) // 2:]
        return "\n\n".join(parts)

    # --- Exp 3: needle at a chosen token offset inside one window -----------------------------------
    def exp3_state(self, i: int, needle: Optional[str], offset_tokens: int, total_tokens: int = 950) -> Tuple[str, int]:
        """Returns (state, actual needle token offset). Haystack shared across offsets for item i."""
        rng = random.Random(self.seed * 1000 + 300 + i)
        hay = self._fill(rng, self.non_sports, total_tokens - (self.ntok(needle) if needle else 0))
        if needle is None:
            return "\n\n".join(hay), -1
        parts, n, k = [], 0, 0
        while k < len(hay) and n + self.ntok(hay[k]) + 2 <= offset_tokens:
            parts.append(hay[k]); n += self.ntok(hay[k]) + 2; k += 1
        parts.append(needle)
        return "\n\n".join(parts + hay[k:]), n
