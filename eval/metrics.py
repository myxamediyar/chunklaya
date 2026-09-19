"""Small rank-based metrics used across experiments."""
from typing import Sequence


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(random positive scores above random negative); ties count half. 0.5 = chance."""
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def recall_at_spec(pos: Sequence[float], neg: Sequence[float], spec: float = 0.9) -> float:
    """Recall at the threshold that keeps `spec` of negatives below it."""
    thr = sorted(neg)[min(len(neg) - 1, int(spec * len(neg)))]
    return sum(p >= thr for p in pos) / len(pos)


def quartiles(xs: Sequence[float]) -> str:
    s = sorted(xs); n = len(s)
    return f"q25={s[n//4]:.3f} q50={s[n//2]:.3f} q75={s[3*n//4]:.3f}"


def auc_ci(pos: Sequence[float], neg: Sequence[float], n_boot: int = 1000, seed: int = 0):
    """AUC with a bootstrap 95% interval (resample positives and negatives independently)."""
    import numpy as np
    P, N = np.asarray(pos, float), np.asarray(neg, float)
    if len(P) == 0 or len(N) == 0:
        return float("nan"), float("nan"), float("nan")
    def _auc(p, n):
        d = p[:, None] - n[None, :]
        return float(((d > 0) + 0.5 * (d == 0)).mean())
    rng = np.random.default_rng(seed)
    boots = [_auc(P[rng.integers(0, len(P), len(P))], N[rng.integers(0, len(N), len(N))]) for _ in range(n_boot)]
    return _auc(P, N), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
