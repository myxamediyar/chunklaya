"""Combine per-chunk distributions into one answer. Every rule takes optional per-chunk weights
(the relevance gate's P(true)); unweighted means uniform."""
from typing import Optional, Sequence, Tuple

import numpy as np


def _w(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


# --- noul -----------------------------------------------------------------------------------------

def noul_max(ps: Sequence[float]) -> float:
    """'Does X occur anywhere': the strongest chunk decides. Robust to many low-scoring chunks."""
    return float(np.max(ps))


def noul_any(ps: Sequence[float]) -> float:
    """Noisy-OR: P(at least one chunk is true) under independence. Inflates with many chunks whose
    scores are small but non-zero — prefer noul_max unless the detector is well calibrated."""
    return float(1.0 - np.prod(1.0 - np.asarray(ps, dtype=float)))


def noul_mean(ps: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """'Does X hold overall': relevance-weighted average."""
    return float(_w(weights, len(ps)) @ np.asarray(ps, dtype=float))


# --- choice / score ---------------------------------------------------------------------------------

def choice_mixture(dists: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Weighted average of distributions. Treats chunks as alternative sources; soft and stable."""
    D = np.asarray(dists, dtype=float)
    return _w(weights, len(D)) @ D


def choice_loglinear(dists: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None,
                     floor: float = 1e-6) -> np.ndarray:
    """Weighted geometric pooling. Treats chunks as independent evidence; sharpens agreement, but one
    confident wrong chunk can zero the right option — keep the relevance gate in the weights."""
    D = np.clip(np.asarray(dists, dtype=float), floor, 1.0)
    logp = _w(weights, len(D)) @ np.log(D)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def score_expected(dists: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None) -> Tuple[float, np.ndarray]:
    """Expectation is linear, so mixing then taking E[level] equals the weighted mean of per-chunk E."""
    mix = choice_mixture(dists, weights)
    return float((np.arange(len(mix)) * mix).sum()), mix
