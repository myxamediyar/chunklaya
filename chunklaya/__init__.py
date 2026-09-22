"""chunklaya — long-input harness for Laya (open-weights System 1 decision model).

    chunk   token windows with overlap, so nothing sits deep in the decision head's decay zone
    locate  optional BM25 prefilter: hand Laya only the passages the question is about
    gate    one relevance noul per chunk, batched with the answers into one forward pass
    decide  aggregate per-chunk distributions (max / noisy-OR / mean / mixture / log-linear),
            or stack: Laya reads its own per-chunk outputs as a compact second-order state

    from chunklaya import ChunkLaya, predict_many
    cj = ChunkLaya(laya.load("convaiinnovations/laya", subfolder="multilingual"), chunk_tokens=750, stride=375)
    cj.ask(long_text, {"q": {"type": "noul", "instructions": "..."}})
    idx = cj.index(long_text)          # chunk + index once when one text takes many questions
    cj.ask(idx, {"q": ...})
"""
from .aggregate import choice_loglinear, choice_mixture, noul_any, noul_max, noul_mean, score_expected
from .batch import predict_many
from .chunk import Chunk, chunk_paragraphs, chunk_text
from .cache import PredictionCache
from .harness import ChunkIndex, ChunkLaya
from .prefilter import BM25Index, bm25_rank

__version__ = "0.1.0"
__all__ = ["ChunkLaya", "ChunkIndex", "BM25Index", "PredictionCache", "predict_many", "chunk_text", "chunk_paragraphs", "Chunk", "bm25_rank", "noul_max", "noul_any", "noul_mean",
           "choice_mixture", "choice_loglinear", "score_expected"]
