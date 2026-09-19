"""Token-bounded windows over a string, with overlap. Chunk text is an exact substring of the input."""
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    index: int
    text: str
    tok_start: int
    tok_end: int      # exclusive
    char_start: int
    char_end: int     # exclusive

    @property
    def n_tokens(self) -> int:
        return self.tok_end - self.tok_start


def chunk_text(tok, text: str, chunk_tokens: int = 750, stride: int = None) -> List[Chunk]:
    """Windows of `chunk_tokens` tokens every `stride` tokens (stride == chunk_tokens → no overlap).

    With stride <= chunk_tokens/2 every span shorter than chunk_tokens/2 lies wholly inside at least one
    window, and every token is within the first `stride` tokens of some window — which is what keeps a
    needle out of the decision head's positional decay zone (see README). A trailing window shorter than
    chunk_tokens/4 is merged into the previous one rather than emitted as a stub.
    """
    stride = stride or chunk_tokens
    if not 0 < stride <= chunk_tokens:
        raise ValueError("need 0 < stride <= chunk_tokens")
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n = len(offs)
    if n <= chunk_tokens:
        return [Chunk(0, text, 0, n, 0, len(text))]
    chunks, s = [], 0
    while True:
        e = min(s + chunk_tokens, n)
        cs, ce = offs[s][0], offs[e - 1][1]
        chunks.append(Chunk(len(chunks), text[cs:ce], s, e, cs, ce))
        if e >= n:
            break
        s += stride
    if len(chunks) > 1 and chunks[-1].n_tokens < chunk_tokens // 4:
        tail = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(prev.index, text[prev.char_start:tail.char_end], prev.tok_start, tail.tok_end,
                           prev.char_start, tail.char_end)
    return chunks


def chunk_paragraphs(tok, text: str, max_tokens: int = 750, sep: str = "\n\n") -> List[Chunk]:
    """One semantic unit per chunk: split on `sep` (blank lines by default), no packing.

    Exists because the decision head characterizes a state by its opening: in a window of several
    unrelated passages, a passage that starts past ~200 tokens is nearly invisible to a noul
    (results/2026-09-19-chunked, Exp 3). Making each passage its own chunk puts every passage at
    offset 0, and turns an existence question into per-passage classification. A single paragraph
    longer than max_tokens falls back to token windows.
    """
    chunks, pos, tok_pos = [], 0, 0
    for piece in text.split(sep):
        start = text.index(piece, pos) if piece else pos
        end = start + len(piece)
        pos = end
        if not piece.strip():
            continue
        n = len(tok(piece, add_special_tokens=False)["input_ids"])
        if n <= max_tokens:
            chunks.append(Chunk(len(chunks), piece, tok_pos, tok_pos + n, start, end))
        else:
            for sub in chunk_text(tok, piece, max_tokens, max_tokens // 2):
                chunks.append(Chunk(len(chunks), sub.text, tok_pos + sub.tok_start, tok_pos + sub.tok_end,
                                    start + sub.char_start, start + sub.char_end))
        tok_pos += n
    return chunks or [Chunk(0, text, 0, 0, 0, len(text))]
