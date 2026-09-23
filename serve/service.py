"""chunklaya as a System One endpoint: one document, many questions, one index.

Speaks the request and response shape classifier.dev's Worker (`src/jev.ts`
there) already sends to Beam and TypeSafe, so it can be reached through a
`Backend` descriptor (a URL and a bearer) rather than a new client. Where it
differs from those endpoints, it is deliberate:

  - Exactly one `state` per request. The document is chunked and indexed once
    and every question in the request is answered off that index, which is the
    `input` + `dimensions` shape of POST /v1/classify. A request with more
    states is refused whole, never partly answered.
  - Two strategies. `scan` scores every passage and aggregates: the right
    shape for "which category is this" and "does X occur anywhere", costing one
    Laya row per passage plus a gate row for choice and score questions.
    `locate` ranks passages by BM25 against the question and scores only the
    top `top_k`: one forward pass at any document size, which is what held to
    a million tokens in the needle results. Scan is bounded by passage count
    so a million-token document cannot become ten thousand rows behind a
    proxy's 100-second limit; past the bound the refusal says to use locate.
  - Indexes live in memory only, keyed by a hash of the text, bounded by count
    and by total passages, and expire. Nothing is written to disk, no request
    text reaches a log line, and `cache=None` is passed to chunklaya so its
    on-disk prediction cache stays off. Logs carry counts and durations.

Refusals use the vocabulary `beamErrorType` in that Worker already reads:
`{error: {code}}` on every error, and a plain `detail` string on 422.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

try:  # numpy is only needed to flatten chunklaya's answers into plain JSON.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

log = logging.getLogger("chunklaya")

MODEL_ID = "convaiinnovations/laya"
DEFAULT_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
STRATEGIES = ("scan", "locate")
QUESTION_TYPES = ("choice", "noul", "score")
AGG_MODES = {"noul": ("max", "any", "mean"), "choice": ("mixture", "loglinear", "stack"), "score": ("mixture", "stack")}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Limits:
    """Every ceiling the service enforces. Defaults are read from the environment once, at startup."""

    max_chars: int = 4_000_000  # about a million tokens of prose
    max_body_bytes: int = 12_000_000
    max_questions: int = 32
    max_labels: int = 32
    max_instruction_chars: int = 4_000
    max_scan_chunks: int = 256
    max_rows: int = 4_096
    max_top_k: int = 8
    max_inflight: int = 8
    index_workers: int = 2
    queue_wait_s: float = 5.0
    request_budget_s: float = 80.0  # under the RunPod proxy's 100-second cut-off
    cache_items: int = 64
    cache_chunks: int = 200_000
    cache_ttl_s: float = 600.0

    @classmethod
    def from_env(cls) -> "Limits":
        out = {}
        for name, default in cls.__dataclass_fields__.items():
            env = name.upper()
            out[name] = _env_float(env, default.default) if isinstance(default.default, float) else _env_int(env, default.default)
        return cls(**out)

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("max_chars", "max_questions", "max_labels", "max_scan_chunks", "max_rows", "max_top_k", "request_budget_s")}


class Refusal(Exception):
    """An answer the caller can act on: an HTTP status, a stable code, and a sentence."""

    def __init__(self, status: int, code: str, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after

    def response(self) -> JSONResponse:
        headers = {"retry-after": str(self.retry_after)} if self.retry_after else {}
        body: dict[str, Any] = {"error": {"code": self.code, "type": self.code, "message": self.message}}
        if self.status == 422:
            body["detail"] = self.message
        return JSONResponse(body, status_code=self.status, headers=headers)


class IndexCache:
    """LRU of ChunkIndex objects, in memory only, bounded by count and by total passages, with a TTL."""

    def __init__(self, max_items: int, max_chunks: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self._items: "OrderedDict[str, tuple[Any, float, int]]" = OrderedDict()
        self._chunks = 0
        self._lock = threading.Lock()
        self.max_items, self.max_chunks, self.ttl_s, self.clock = max_items, max_chunks, ttl_s, clock
        self.hits = self.misses = 0

    def get(self, key: str):
        with self._lock:
            self._expire()
            entry = self._items.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key: str, index: Any) -> None:
        n = len(index)
        if n > self.max_chunks or self.max_items <= 0:
            return  # too large to ever fit; serve it once and let it go
        with self._lock:
            if key in self._items:
                return
            self._items[key] = (index, self.clock() + self.ttl_s, n)
            self._chunks += n
            while self._items and (len(self._items) > self.max_items or self._chunks > self.max_chunks):
                self._drop(next(iter(self._items)))

    def _expire(self) -> None:
        now = self.clock()
        for key in [k for k, (_, expires, _) in self._items.items() if expires <= now]:
            self._drop(key)

    def _drop(self, key: str) -> None:
        _, _, n = self._items.pop(key)
        self._chunks -= n

    def stats(self) -> dict:
        with self._lock:
            return {"items": len(self._items), "chunks": self._chunks, "hits": self.hits, "misses": self.misses}


class Admission:
    """Bounded work. Refusing is explicit and immediate; nothing queues on the GPU without a deadline."""

    def __init__(self, limits: Limits):
        self.inflight = threading.BoundedSemaphore(limits.max_inflight)
        self.index = threading.Semaphore(limits.index_workers)
        self.gpu = threading.Lock()
        self.queue_wait_s = limits.queue_wait_s
        self._count = 0
        self._count_lock = threading.Lock()

    def enter(self) -> bool:
        if not self.inflight.acquire(blocking=False):
            return False
        with self._count_lock:
            self._count += 1
        return True

    def leave(self) -> None:
        with self._count_lock:
            self._count -= 1
        self.inflight.release()

    @property
    def pending(self) -> int:
        with self._count_lock:
            return self._count


@dataclass
class Runtime:
    """Everything a request needs, built once at startup. Tests build one around a fake harness."""

    model: str
    checkpoint: str
    device: str
    scan: Any  # a ChunkLaya (or stand-in) with prefilter=None
    locate_factory: Callable[[int], Any]  # top_k -> ChunkLaya with prefilter="bm25"
    limits: Limits
    cache: IndexCache
    admission: Admission
    inference_context: Callable[[], Any] = contextlib.nullcontext
    ready: bool = False
    _locators: dict = field(default_factory=dict)
    _locators_lock: threading.Lock = field(default_factory=threading.Lock)

    def locate(self, top_k: int):
        with self._locators_lock:
            harness = self._locators.get(top_k)
            if harness is None:
                harness = self._locators[top_k] = self.locate_factory(top_k)
            return harness


@dataclass
class Parsed:
    text: str
    questions: dict[str, dict]
    strategy: str
    top_k: int
    agg: Optional[dict[str, str]]
    detectors: Optional[dict[str, tuple]]
    detail: bool


# --- request parsing -------------------------------------------------------

def _is_str(v: Any, limit: int) -> bool:
    return isinstance(v, str) and 0 < len(v) <= limit


def parse_request(body: Any, limits: Limits) -> Parsed:
    if not isinstance(body, dict):
        raise Refusal(400, "invalid_request", "request body must be a JSON object")
    state = body.get("state")
    if not isinstance(state, list) or not state:
        raise Refusal(422, "invalid_request", "state must be a list holding exactly one item")
    if len(state) != 1:
        raise Refusal(422, "one_state_per_request", "this endpoint answers questions about one document per request; send one state item")
    item = state[0]
    if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
        raise Refusal(422, "invalid_request", "state[0].text must be non-empty text")
    text = item["text"]
    if len(text) > limits.max_chars:
        raise Refusal(413, "document_too_large", f"document is {len(text):,} characters; this endpoint accepts at most {limits.max_chars:,}")
    rubric = item.get("rubric")
    if rubric is not None and not _is_str(rubric, limits.max_instruction_chars):
        raise Refusal(422, "invalid_request", "rubric must be a short string when present")

    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise Refusal(422, "invalid_request", "questions must be a non-empty object")
    if len(questions) > limits.max_questions:
        raise Refusal(422, "invalid_request", f"at most {limits.max_questions} questions per request")
    parsed_questions: dict[str, dict] = {}
    for qid, q in questions.items():
        if not _is_str(qid, 128) or not isinstance(q, dict):
            raise Refusal(422, "invalid_request", "each question needs a short id and an object")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise Refusal(422, "invalid_request", f"question {qid!r}: type must be one of {', '.join(QUESTION_TYPES)}")
        instructions = q.get("instructions")
        if not _is_str(instructions, limits.max_instruction_chars):
            raise Refusal(422, "invalid_request", f"question {qid!r}: instructions must be a non-empty string")
        out: dict[str, Any] = {"type": qtype, "instructions": instructions if not rubric else f"{instructions}\nRubric: {rubric}"}
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not (2 <= len(criteria) <= limits.max_labels):
                raise Refusal(422, "invalid_request", f"question {qid!r}: choice needs criteria with 2-{limits.max_labels} labels")
            for label, meaning in criteria.items():
                if not _is_str(label, 200) or not (meaning is None or _is_str(meaning, 500)):
                    raise Refusal(422, "invalid_request", f"question {qid!r}: each criterion is a label with null or a short meaning")
            out["criteria"] = dict(criteria)
        elif qtype == "score":
            if not isinstance(criteria, list) or not (2 <= len(criteria) <= limits.max_labels) or not all(_is_str(c, 500) for c in criteria):
                raise Refusal(422, "invalid_request", f"question {qid!r}: score needs a list of 2-{limits.max_labels} rung descriptions")
            out["criteria"] = list(criteria)
        elif criteria is not None:
            raise Refusal(422, "invalid_request", f"question {qid!r}: noul takes no criteria")
        parsed_questions[qid] = out

    strategy = body.get("strategy", "scan")
    if strategy not in STRATEGIES:
        raise Refusal(422, "invalid_request", f"strategy must be one of {', '.join(STRATEGIES)}")
    top_k = body.get("top_k", 1)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not (1 <= top_k <= limits.max_top_k):
        raise Refusal(422, "invalid_request", f"top_k must be an integer from 1 to {limits.max_top_k}")

    agg = body.get("agg")
    if agg is not None:
        if not isinstance(agg, dict):
            raise Refusal(422, "invalid_request", "agg maps question ids to aggregation modes")
        for qid, mode in agg.items():
            if qid not in parsed_questions or mode not in AGG_MODES[parsed_questions[qid]["type"]]:
                raise Refusal(422, "invalid_request", f"agg[{qid!r}] must be one of {', '.join(AGG_MODES.get(parsed_questions.get(qid, {}).get('type'), ()))}")
        agg = dict(agg)

    detectors = body.get("detectors")
    if detectors is not None:
        if not isinstance(detectors, dict):
            raise Refusal(422, "invalid_request", "detectors maps a noul question id to [choice question, option]")
        parsed_detectors: dict[str, tuple] = {}
        for qid, spec in detectors.items():
            target = parsed_questions.get(qid)
            if target is None or target["type"] != "noul":
                raise Refusal(422, "invalid_request", f"detectors[{qid!r}] must name a noul question in this request")
            if not (isinstance(spec, list) and len(spec) == 2 and isinstance(spec[0], dict) and isinstance(spec[1], str)):
                raise Refusal(422, "invalid_request", f"detectors[{qid!r}] must be [choice question, option label]")
            det_q, option = spec
            det_criteria = det_q.get("criteria")
            if det_q.get("type") != "choice" or not _is_str(det_q.get("instructions"), limits.max_instruction_chars) \
                    or not isinstance(det_criteria, dict) or option not in det_criteria or not (2 <= len(det_criteria) <= limits.max_labels):
                raise Refusal(422, "invalid_request", f"detectors[{qid!r}]: the detector is a choice question whose criteria include the option")
            parsed_detectors[qid] = ({"type": "choice", "instructions": det_q["instructions"], "criteria": dict(det_criteria)}, option)
        detectors = parsed_detectors

    detail = body.get("detail", False)
    if not isinstance(detail, bool):
        raise Refusal(422, "invalid_request", "detail must be true or false")
    return Parsed(text=text, questions=parsed_questions, strategy=strategy, top_k=top_k, agg=agg, detectors=detectors, detail=detail)


# --- answering -------------------------------------------------------------

def _plain(value: Any) -> Any:
    """chunklaya's answers carry numpy scalars and arrays; the wire carries JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def project_answer(qdef: dict, answer: dict, detail: bool) -> dict:
    """The fields a caller validates, plus what an operator wants to know, minus chunklaya's internals."""
    qtype = qdef["type"]
    out: dict[str, Any] = {"type": qtype}
    if qtype == "choice":
        out.update(choice=answer.get("choice"), confidence=answer.get("confidence"), probabilities=answer.get("probabilities"))
    elif qtype == "noul":
        out.update(noul=answer.get("noul"), confidence=answer.get("confidence"))
    else:
        out.update(score=answer.get("score"), confidence=answer.get("confidence"), probabilities=answer.get("probabilities"), legend=answer.get("legend"))
    for key in ("act_probability", "agg", "n_scored"):
        if key in answer:
            out[key] = answer[key]
    if detail and "chunks" in answer:
        out["chunks"] = answer["chunks"]
    return _plain(out)


def _rows_estimate(strategy: str, n_chunks: int, top_k: int, questions: dict[str, dict]) -> int:
    scored = n_chunks if strategy == "scan" else min(top_k, n_chunks)
    gate = scored > 1
    return sum(scored + (scored if gate and q["type"] != "noul" else 0) for q in questions.values())


def cache_key(rt: Runtime, text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() + ":" + rt.model


def answer(rt: Runtime, parsed: Parsed) -> dict:
    """Runs on a worker thread. Index on CPU under its own bound, then one GPU turn with a deadline."""
    limits = rt.limits
    harness = rt.scan if parsed.strategy == "scan" else rt.locate(parsed.top_k)
    key = cache_key(rt, parsed.text)
    started = time.perf_counter()
    index = rt.cache.get(key)
    cached = index is not None
    if index is None:
        if not rt.admission.index.acquire(timeout=rt.admission.queue_wait_s):
            raise Refusal(429, "busy", "indexing capacity is busy; retry shortly", retry_after=1)
        try:
            index = harness.index(parsed.text)
        finally:
            rt.admission.index.release()
        rt.cache.put(key, index)
    index_ms = (time.perf_counter() - started) * 1000
    n_chunks = len(index)
    if parsed.strategy == "scan" and n_chunks > limits.max_scan_chunks:
        raise Refusal(422, "too_many_passages",
                      f"the document has {n_chunks} passages and scan scores at most {limits.max_scan_chunks}; "
                      f"use strategy \"locate\" for a lookup, or send a shorter document")
    rows = _rows_estimate(parsed.strategy, n_chunks, parsed.top_k, parsed.questions)
    if rows > limits.max_rows:
        raise Refusal(422, "too_many_rows", f"this request would score {rows} passages across its questions; the ceiling is {limits.max_rows}. Ask fewer questions or use strategy \"locate\"")

    if not rt.admission.gpu.acquire(timeout=rt.admission.queue_wait_s):
        raise Refusal(429, "busy", "the model is busy; retry shortly", retry_after=1)
    asked = time.perf_counter()
    try:
        with rt.inference_context():
            result = harness.ask(index, parsed.questions, agg=parsed.agg, detectors=parsed.detectors)
    except (ValueError, TypeError, KeyError) as e:
        raise Refusal(422, "invalid_request", f"the model refused the question: {e}") from e
    except RuntimeError as e:  # CUDA out of memory arrives here
        raise Refusal(503, "inference_unavailable", "inference failed; retry shortly", retry_after=5) from e
    finally:
        rt.admission.gpu.release()
    ask_ms = (time.perf_counter() - asked) * 1000

    answers = {qid: project_answer(parsed.questions[qid], result["answers"][qid], parsed.detail) for qid in parsed.questions}
    usage = result.get("usage") or {}
    document_tokens = sum(max(0, c.tok_end - c.tok_start) for c in index.chunks)
    return {
        "model": rt.model,
        "checkpoint": rt.checkpoint,
        "strategy": parsed.strategy,
        "n_chunks": n_chunks,
        "answers": answers,
        "usage": {"input_tokens": int(usage.get("input_tokens", 0) or 0), "output_tokens": 0, "document_tokens": int(document_tokens)},
        "timing": {"index_ms": round(index_ms, 1), "ask_ms": round(ask_ms, 1), "cached": cached},
    }


# --- HTTP ------------------------------------------------------------------

async def read_json(request: Request, max_bytes: int) -> Any:
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > max_bytes:
            raise Refusal(413, "request_too_large", f"request body exceeds {max_bytes:,} bytes")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise Refusal(400, "invalid_json", "request body is not valid JSON") from e


def _token_from(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[7:] if header.lower().startswith("bearer ") else ""


def create_app(runtime_factory: Callable[[], Runtime], token: Optional[str] = None) -> FastAPI:
    """`runtime_factory` runs once, on a thread, when the server starts; /health answers 503 until it returns."""
    if token is None:
        token = os.environ.get("CHUNKLAYA_TOKEN", "")
        if not token and os.environ.get("CHUNKLAYA_ALLOW_ANON") != "1":
            raise RuntimeError("CHUNKLAYA_TOKEN is not set; set it, or CHUNKLAYA_ALLOW_ANON=1 for a local run")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = await asyncio.to_thread(runtime_factory)
        rt.ready = True
        app.state.runtime = rt
        log.info("ready model=%s checkpoint=%s device=%s", rt.model, rt.checkpoint, rt.device)
        yield

    api = FastAPI(title="chunklaya", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    api.state.runtime = None

    def runtime() -> Runtime:
        rt = api.state.runtime
        if rt is None or not rt.ready:
            raise Refusal(503, "starting", "the model is still loading", retry_after=5)
        return rt

    def authorized(request: Request) -> bool:
        return not token or hmac.compare_digest(_token_from(request).encode(), token.encode())

    @api.exception_handler(Refusal)
    async def _refused(_: Request, exc: Refusal):
        return exc.response()

    @api.get("/")
    async def root():
        return PlainTextResponse("chunklaya: POST /v1/systemone with one state and up to 32 questions. GET /health.\n")

    @api.get("/health")
    async def health():
        rt = api.state.runtime
        if rt is None or not rt.ready:
            return JSONResponse({"ok": False, "status": "starting"}, status_code=503, headers={"retry-after": "5"})
        return {"ok": True, "model": rt.model, "checkpoint": rt.checkpoint, "device": rt.device, "strategies": list(STRATEGIES),
                "limits": rt.limits.public(), "cache": rt.cache.stats(), "pending": rt.admission.pending}

    @api.post("/v1/systemone")
    async def systemone(request: Request):
        rt = runtime()
        if not authorized(request):
            raise Refusal(401, "unauthorized", "a valid bearer token is required")
        parsed = parse_request(await read_json(request, rt.limits.max_body_bytes), rt.limits)
        if not rt.admission.enter():
            raise Refusal(429, "busy", "too many requests in flight; retry shortly", retry_after=1)
        started = time.perf_counter()
        status, result = 200, None

        def work():
            try:
                return answer(rt, parsed)
            finally:
                rt.admission.leave()

        future = asyncio.get_running_loop().run_in_executor(None, work)
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=rt.limits.request_budget_s)
        except asyncio.TimeoutError:
            status = 504
            # The thread finishes on its own and releases every lock it holds; only the answer is abandoned.
            raise Refusal(504, "timeout", f"the request did not finish within {rt.limits.request_budget_s:.0f} seconds")
        except Refusal as e:
            status = e.status
            raise
        finally:
            log.info("systemone status=%d strategy=%s questions=%d chunks=%s cached=%s ms=%.0f",
                     status, parsed.strategy, len(parsed.questions),
                     result["n_chunks"] if result else "-", result["timing"]["cached"] if result else "-",
                     (time.perf_counter() - started) * 1000)
        return result

    return api


# --- the real runtime ------------------------------------------------------

def bm25_query(qdef: dict) -> str:
    """What the question says, for ranking passages: the instructions and any label meanings.

    Label *names* are deliberately left out. A passage that happens to contain
    the word "technical" is not the passage the question is about, and putting
    names in the query hands Laya exactly that passage, with certainty (the
    first live run flipped a billing answer to technical this way). Meanings are
    what the author's `question_query` uses; the only change here is that a
    `{label: null}` criterion contributes nothing instead of the word "None".
    """
    parts = [qdef.get("instructions", "")]
    criteria = qdef.get("criteria")
    if isinstance(criteria, dict):
        parts.extend(str(meaning) for meaning in criteria.values() if meaning is not None)
    elif isinstance(criteria, (list, tuple)):
        parts.extend(str(c) for c in criteria)
    return " ".join(parts)


def build_runtime(limits: Optional[Limits] = None) -> Runtime:
    """Load the pinned checkpoint from the image's offline cache and warm both strategies before serving."""
    import torch
    import laya
    from chunklaya import ChunkLaya
    from huggingface_hub import snapshot_download

    limits = limits or Limits.from_env()
    revision = os.environ.get("LAYA_REVISION", DEFAULT_REVISION)
    subfolder = os.environ.get("LAYA_SUBFOLDER", "multilingual") or None
    device = os.environ.get("LAYA_DEVICE", "cuda")
    path = os.environ.get("LAYA_MODEL_PATH") or snapshot_download(MODEL_ID, revision=revision, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    agent = laya.load(path, device=device, subfolder=subfolder)
    if device == "cuda" and agent.device.type != "cuda":
        raise RuntimeError("Laya did not initialise on CUDA; refusing to serve on CPU by accident")

    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    chunk_tokens = _env_int("CHUNK_TOKENS", 750)
    ceiling = max_len - head_max_len - 8
    if chunk_tokens > ceiling:
        log.warning("CHUNK_TOKENS=%d does not fit this checkpoint's %d-token window beside a %d-token question; using %d", chunk_tokens, max_len, head_max_len, ceiling)
        chunk_tokens = ceiling
    batch_size = _env_int("BATCH_SIZE", 32)

    class Harness(ChunkLaya):
        def _select(self, idx, qdef):
            chunks = idx.chunks
            if self.prefilter is None or len(chunks) <= self.top_k:
                return chunks
            keep = sorted(idx.bm25.rank(bm25_query(qdef))[: self.top_k])
            return [chunks[i] for i in keep]

    common = dict(chunk_tokens=chunk_tokens, batch_size=batch_size, mode="paragraphs", cache=None)
    scan = Harness(agent, **common)
    rt = Runtime(
        model=f"chunklaya/{subfolder or 'english'}",
        checkpoint=revision,
        device=str(agent.device),
        scan=scan,
        locate_factory=lambda top_k: Harness(agent, prefilter="bm25", top_k=top_k, **common),
        limits=limits,
        cache=IndexCache(limits.cache_items, limits.cache_chunks, limits.cache_ttl_s),
        admission=Admission(limits),
        inference_context=torch.inference_mode,
    )

    # Pay for tokenizer, CUDA kernels and allocator setup now, not on the first real request.
    warm_text = "The invoice was paid twice on Monday.\n\nA refund was requested by email.\n\nSupport confirmed the duplicate charge."
    warm_q = {"q": {"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": None, "technical": None}},
              "n": {"type": "noul", "instructions": "Does the text mention a refund?"}}
    with torch.inference_mode():
        idx = scan.index(warm_text)
        scan.ask(idx, warm_q)
        rt.locate(1).ask(idx, {"n": warm_q["n"]})
    log.info("warm: chunk_tokens=%d max_len=%d batch_size=%d", chunk_tokens, max_len, batch_size)
    return rt


if os.environ.get("CHUNKLAYA_NO_APP") != "1":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    api = create_app(build_runtime)
