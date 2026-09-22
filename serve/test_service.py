"""Contract tests for the service around a fake harness. No torch, no weights, no GPU.

Run: CHUNKLAYA_NO_APP=1 python -m unittest serve/test_service.py
Needs fastapi and httpx; skips itself cleanly when they are absent so the
harness's own tests are unaffected.
"""
import os
import sys
import unittest
from dataclasses import dataclass

os.environ["CHUNKLAYA_NO_APP"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError as e:  # the web stack is a serving dependency, not a harness one
    raise unittest.SkipTest(f"service tests need fastapi and httpx: {e}")

import service
from service import Admission, IndexCache, Limits, Runtime, create_app, parse_request, Refusal

TOKEN = "test-token"


@dataclass
class Chunk:
    index: int
    text: str
    tok_start: int
    tok_end: int


class FakeIndex:
    def __init__(self, text, params):
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        self.chunks = [Chunk(i, p, i * 10, i * 10 + 10) for i, p in enumerate(paragraphs)]
        self.params = params

    def __len__(self):
        return len(self.chunks)


class FakeHarness:
    """Answers with the shape chunklaya returns, including numpy scalars, and records what it was asked."""

    def __init__(self, prefilter=None, top_k=1):
        self.prefilter, self.top_k = prefilter, top_k
        self.params = ("paragraphs", 750, 375)
        self.indexed = 0
        self.asked = []
        self.fail_with = None

    def index(self, text):
        self.indexed += 1
        return FakeIndex(text, self.params)

    def ask(self, index, questions, agg=None, detectors=None):
        if self.fail_with:
            raise self.fail_with
        self.asked.append((dict(questions), agg, detectors))
        answers = {}
        m = len(index)
        scored = m if self.prefilter is None else min(self.top_k, m)
        for qid, q in questions.items():
            detail = [{"index": i, "tok_start": i * 10, "tok_end": i * 10 + 10, "p": np.array([0.2, 0.8]), "gate": None} for i in range(scored)]
            if q["type"] == "choice":
                keys = list(q["criteria"])
                probabilities = {k: (np.float64(0.9) if i == 0 else np.float64(0.1 / (len(keys) - 1))) for i, k in enumerate(keys)}
                answers[qid] = {"type": "choice", "choice": keys[0], "confidence": np.float32(0.88), "probabilities": probabilities,
                                "agg": "mixture", "n_scored": scored, "chunks": detail, "p": np.array([0.9, 0.1]), "keys": keys, "seq_len": 40}
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75, "confidence": 0.75, "agg": "max", "n_scored": scored, "chunks": detail}
            else:
                answers[qid] = {"type": "score", "score": np.float64(1.5), "confidence": 0.6, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.5},
                                "legend": {"0": "low", "1": "mid", "2": "high"}, "agg": "mixture", "n_scored": scored, "chunks": detail}
        return {"model": "chunklaya", "answers": answers, "n_chunks": m, "usage": {"input_tokens": 40 * len(questions) * scored, "output_tokens": 0}}


def make_runtime(limits=None):
    limits = limits or Limits()
    scan = FakeHarness()
    locators = {}

    def locate_factory(top_k):
        locators[top_k] = FakeHarness(prefilter="bm25", top_k=top_k)
        return locators[top_k]

    rt = Runtime(model="chunklaya/test", checkpoint="rev", device="cpu", scan=scan, locate_factory=locate_factory, limits=limits,
                 cache=IndexCache(limits.cache_items, limits.cache_chunks, limits.cache_ttl_s), admission=Admission(limits))
    rt.locators = locators
    return rt


DOC = "Please refund the duplicate invoice payment.\n\nThe technical team reset the router.\n\nNothing else happened."
CHOICE = {"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": None, "technical": None}}
NOUL = {"type": "noul", "instructions": "Does the text mention a refund?"}


def client_for(rt, token=TOKEN):
    app = create_app(lambda: rt, token=token)
    return TestClient(app), rt


def post(client, body, token=TOKEN):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/systemone", json=body, headers=headers)


class ParseTests(unittest.TestCase):
    def test_rejects_more_than_one_state(self):
        with self.assertRaises(Refusal) as cm:
            parse_request({"state": [{"text": "a"}, {"text": "b"}], "questions": {"q": NOUL}}, Limits())
        self.assertEqual((cm.exception.status, cm.exception.code), (422, "one_state_per_request"))

    def test_document_over_cap_is_413(self):
        with self.assertRaises(Refusal) as cm:
            parse_request({"state": [{"text": "x" * 11}], "questions": {"q": NOUL}}, Limits(max_chars=10))
        self.assertEqual(cm.exception.status, 413)

    def test_question_shapes(self):
        bad = [
            {"q": {"type": "choice", "instructions": "?", "criteria": {"only": None}}},
            {"q": {"type": "noul", "instructions": "?", "criteria": {"a": None, "b": None}}},
            {"q": {"type": "score", "instructions": "?", "criteria": "not a list"}},
            {"q": {"type": "essay", "instructions": "?"}},
            {"q": {"type": "noul"}},
        ]
        for questions in bad:
            with self.assertRaises(Refusal, msg=questions):
                parse_request({"state": [{"text": DOC}], "questions": questions}, Limits())

    def test_rubric_travels_with_every_question(self):
        parsed = parse_request({"state": [{"text": DOC, "rubric": "billing means money"}], "questions": {"q": NOUL, "c": CHOICE}}, Limits())
        for q in parsed.questions.values():
            self.assertTrue(q["instructions"].endswith("\nRubric: billing means money"))

    def test_detectors_and_agg_validate_against_the_questions(self):
        body = {"state": [{"text": DOC}], "questions": {"q": NOUL, "c": CHOICE}}
        parsed = parse_request({**body, "detectors": {"q": [CHOICE, "billing"]}, "agg": {"q": "mean", "c": "loglinear"}}, Limits())
        self.assertEqual(parsed.detectors["q"][1], "billing")
        self.assertEqual(parsed.agg, {"q": "mean", "c": "loglinear"})
        for extra in ({"detectors": {"c": [CHOICE, "billing"]}}, {"detectors": {"q": [CHOICE, "shipping"]}}, {"agg": {"c": "max"}}, {"top_k": 0}, {"strategy": "grep"}):
            with self.assertRaises(Refusal, msg=extra):
                parse_request({**body, **extra}, Limits())


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client, self.rt = client_for(make_runtime())
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_and_root(self):
        health = self.client.get("/health").json()
        self.assertTrue(health["ok"])
        self.assertEqual(health["model"], "chunklaya/test")
        self.assertIn("max_scan_chunks", health["limits"])
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_requires_bearer(self):
        self.assertEqual(post(self.client, {"state": [{"text": DOC}], "questions": {"q": NOUL}}, token="").status_code, 401)
        self.assertEqual(post(self.client, {"state": [{"text": DOC}], "questions": {"q": NOUL}}, token="wrong").status_code, 401)

    def test_scan_answers_in_the_system_one_shape(self):
        r = post(self.client, {"state": [{"id": "a", "text": DOC}], "model": "jev/laya", "questions": {"a": CHOICE, "a_0": NOUL}})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["model"], "chunklaya/test")
        self.assertEqual(body["n_chunks"], 3)
        choice = body["answers"]["a"]
        self.assertEqual(choice["choice"], "billing")
        self.assertIn(choice["choice"], CHOICE["criteria"])
        self.assertEqual(set(choice["probabilities"]), set(CHOICE["criteria"]))
        self.assertTrue(all(0 <= p <= 1 for p in choice["probabilities"].values()))
        self.assertAlmostEqual(choice["confidence"], 0.88, places=2)
        self.assertNotIn("chunks", choice)
        self.assertNotIn("p", choice)
        self.assertEqual(body["answers"]["a_0"]["noul"], 0.75)
        self.assertEqual(body["usage"]["output_tokens"], 0)
        self.assertEqual(body["usage"]["document_tokens"], 30)
        self.assertFalse(body["timing"]["cached"])

    def test_detail_returns_passages_as_plain_json(self):
        r = post(self.client, {"state": [{"text": DOC}], "questions": {"s": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}}, "detail": True})
        self.assertEqual(r.status_code, 200, r.text)
        score = r.json()["answers"]["s"]
        self.assertEqual(score["score"], 1.5)
        self.assertEqual(score["chunks"][0]["p"], [0.2, 0.8])
        self.assertEqual(len(score["chunks"]), 3)

    def test_second_request_hits_the_index_cache(self):
        body = {"state": [{"text": DOC}], "questions": {"q": NOUL}}
        post(self.client, body)
        r = post(self.client, body)
        self.assertTrue(r.json()["timing"]["cached"])
        self.assertEqual(self.rt.scan.indexed, 1)
        self.assertEqual(self.rt.cache.stats()["hits"], 1)

    def test_locate_uses_its_own_harness_and_shares_the_index(self):
        body = {"state": [{"text": DOC}], "questions": {"q": NOUL}}
        post(self.client, body)
        r = post(self.client, {**body, "strategy": "locate", "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["answers"]["q"]["n_scored"], 2)
        self.assertTrue(r.json()["timing"]["cached"])
        self.assertEqual(self.rt.locators[2].indexed, 0)

    def test_scan_is_bounded_by_passage_count(self):
        client, rt = client_for(make_runtime(Limits(max_scan_chunks=2)))
        with client:
            r = post(client, {"state": [{"text": DOC}], "questions": {"q": NOUL}})
            self.assertEqual(r.status_code, 422)
            self.assertEqual(r.json()["error"]["code"], "too_many_passages")
            self.assertIn("locate", r.json()["detail"])
            r = post(client, {"state": [{"text": DOC}], "questions": {"q": NOUL}, "strategy": "locate"})
            self.assertEqual(r.status_code, 200, r.text)

    def test_rows_are_bounded_across_questions(self):
        client, _ = client_for(make_runtime(Limits(max_rows=5)))
        with client:
            r = post(client, {"state": [{"text": DOC}], "questions": {"a": CHOICE, "b": CHOICE}})
            self.assertEqual((r.status_code, r.json()["error"]["code"]), (422, "too_many_rows"))

    def test_model_refusal_is_422_without_the_word_tokens(self):
        self.rt.scan.fail_with = ValueError("question 'q' options exceed head_max_len=192")
        r = post(self.client, {"state": [{"text": DOC}], "questions": {"q": CHOICE}})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("token", r.json()["detail"].lower())

    def test_runtime_failure_is_503(self):
        self.rt.scan.fail_with = RuntimeError("CUDA out of memory")
        r = post(self.client, {"state": [{"text": DOC}], "questions": {"q": CHOICE}})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get("retry-after"), "5")

    def test_full_lanes_refuse_immediately(self):
        client, rt = client_for(make_runtime(Limits(max_inflight=1)))
        with client:
            self.assertTrue(rt.admission.enter())
            r = post(client, {"state": [{"text": DOC}], "questions": {"q": NOUL}})
            self.assertEqual((r.status_code, r.headers.get("retry-after")), (429, "1"))
            rt.admission.leave()
            self.assertEqual(post(client, {"state": [{"text": DOC}], "questions": {"q": NOUL}}).status_code, 200)

    def test_busy_gpu_refuses_after_the_queue_wait(self):
        client, rt = client_for(make_runtime(Limits(queue_wait_s=0.05)))
        with client:
            rt.admission.gpu.acquire()
            try:
                r = post(client, {"state": [{"text": DOC}], "questions": {"q": NOUL}})
                self.assertEqual((r.status_code, r.json()["error"]["code"]), (429, "busy"))
            finally:
                rt.admission.gpu.release()

    def test_oversized_body_is_413_and_bad_json_400(self):
        client, _ = client_for(make_runtime(Limits(max_body_bytes=50)))
        with client:
            r = client.post("/v1/systemone", content=b"{" + b" " * 60 + b"}", headers={"authorization": f"Bearer {TOKEN}", "content-type": "application/json"})
            self.assertEqual(r.status_code, 413)
        r = self.client.post("/v1/systemone", content=b"{not json", headers={"authorization": f"Bearer {TOKEN}", "content-type": "application/json"})
        self.assertEqual(r.status_code, 400)


class CacheTests(unittest.TestCase):
    def test_bounds_and_expiry(self):
        now = [0.0]
        cache = IndexCache(max_items=2, max_chunks=5, ttl_s=10, clock=lambda: now[0])
        a, b, c = FakeIndex("1\n\n2", ()), FakeIndex("1\n\n2\n\n3", ()), FakeIndex("1", ())
        cache.put("a", a); cache.put("b", b)
        self.assertEqual(cache.stats()["chunks"], 5)
        cache.put("c", c)  # over both bounds: the oldest goes
        self.assertIsNone(cache.get("a"))
        self.assertIs(cache.get("b"), b)
        cache.put("huge", FakeIndex("\n\n".join("x" * 6), ()))  # larger than the whole budget: never stored
        self.assertIsNone(cache.get("huge"))
        now[0] = 11
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.stats()["items"], 0)


class ServiceWithoutTokenTests(unittest.TestCase):
    def test_refuses_to_start_without_a_token(self):
        env = dict(os.environ)
        os.environ.pop("CHUNKLAYA_TOKEN", None); os.environ.pop("CHUNKLAYA_ALLOW_ANON", None)
        try:
            with self.assertRaises(RuntimeError):
                create_app(lambda: make_runtime())
            os.environ["CHUNKLAYA_ALLOW_ANON"] = "1"
            create_app(lambda: make_runtime())
        finally:
            os.environ.clear(); os.environ.update(env)

    def test_bm25_query_names_the_labels_not_none(self):
        q = service.bm25_query({"instructions": "Which department?", "criteria": {"billing": None, "technical": "network and hardware"}})
        self.assertIn("billing", q); self.assertIn("network and hardware", q); self.assertNotIn("None", q)


if __name__ == "__main__":
    unittest.main()
