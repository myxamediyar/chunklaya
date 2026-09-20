"""Smoke tests against the real model (~1 min on MPS). Run from repo root: .venv/bin/python tests/test_smoke.py"""
import json, time
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import laya
from chunklaya import ChunkLaya, predict_many, chunk_text

t0 = time.time()
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
print(f"loaded on {agent.device} in {time.time()-t0:.0f}s")
tok = agent.tok
D = json.load(open("eval/data/ag_news.json"))
arts = [r["text"] for r in D["rows"]]

Q_CHOICE = {"type": "choice", "instructions": "Which news category does this text belong to?",
            "criteria": {"World": "international and political news", "Sports": "sports and athletics",
                         "Business": "business, markets, and finance", "Sci/Tech": "science and technology"}}
Q_NOUL = {"type": "noul", "instructions": "Does the text contain a news item about sports or athletics?"}
Q_SCORE = {"type": "score", "instructions": "How formal is the writing?", "criteria": ["casual", "neutral", "formal"]}

# 1. predict_many == agent.predict, all three types, to 1e-6
state = arts[0]
ref = agent.predict(state, {"c": Q_CHOICE, "n": Q_NOUL, "s": Q_SCORE})["answers"]
got = predict_many(agent, [(state, Q_CHOICE), (state, Q_NOUL), (state, Q_SCORE)])
for r, g, key in zip([ref["c"], ref["n"], ref["s"]], got, ["probabilities", "noul", "probabilities"]):
    a, b = r[key], g[key]
    if isinstance(a, dict):
        assert all(abs(a[k] - round(b[k], 4)) < 1e-4 for k in a), (a, b)
    else:
        assert abs(a - round(b, 4)) < 1e-4, (a, b)
    assert abs(r["confidence"] - round(g["confidence"], 4)) < 1e-4
print("1. predict_many matches agent.predict for choice / noul / score")

# 2. batching does not change results (row order / padding independence)
many = [(a, Q_CHOICE) for a in arts[:12]]
single = [predict_many(agent, [p])[0]["p"] for p in many]
batched = [r["p"] for r in predict_many(agent, many, batch_size=12)]
md = max(float(np.abs(s - b).max()) for s, b in zip(single, batched))
assert md < 2e-3, md   # fp32 on MPS with different padding lengths: tiny numeric drift is expected
print(f"2. batch of 12 vs singles: max |Δp| = {md:.2e}")

# 3. chunker: exact substrings, full coverage, overlap invariant
text = "\n\n".join(arts[:70])
for ct, st in [(750, 375), (750, 750), (512, 256), (1024, 512)]:
    ch = chunk_text(tok, text, ct, st)
    assert all(c.text == text[c.char_start:c.char_end] for c in ch)
    assert ch[0].tok_start == 0 and ch[-1].tok_end == len(tok(text, add_special_tokens=False)["input_ids"])
    assert all(ch[i+1].tok_start - ch[i].tok_start == st for i in range(len(ch) - 2)), "stride"
    assert all(c.n_tokens <= ct for c in ch) and all(c.n_tokens >= ct // 4 for c in ch)
    print(f"3. chunk_text({ct},{st}) → {len(ch)} chunks, sizes {[c.n_tokens for c in ch]}")

# 4. harness passthrough on short input == agent.predict
cj = ChunkLaya(agent, 750, 375)
r = cj.ask(arts[3], {"c": Q_CHOICE})
assert r["n_chunks"] == 1 and r["answers"]["c"]["agg"] == "passthrough"
assert r["answers"]["c"]["choice"] == agent.predict(arts[3], {"c": Q_CHOICE})["answers"]["c"]["choice"]
print("4. passthrough on 1-chunk input matches agent.predict")

# 5. harness on a long input, every mode, shapes sane
t = time.time()
r = cj.ask(text, {"n": Q_NOUL, "c": Q_CHOICE, "s": Q_SCORE})
print(f"5. long input: {r['n_chunks']} chunks, {r['usage']['input_tokens']} tokens in, {time.time()-t:.1f}s")
n, c, s = r["answers"]["n"], r["answers"]["c"], r["answers"]["s"]
assert 0 <= n["noul"] <= 1 and n["agg"] == "max" and len(n["chunks"]) == r["n_chunks"] and n["chunks"][0]["gate"] is None
assert abs(sum(c["probabilities"].values()) - 1) < 1e-6 and c["agg"] == "mixture" and c["chunks"][0]["gate"] is not None
assert 0 <= s["score"] <= 2 and s["agg"] == "mixture"
print(f"   noul={n['noul']:.3f} (max over chunks {[round(x['p'][1],2) for x in n['chunks']]})")
print(f"   choice={c['choice']} conf={c['confidence']:.3f} gates={[round(x['gate'],2) for x in c['chunks']]}")
r2 = cj.ask(text, {"c": Q_CHOICE}, agg={"c": "loglinear"}); r3 = cj.ask(text, {"c": Q_CHOICE}, agg={"c": "stack"})
print(f"   loglinear={r2['answers']['c']['choice']}  stack={r3['answers']['c']['choice']}")

# 6. prefilter: BM25 hands Laya only the chunk that matters, and Laya verifies it
from chunklaya import bm25_rank
assert bm25_rank(["the cat sat", "a dog barked", "cat and cat again"], "cat")[:2] == [2, 0]
needle = "The access code for the teal harbor locker is 4417."
paras = arts[:40] + [needle] + arts[40:70]
text2 = "\n\n".join(paras)
def q(code):
    return {"type": "noul", "instructions": f"Does the text state that the access code for the teal harbor locker is {code}?",
            "criteria": {"true": f"The text says the access code for the teal harbor locker is exactly {code}.",
                         "false": f"The text gives a different code for the teal harbor locker, or none at all."}}
def det(code):
    return {"type": "choice", "instructions": f"Does this passage state that the access code for the teal harbor locker is {code}?",
            "criteria": {"yes": f"It says the code for the teal harbor locker is exactly {code}.",
                         "no": f"It gives a code other than {code}, or no code."}}
pf = ChunkLaya(agent, 750, mode="paragraphs", prefilter="bm25", top_k=1)
for label, kw in (("noul", {}), ("choice detector", {"detectors": {"q": (det("4417"), "yes")}})):
    hit = pf.ask(text2, {"q": q("4417")}, **kw)
    kw_miss = {"detectors": {"q": (det("9911"), "yes")}} if kw else {}
    miss = pf.ask(text2, {"q": q("9911")}, **kw_miss)
    for r in (hit, miss):
        a = r["answers"]["q"]
        assert r["prefilter"] == "bm25" and r["n_chunks"] > 1 and a["n_scored"] == 1 and len(a["chunks"]) == 1
        assert a["agg"] == "max", a["agg"]     # not passthrough: one passage was selected out of many
        assert needle in pf.chunks(text2)[a["chunks"][0]["index"]].text, "bm25 did not pick the needle chunk"
    h, m_ = hit["answers"]["q"]["noul"], miss["answers"]["q"]["noul"]
    # the noul alone orders the two but is a soft digit verifier (README: P(yes|mismatch) ~0.86);
    # the choice detector is the recipe, and separates them cleanly
    assert h > m_ if not kw else h > 0.5 > m_, (label, h, m_)
    print(f"6. prefilter [{label}]: {hit['n_chunks']} chunks -> 1 scored; match={h:.3f} mismatch={m_:.3f}")
full = ChunkLaya(agent, 750, mode="paragraphs").ask(text2, {"q": q("4417")})
assert full["answers"]["q"]["n_scored"] == full["n_chunks"] and full["prefilter"] is None
print("   without prefilter every chunk is scored, as before")
print("all smoke tests passed")
