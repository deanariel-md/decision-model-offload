"""Batch route: request identity with the synchronous client, payload shape, submit/collect bookkeeping, unclear
submissions, a failing batch not blocking others, eICU exclusion. No network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import batch as BT
from jevity.clients import LLMClient, FillClient, RawStore, BATCH_MISSING
from jevity.runner import Call

TXT = '{"age": 64, "sex": "fémale"}'
STMT = "This person will die from any cause within ten years of this examination."


def _completion(p):
    return {"id": "gen-1", "model": "openai/gpt-5.6-sol-20260709",
            "choices": [{"message": {"role": "assistant", "content": json.dumps({"p": p})}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 410}}


def _calls():
    return [Call("gpt", 1, "baseline", False, 0, TXT), Call("gpt", 1, "baseline", False, 1, TXT),
            Call("gpt", 1, "baseline", False, 0, TXT, "M1"), Call("claude", 1, "baseline", False, 0, TXT),
            Call("glm", 1, "baseline", False, 0, TXT), Call("jev", 1, "baseline", False, 0, TXT)]


def test_routes(tmp_path):
    store = RawStore(tmp_path)
    g = LLMClient("gpt", store)
    req = g.request(TXT, STMT)
    assert g.batch is not None and not req["model"].endswith(":batch") and req["provider"] == {"only": ["openai"]}
    with pytest.raises(RuntimeError):                       # credentialed eICU never goes to any model
        LLMClient("gpt", store, population="eicu").request(TXT, STMT)
    std = LLMClient("gpt", store, standard=True).request(TXT, STMT)
    assert std["provider"] == {"only": ["openai"], "allow_fallbacks": False}            # own endpoint, no ZDR key
    assert "temperature" not in std and "reasoning" not in std and std["seed"] == 20260922 and std["max_tokens"] == 16000
    assert "seed" not in LLMClient("claude", store, standard=True).request(TXT, STMT)
    assert LLMClient("gpt", store, standard=True).batch is None
    assert FillClient("gpt", store).batch is None
    assert LLMClient("glm", store).batch is None
    r = g.probability(TXT, STMT)                            # never sent synchronously
    assert r["valid"] is False and r["error"] == BATCH_MISSING
    pl = BT.payload_for("gpt", [(BT._sha(req), req)])
    assert list(pl)[:3] == ["endpoint", "model", "provider"] and pl["provider"] == {"only": ["openai"]}
    body = pl["requests"][0]["body"]
    assert "model" not in body and "provider" not in body and not any(k.startswith("_") for k in body)


def test_submit_collect(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()
    todo = BT.pending_requests(calls, store, "nhanes")
    assert sorted(f for f, _, _ in todo) == ["claude", "gpt", "gpt", "gpt"]   # repeats and M1 are distinct requests
    sent = []

    def post(payload):
        sent.append(payload)
        return {"id": f"b{len(sent)}", "status": "validating"}

    made = BT.submit(calls, store, "nhanes", post=post)
    assert len(made) == 2 and sum(m["n"] for m in made) == 4
    assert BT.submit(calls, store, "nhanes", post=post) == []           # pending requests are not sent twice

    def get(url, params):
        bid = url.rsplit("/", 1)[-1]
        if bid == "b1":
            raise RuntimeError("boom")                                  # one failing batch must not block the other
        payload = sent[int(bid[1:]) - 1]
        return {"id": bid, "status": "completed", "request_counts": {"completed": len(payload["requests"])},
                "results": [{"id": "r", "custom_id": r["custom_id"],
                             "response": {"status_code": 200, "request_id": "x", "body": _completion(0.25)}}
                            for r in payload["requests"]]}

    t = BT.collect(store, get=get)
    assert t["errors"] == 1 and t["written"] == len(sent[1]["requests"])
    fam = "claude" if sent[1]["model"].startswith("anthropic") else "gpt"
    r = LLMClient(fam, store, repeat=0).probability(TXT, STMT)
    assert r["valid"] and abs(r["p"] - 0.25) < 1e-9
    meta = json.loads(Path(r["raw_path"]).read_text())["meta"]
    assert meta["route"] == "batch" and r["provider"] == meta["provider_pinned"]


def test_failed_batch_resubmitted(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()[:1]
    BT.submit(calls, store, "nhanes", post=lambda p: {"id": "b1", "status": "validating"})
    BT.collect(store, get=lambda u, p: {"id": "b1", "status": "expired", "results": None})
    again = BT.submit(calls, store, "nhanes", post=lambda p: {"id": "b2", "status": "validating"})
    assert len(again) == 1 and again[0]["id"] == "b2"


def test_unclear_submission_stops(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()[:1]

    def post(p):
        raise BT.Unclear("ReadTimeout")

    with pytest.raises(SystemExit):
        BT.submit(calls, store, "nhanes", post=post)
    with pytest.raises(SystemExit):                                     # blocked until adopt or drop
        BT.submit(calls, store, "nhanes", post=lambda p: {"id": "b9"})
    key = BT.Manifest(store).posting()[0]["key"]
    BT.drop(store, key)
    assert BT.submit(calls, store, "nhanes", post=lambda p: {"id": "b9", "status": "validating"})[0]["id"] == "b9"


def test_fallback_to_standard(tmp_path, monkeypatch):
    from jevity import clients as C
    store = RawStore(tmp_path)
    monkeypatch.setitem(C.MODELS["batch"], "fallback_to_standard", ["gpt"])
    sent = []
    monkeypatch.setattr(C, "_post", lambda url, h, body: sent.append(body) or {**_completion(0.4), "provider": "Azure"})
    monkeypatch.setattr(C, "_key", lambda name: "x")
    r = LLMClient("gpt", store).probability(TXT, STMT)        # batch answer missing -> standard route
    assert r["valid"] and abs(r["p"] - 0.4) < 1e-9 and sent and sent[0]["provider"]["only"] == ["openai"]
    assert "zdr" not in sent[0]["provider"]


def test_undecodable_reply_keeps_entry(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()[:1]

    def post(p):
        raise ValueError("Expecting value: line 1 column 1")   # e.g. JSONDecodeError after a 200

    with pytest.raises(SystemExit):
        BT.submit(calls, store, "nhanes", post=post)
    assert len(BT.Manifest(store).posting()) == 1              # evidence of the POST is kept

    with pytest.raises(BT.Refused):
        BT.drop(store, BT.Manifest(store).posting()[0]["key"])
        BT.submit(calls, store, "nhanes", post=lambda p: (_ for _ in ()).throw(BT.Refused("400")))
    assert BT.Manifest(store).posting() == []                   # a positive refusal leaves nothing behind


def test_empty_completion_stored_and_first_answer_immutable(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()[:1]
    sent = []
    BT.submit(calls, store, "nhanes", post=lambda p: sent.append(p) or {"id": "b1", "status": "validating"})
    cid = sent[0]["requests"][0]["custom_id"]
    empty = {"id": "gen-2", "choices": [], "usage": {}}
    BT.collect(store, get=lambda u, p: {"id": "b1", "status": "completed", "results": [
        {"custom_id": cid, "response": {"status_code": 200, "body": empty}}]})
    r = LLMClient("gpt", store).probability(TXT, STMT)
    assert r["valid"] is False and r["parse"] == "empty" and r["route"] == "batch"
    req = LLMClient("gpt", store).request(TXT, STMT)
    assert store.put(req, _completion(0.3), {}) is False        # a later answer never replaces the first
    assert LLMClient("gpt", store).probability(TXT, STMT)["valid"] is False


def test_retry_limit(tmp_path):
    store = RawStore(tmp_path)
    calls = _calls()[:1]
    for i in range(3):
        assert len(BT.submit(calls, store, "nhanes", post=lambda p: {"id": f"b{i}", "status": "validating"})) == 1
        BT.collect(store, get=lambda u, p: {"id": u.rsplit("/", 1)[-1], "status": "expired", "results": None})
    assert BT.submit(calls, store, "nhanes", post=lambda p: {"id": "b9", "status": "validating"}) == []


def test_single_flight(tmp_path, monkeypatch):
    import threading
    import time as _t
    from jevity import clients as C
    store = RawStore(tmp_path)
    n = []

    def slow(url, h, body):
        n.append(1); _t.sleep(0.2)
        return _completion(0.2)

    monkeypatch.setattr(C, "_post", slow)
    monkeypatch.setattr(C, "_key", lambda name: "x")
    ts = [threading.Thread(target=lambda: LLMClient("glm", store).probability(TXT, STMT)) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(n) == 1


def test_not_found_batch_left_open(tmp_path):
    """A batch the current key cannot find (another account, or deleted) is never written off as answered-nothing:
    it stays open, its requests stay reserved, and only an explicit drop releases them."""
    store = RawStore(tmp_path)
    man = BT.Manifest(store)
    man.items = [{"key": "k1", "id": "batch-x", "family": "gpt", "status": "in_progress", "submitted": 0}]
    (man.dir / "k1.requests.json").write_text(json.dumps({"c1": {}, "c2": {}}))
    man.save()
    t = BT.collect(store, get=lambda url, p: {"status": "gone", "http_status": 404})
    assert t["errors"] == 1 and t["failed"] == 0
    assert not BT.Manifest(store).items[0].get("ingested")
    assert BT.Manifest(store).pending_ids() == {"c1", "c2"}
    BT.drop(store, "k1")
    assert BT.Manifest(store).pending_ids() == set()
