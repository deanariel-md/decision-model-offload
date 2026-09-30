"""The every-form runner against a mock transport (no network), on synthetic reports: stores every reply once, retries
429 / 5xx / in-reply errors, hands a refused key's requests to the queue's other key, stops a system whose reply names
another model, and resumes without re-sending a stored request."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "llm_forms"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import run as R  # noqa: E402
import synthetic_llm_forms as SL  # noqa: E402
from jevity import llm_forms as LF  # noqa: E402

KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3", "HF_TOKEN", "HF_TOKEN_2")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tree")
    frames = SL.synthetic_frames()
    pool = pd.concat([pd.DataFrame({"report_id": frames[s].report_id, "set": s}) for s in LF.SETS], ignore_index=True)
    SL.write_tree(root, frames, pool)
    yield root, pool
    SL.clear_caches()


def reply(model, content='{"stage": "I", "probabilities": {"I": 1}}'):
    return {"id": "x", "model": model, "provider": "P", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0001}}


@pytest.fixture
def setup(tree, tmp_path, monkeypatch):
    SL.use(monkeypatch, *tree)
    for name in ("STORE", "LEDGER", "STATE"):
        monkeypatch.setattr(R, name, tmp_path / name.lower())
    ids = sorted(LF.pool().report_id)[:6]
    cov = pd.DataFrame([{"system": s, "form": f, "item_id": i, "status": "gap"}
                        for s in ("gpt", "glm", "muse", "medgemma", "gemma") for f in ("names", "documented") for i in ids])
    p = tmp_path / "coverage.parquet"
    cov.to_parquet(p)
    monkeypatch.setattr(R, "COVERAGE", p)
    monkeypatch.setattr(R, "dotenv_values", lambda _: {k: f"key-{k}" for k in KEYS})
    monkeypatch.setattr(R, "smoke_ids", lambda: ids[:2])
    yield tmp_path, ids
    SL.clear_caches()


def patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def make(*a, **k):
        k.pop("limits", None)
        return real(transport=httpx.MockTransport(handler), timeout=k.get("timeout"))
    monkeypatch.setattr(R.httpx, "AsyncClient", make)


def go(group, mode="full", systems=None, keys=None):
    a = type("A", (), {"group": group, "mode": mode, "systems": systems, "keys": keys})()
    asyncio.run(R.main_async(a))


def _fast_sleep(real):
    async def s(d, *a, **k):
        return await real(min(d, 0.01), *a, **k)
    return s


def test_openrouter_group_retries_and_stores_once(setup, monkeypatch):
    tmp, ids = setup
    seen = {"n": 0, "bodies": []}

    def handler(req: httpx.Request):
        if req.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900}})
        if req.url.path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"total_credits": 1910, "total_usage": 1}})
        seen["n"] += 1
        body = json.loads(req.content)
        assert not any(k.startswith("_") for k in body)
        assert body["provider"]["allow_fallbacks"] is False
        seen["bodies"].append(body)
        k = seen["n"]
        if k % 7 == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"})
        if k % 7 == 2:
            return httpx.Response(503)
        if k % 7 == 3:
            return httpx.Response(200, json={"error": {"message": "upstream"}})
        return httpx.Response(200, json=reply(R.EXPECTED_MODEL[{"openai/gpt-5.6-sol": "gpt"}.get(body["model"], "glm")]))
    patch_transport(monkeypatch, handler)
    monkeypatch.setattr(R, "MAX_ATTEMPTS", 20)
    monkeypatch.setattr(R.asyncio, "sleep", _fast_sleep(R.asyncio.sleep))
    go("openrouter")
    stored = list((tmp / "store").glob("*.json"))
    assert len(stored) == 2 * 2 * len(ids)                 # gpt and glm, two forms, six reports: each stored once
    led = [json.loads(l) for l in (tmp / "ledger" / "openrouter.jsonl").read_text().splitlines()]
    assert len(led) == len(stored) and all(r["stored"] for r in led)
    n_before = seen["n"]
    go("openrouter")                                        # resume: nothing re-sent
    assert seen["n"] == n_before


def test_wrong_model_stops_that_system_only(setup, monkeypatch):
    tmp, ids = setup

    def handler(req):
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900, "total_credits": 1910, "total_usage": 1}})
        body = json.loads(req.content)
        m = "openai/gpt-5.6-sol-2099" if body["model"] == "openai/gpt-5.6-sol" else "z-ai/glm-5.3"
        return httpx.Response(200, json=reply(m))
    patch_transport(monkeypatch, handler)
    go("openrouter")
    summ = json.loads((tmp / "state" / "openrouter_full_summary.json").read_text())
    assert "gpt" in summ["stopped"] and "glm" not in summ["stopped"]
    gpt_n = sum(1 for l in (tmp / "ledger" / "openrouter.jsonl").read_text().splitlines() if '"gpt"' in l)
    assert gpt_n <= 32                                      # at most the requests already in flight when it stopped
    assert sum(1 for l in (tmp / "ledger" / "openrouter.jsonl").read_text().splitlines() if '"glm"' in l) == 2 * len(ids)


def test_refused_key_hands_over_to_the_other(setup, monkeypatch):
    tmp, ids = setup
    used = {}

    def handler(req):
        auth = req.headers["authorization"]
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 80, "total_credits": 100, "total_usage": 1}})
        used[auth] = used.get(auth, 0) + 1
        if auth.endswith("OPENROUTER_API_KEY_3"):
            return httpx.Response(403, json={"error": "refused"})
        return httpx.Response(200, json=reply("meta/muse-spark-1.1"))
    patch_transport(monkeypatch, handler)
    go("muse", keys=["OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"])
    assert len(list((tmp / "store").glob("*.json"))) == 2 * len(ids)
    assert "Bearer key-OPENROUTER_API_KEY" not in used      # only the keys named
    assert used.get("Bearer key-OPENROUTER_API_KEY_2")


def test_default_key_is_the_groups(setup, monkeypatch):
    tmp, ids = setup
    used = set()

    def handler(req):
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900, "total_credits": 1910, "total_usage": 1}})
        used.add(req.headers["authorization"])
        return httpx.Response(200, json=reply("meta/muse-spark-1.1"))
    patch_transport(monkeypatch, handler)
    go("muse")
    assert used == {"Bearer key-OPENROUTER_API_KEY"}
    assert len(list((tmp / "store").glob("*.json"))) == 2 * len(ids)


def test_hf_pool_uses_both_tokens_and_caps_in_flight(setup, monkeypatch):
    tmp, ids = setup
    by_tok, inflight, peak = {}, {}, {}

    async def handler(req):
        tok = req.headers["authorization"]
        by_tok[tok] = by_tok.get(tok, 0) + 1
        inflight[tok] = inflight.get(tok, 0) + 1
        peak[tok] = max(peak.get(tok, 0), inflight[tok])
        await asyncio.sleep(0.02)
        inflight[tok] -= 1
        body = json.loads(req.content)
        assert "provider" not in body
        m = "google/medgemma-27b-text-it" if "medgemma" in body["model"] else "google/gemma-3-27b-it"
        return httpx.Response(200, json=reply(m))
    patch_transport(monkeypatch, handler)
    go("hf", keys=["HF_TOKEN", "HF_TOKEN_2"])
    assert len(list((tmp / "store").glob("*.json"))) == 2 * 2 * len(ids)
    assert set(by_tok) == {"Bearer key-HF_TOKEN", "Bearer key-HF_TOKEN_2"}
    assert max(peak.values()) <= 5


def test_smoke_selects_the_smoke_reports_only(setup, monkeypatch):
    tmp, ids = setup

    def handler(req):
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900, "total_credits": 1910, "total_usage": 1}})
        body = json.loads(req.content)
        return httpx.Response(200, json=reply({"openai/gpt-5.6-sol": "openai/gpt-5.6-sol"}.get(body["model"], "z-ai/glm-5.3")))
    patch_transport(monkeypatch, handler)
    go("openrouter", mode="smoke")
    items = {json.loads(p.read_text())["request"]["_item"] for p in (tmp / "store").glob("*.json")}
    assert items == set(ids[:2])


def failed_reply(model):
    r = reply(model, content='{"stage": "II')
    r["choices"][0].update(finish_reason="error", error={"code": 502, "message": "JSON error injected into SSE stream"})
    r["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}
    return r


def test_in_reply_provider_error_is_retried_not_stored(setup, monkeypatch):
    tmp, ids = setup
    seen = {}

    def handler(req):
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900, "total_credits": 1910, "total_usage": 1}})
        body = json.loads(req.content)
        m = "openai/gpt-5.6-sol" if body["model"] == "openai/gpt-5.6-sol" else "z-ai/glm-5.3"
        k = json.dumps(body["messages"], sort_keys=True) + body["model"]
        seen[k] = seen.get(k, 0) + 1
        return httpx.Response(200, json=failed_reply(m) if seen[k] == 1 else reply(m))
    patch_transport(monkeypatch, handler)
    monkeypatch.setattr(R.asyncio, "sleep", _fast_sleep(R.asyncio.sleep))
    go("openrouter")
    stored = [json.loads(p.read_text()) for p in (tmp / "store").glob("*.json")]
    assert len(stored) == 2 * 2 * len(ids) and not any(R.transport_failed(s["response"]) for s in stored)
    assert all(v == 2 for v in seen.values())


def test_transport_retry_mode_resends_failed_cells(setup, monkeypatch):
    tmp, ids = setup
    store = R.RawStore(tmp / "store")
    bad = LF.build("glm", "names", ids[0])
    store.put(bad, failed_reply("z-ai/glm-5.3"), {"ts": 0})
    for s in ("gpt", "glm"):                            # every other cell already has a complete reply
        for f in ("names", "documented"):
            for i in ids:
                r = LF.build(s, f, i)
                if store.get(r) is None:
                    store.put(r, reply(R.EXPECTED_MODEL[s]), {"ts": 0})
    sent = []

    def handler(req):
        if req.url.path.endswith(("/key", "/credits")):
            return httpx.Response(200, json={"data": {"usage": 1.0, "limit": 1900, "total_credits": 1910, "total_usage": 1}})
        body = json.loads(req.content)
        assert R.TRANSPORT_RETRY_KEY not in body                     # the cache key is never sent
        sent.append(body)
        return httpx.Response(200, json=reply("z-ai/glm-5.3"))
    patch_transport(monkeypatch, handler)
    go("openrouter", mode="transport-retry")
    assert len(sent) == 1 and sent[0] == R.body_of(bad)             # the same request, content unchanged
    assert R.transport_failed(store.get(bad)["response"])            # the failed reply is kept, never overwritten
    got, k, failed = R.answered(store, bad)
    assert k == 1 and got is not None and failed is not None
    go("openrouter", mode="transport-retry")                        # nothing left to resend
    assert len(sent) == 1

