"""OpenRouter Batch API route for the LLM families listed under `batch` in config/models.yaml.

submit   builds exactly the request LLMClient would send, drops those already answered or pending, and posts them in
         chunks to POST /api/v1/batches: batch-level endpoint, base model slug, provider.only, 24 h window; each item
         {custom_id = RawStore hash of the request, body without model or provider}. The POST is sent once, never
         retried (it is not idempotent): an unclear failure leaves a 'posting' entry and stops (see `adopt`, `drop`).
collect  polls GET /api/v1/batches/:id; for every result with status 200 writes the verbatim body to the same RawStore
         file a synchronous call would have written, so run_calls.py then reads the batch answers from cache. Failed,
         expired, cancelled or deleted batches are closed with what they returned; the next `submit` resends the rest.
status   prints each batch's state.
Layout per the OpenRouter batch quickstart: terminal GET returns `results` inline, each item with
`custom_id` and either `response` {status_code, request_id, body} or `error`; `results` is null unless completed;
data kept 30 days. Nothing here parses probabilities."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

from .clients import LLMClient, RawStore, MODELS, PROMPTS, OPENROUTER, _key, _sha, split_system

TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _headers() -> dict:
    return {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}", "Content-Type": "application/json"}


class Unclear(RuntimeError):
    """The submission may or may not have been created server-side."""


class Refused(RuntimeError):
    """A 4xx reply to the POST: nothing was created server-side."""


def _post_once(payload: dict) -> dict:
    try:
        with httpx.Client(timeout=300) as c:
            r = c.post(f"{OPENROUTER}/api/v1/batches", headers=_headers(), json=payload)
    except httpx.TransportError as e:
        raise Unclear(repr(e)[:300])
    if r.status_code >= 500:
        raise Unclear(f"{r.status_code} {r.text[:300]}")
    if r.status_code >= 400:     # refused: nothing was created
        raise Refused(f"batch refused: {r.status_code} {r.text[:500]}")
    try:
        return r.json()
    except ValueError:           # accepted but unreadable: the batch may exist
        raise Unclear(f"{r.status_code} undecodable reply {r.text[:300]}")


def _get(url: str, params: dict | None = None, retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=120) as c:
                r = c.get(url, headers=_headers(), params=params)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(delay); delay *= 2; continue
            if r.status_code in (404, 410):
                return {"status": "gone", "http_status": r.status_code}
            r.raise_for_status()
            return r.json()
        except httpx.TransportError:
            if attempt == retries - 1:
                raise
            time.sleep(delay); delay *= 2
    raise RuntimeError("retries exhausted")


class Manifest:
    """runs/<split>/batches/manifest.json plus one <key>.requests.json per submission (custom_id -> request)."""

    def __init__(self, store: RawStore):
        self.dir = Path(store.run_dir) / "batches"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "manifest.json"
        self.items: list[dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=1), encoding="utf-8")
        for k in range(20):          # Windows: the replace fails while a reader (e.g. --batch status) has it open
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                if k == 19:
                    raise
                time.sleep(0.05 * (k + 1))

    def requests(self, key: str) -> dict:
        return json.loads((self.dir / f"{key}.requests.json").read_text(encoding="utf-8"))

    def pending_ids(self) -> set[str]:
        out: set[str] = set()
        for b in self.items:
            if not b.get("ingested"):
                out |= set(self.requests(b["key"]))
        return out

    def posting(self) -> list[dict]:
        return [b for b in self.items if b.get("status") == "posting"]

    def attempts(self) -> dict[str, int]:
        """Submissions per request (dropped submissions never existed and do not count)."""
        out: dict[str, int] = {}
        for b in self.items:
            if b.get("status") != "dropped":
                for cid in self.requests(b["key"]):
                    out[cid] = out.get(cid, 0) + 1
        return out


def pending_requests(calls, store: RawStore, population: str) -> list[tuple[str, str, dict]]:
    """(family, custom_id, request) for every planned LLM call on the batch route without a stored response."""
    stmt = PROMPTS["statements"][population]
    out, seen = [], set()
    for c in calls:
        fam, _ = split_system(c.model)             # 'gpt@low' (reasoning arm) is batched with its family
        if fam not in MODELS["families"]:
            continue
        cl = LLMClient(c.model, store, population, repeat=c.repeat, variant=c.variant)
        if cl.batch is None:
            continue
        req = cl.request(c.text, stmt)
        cid = _sha(req)
        if cid in seen or store.get(req) is not None:
            continue
        seen.add(cid)
        out.append((fam, cid, req))
    return out


def body_of(req: dict) -> dict:
    """Per-request body: model inherited from the batch, provider is batch-level, cache-only keys dropped."""
    return {k: v for k, v in req.items() if not k.startswith("_") and k not in ("model", "provider")}


def payload_for(fam: str, chunk: list[tuple[str, dict]]) -> dict:
    return {"endpoint": "/v1/chat/completions", "model": MODELS["families"][fam]["slug"],
            "provider": {"only": list(MODELS["batch"]["families"][fam]["provider_only"])},
            "completion_window": "24h",
            "requests": [{"custom_id": cid, "body": body_of(req)} for cid, req in chunk]}


def submit(calls, store: RawStore, population: str, max_requests: int | None = None, dry: bool = False,
           post=None, max_total: int | None = None) -> list[dict]:
    """Submit pending batch-route requests in chunks of max_requests; max_total caps the requests sent in this call
    (rolling submission on a small balance: OpenRouter holds each batch's worst case until it finishes)."""
    post = post or _post_once
    man = Manifest(store)
    if man.posting():
        raise SystemExit("A previous submission may or may not exist on OpenRouter: "
                         + ", ".join(f"{b['key']} ({b['family']}, {b['n']} requests, {time.ctime(b['submitted'])})" for b in man.posting())
                         + ". Check openrouter.ai activity/batches; then  run_calls.py <split> --batch adopt <key> <batch_id>"
                           "  or  --batch drop <key>  if none was created.")
    busy = man.pending_ids()
    tries, cap = man.attempts(), int(MODELS["batch"].get("max_attempts", 3))
    todo = [t for t in pending_requests(calls, store, population) if t[1] not in busy]
    spent = [t for t in todo if tries.get(t[1], 0) >= cap]
    if spent:
        print(f"{len(spent)} requests reached {cap} submissions without a completed response: not sent again "
              f"(retry limit); they count as unusable.", flush=True)
    todo = [t for t in todo if tries.get(t[1], 0) < cap]
    n = max_requests or int(MODELS["batch"].get("max_requests", 1000))
    if max_total is not None:
        todo = todo[:max_total]
    made = []
    for fam in sorted({f for f, _, _ in todo}):
        items = [(cid, req) for f, cid, req in todo if f == fam]
        for i in range(0, len(items), n):
            chunk = items[i:i + n]
            if dry:
                made.append({"family": fam, "n": len(chunk)})
                continue
            key = uuid.uuid4().hex[:12]
            (man.dir / f"{key}.requests.json").write_text(json.dumps({cid: req for cid, req in chunk}), encoding="utf-8")
            entry = {"key": key, "id": None, "family": fam, "n": len(chunk), "submitted": time.time(),
                     "provider_pinned": MODELS["batch"]["families"][fam]["provider_only"], "status": "posting",
                     "ingested": False}
            man.items.append(entry); man.save()           # recorded before the POST: a lost reply cannot orphan it
            try:
                resp = post(payload_for(fam, chunk))
            except Refused:
                man.items.remove(entry); man.save()       # positively refused: nothing exists server-side
                raise
            except Exception as e:                        # anything else after sending may have created a batch
                man.save()
                raise SystemExit(f"Submission {key} unclear ({e!r:.300}). Nothing more is sent. Check openrouter.ai for a "
                                 f"batch of {len(chunk)} {fam} requests created just now; adopt it or drop {key}.")
            bid = resp.get("id")
            if not bid:
                man.save()
                raise SystemExit(f"Submission {key} returned no id: {json.dumps(resp)[:400]}")
            entry.update(id=bid, status=resp.get("status"))
            man.save(); made.append(entry)
            print(f"submitted {bid}: {fam} {len(chunk)} requests ({resp.get('status')})", flush=True)
    return made


def adopt(store: RawStore, key: str, batch_id: str) -> None:
    man = Manifest(store)
    b = next(b for b in man.items if b["key"] == key)
    b.update(id=batch_id, status="adopted"); man.save()


def drop(store: RawStore, key: str) -> None:
    man = Manifest(store)
    b = next(b for b in man.items if b["key"] == key)
    assert b["status"] == "posting" or (b.get("last_error") or "").startswith("not found"), \
        "only an unclear submission or a batch the current key cannot find can be dropped"
    b.update(status="dropped", ingested=True, written=0); man.save()


def _body(item: dict) -> tuple[dict | None, int | None]:
    resp = item.get("response")
    if not isinstance(resp, dict):
        return None, None
    body = resp.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = None
    return (body if isinstance(body, dict) else None), resp.get("status_code")


def collect(store: RawStore, get=None) -> dict:
    """Ingest every finished batch; returns counts. Safe to run repeatedly; one bad batch never blocks the others."""
    get = get or _get
    man = Manifest(store)
    tally = {"written": 0, "failed": 0, "waiting_batches": 0, "errors": 0}
    tries = man.attempts()
    for b in man.items:
        if b.get("ingested") or not b.get("id"):
            continue
        try:
            head = get(f"{OPENROUTER}/api/v1/batches/{b['id']}", None)
            if head.get("status") == "gone":      # not found for this key (another account's key, or deleted): the batch
                tally["errors"] += 1              # stays open, never written off; `drop` releases it once confirmed lost
                b["last_error"] = f"not found ({head.get('http_status')}) for the current key"
                print(f"{b['id']} {b['family']}: {b['last_error']}; left open", flush=True)
                continue
            b["status"], b["request_counts"] = head.get("status"), head.get("request_counts")
            if b["status"] not in TERMINAL:
                tally["waiting_batches"] += 1
                continue
            results = head.get("results") or []
            (man.dir / f"{b['key']}.result.json").write_text(json.dumps(head), encoding="utf-8")
            reqs = man.requests(b["key"])
            ok = 0
            for r in results:
                cid = r.get("custom_id")
                body, code = _body(r)
                if cid not in reqs or body is None or code != 200:
                    continue                              # no completion: resubmitted up to the retry limit
                meta = {"ts": time.time(), "family": b["family"], "route": "batch", "batch_id": b["id"],
                        "provider_pinned": ",".join(b.get("provider_pinned") or []),
                        "provider_reported": body.get("provider"), "model_reported": body.get("model"),
                        "usage": body.get("usage"), "attempt": tries.get(cid, 1)}
                if store.put(reqs[cid], body, meta):      # every completion is stored, usable or not;
                    ok += 1                               # the first stored answer is never replaced
                else:
                    tally["duplicates"] = tally.get("duplicates", 0) + 1
            b["written"], b["ingested"] = ok, True
            tally["written"] += ok
            tally["failed"] += len(reqs) - ok
            print(f"{b['id']} {b['family']}: {b['status']}, {ok}/{len(reqs)} written", flush=True)
        except Exception as e:   # keep going; this batch is retried on the next collect
            tally["errors"] += 1
            b["last_error"] = repr(e)[:300]
            print(f"{b['id']} {b['family']}: collect error {b['last_error']}", flush=True)
        finally:
            man.save()
    return tally


def status(store: RawStore, get=None) -> list[dict]:
    get = get or _get
    man = Manifest(store)
    rows = []
    for b in man.items:
        if b.get("id") and not b.get("ingested"):
            try:
                head = get(f"{OPENROUTER}/api/v1/batches/{b['id']}", None)
                b["status"], b["request_counts"] = head.get("status"), head.get("request_counts")
            except Exception as e:
                b["last_error"] = repr(e)[:300]
        rows.append({k: b.get(k) for k in ("key", "id", "family", "n", "status", "request_counts", "ingested", "written")})
    man.save()
    return rows
