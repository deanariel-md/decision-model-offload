"""Every LLM in every staging form: one asynchronous runner for every LLM x form x report still missing.

Requests come only from jevity.llm_forms.build (built from Jev's stored requests, content asserted per item); the gaps
come from results/arm3_llm_forms/coverage.parquet (status 'gap', scripts/llm_forms/audit.py). Every response is written
to the raw store (runs/arm3_llm_forms, clients.RawStore: first write wins, never overwritten) as it arrives, keyed by
the request; a restart reads the store and never re-sends a stored request.

Transport: the real-time endpoint for every family (OpenRouter chat completions with the pinned provider and
allow_fallbacks false; the Hugging Face router for MedGemma and Gemma), asyncio + httpx, read timeout 600 s.
Concurrency per provider, adaptive: halved on HTTP 429 (never below 1), raised by one after a full window of successes
(never above its maximum); every 429 pauses that provider with exponential backoff and jitter (Retry-After when given).
Retries only for transport errors, HTTP 429 and 500/502/503/529, a 200 reply that carries an error object and no
choices (no model output), and a 200 reply whose choice carries an error object (an in-reply provider error: cut-off
text, no model answer); 8 attempts, 429s count only after 20. Such replies are never stored. Any other HTTP error is
logged and not stored (the request stays a gap for the next pass).

Stops (logged; the rest continues): a reply naming a model other than EXPECTED_MODEL for that system stops that system;
with --max-usd, the new calls' cost (all processes, from the shared ledger) stops every process; with --key-margin-usd
or --account-margin-usd, a key's pool stops near the key's own limit or when its account runs low (OpenRouter keys).

Keys are read from .env (never printed). Groups, one process each: openrouter (gpt, claude, gemini, glm, gpt_free,
claude_free, gemini_free), muse, hf (medgemma and gemma). --keys names the .env variables of the Muse or Hugging Face
queue, one worker pool per key (default: the group's key); each Hugging Face token is held at 5 requests in flight and
each Muse key at 20 requests a minute.

Mode transport-retry resends, once, a cell whose stored replies all carry an in-reply provider error, as the same
request with the cache key _transport_retry (never sent); collect.py analyses the first complete reply (answered()).

  python scripts/llm_forms/run.py --group openrouter --mode smoke|full|dry [--max-usd X]
  python scripts/llm_forms/run.py --group hf --mode full [--keys HF_TOKEN OTHER_HF_TOKEN]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import llm_forms as LF  # noqa: E402
from jevity.clients import MODELS, RawStore, _sha  # noqa: E402

ENV = ROOT / ".env"
STORE = ROOT / "runs" / "arm3_llm_forms"
LEDGER = ROOT / "runs" / "arm3_llm_forms_ledger"
STATE = ROOT / "runs" / "arm3_llm_forms_state"
COVERAGE = ROOT / "results" / "arm3_llm_forms" / "coverage.parquet"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

BUDGET_USD: float | None = None           # new calls' total (list cost or reported cost, the larger); None: no stop
KEY_MARGIN_USD: float | None = None       # stop a key's pool this far short of the key's own limit; None: no stop
ACCOUNT_MARGIN_USD: float | None = None   # stop a key's pool when its account has less than this left; None: no stop
SMOKE_N = 5
SMOKE_SEED = 20260929

GROUPS = {
    "openrouter": {"systems": ["gpt", "claude", "gemini", "glm", "gpt_free", "claude_free", "gemini_free"],
                   "key": "OPENROUTER_API_KEY", "url": OPENROUTER_URL},
    "muse": {"systems": ["muse"], "key": "OPENROUTER_API_KEY", "url": OPENROUTER_URL},
    "hf": {"systems": ["medgemma", "gemma"], "key": "HF_TOKEN", "url": HF_URL},
}
# Queues with one worker pool per key (--keys): start, maximum in flight, requests per minute (None: no rate cap).
PER_KEY = {"hf": (4, 5, None), "meta": (3, 5, 20)}
PROVIDER_OF = {"gpt": "openai", "gpt_free": "openai", "claude": "anthropic", "claude_free": "anthropic",
               "gemini": "google", "gemini_free": "google", "glm": "zai", "muse": "meta",
               "medgemma": "hf", "gemma": "hf"}
# start, maximum in flight; rate: requests per minute (None: no rate cap)
LIMITS = {"openai": (16, 32, None), "anthropic": (16, 32, None), "google": (16, 32, None), "zai": (100, 128, None)}
# The main runs' model for each system (results/arm3/calls.parquet model_reported, every row)
EXPECTED_MODEL = {"gpt": "openai/gpt-5.6-sol", "claude": "anthropic/claude-opus-5.5", "gemini": "google/gemini-3.8-flash",
                  "muse": "meta/muse-spark-1.1", "glm": "z-ai/glm-5.3", "gpt_free": "openai/gpt-5.6-luna",
                  "claude_free": "anthropic/claude-sonnet-5", "gemini_free": "google/gemini-3.5-flash-lite",
                  "medgemma": "google/medgemma-27b-text-it", "gemma": "google/gemma-3-27b-it"}
RETRY_STATUS = {500, 502, 503, 529}
MAX_ATTEMPTS = 8
MAX_429_FREE = 20
# A 200 whose choice carries an error object (OpenRouter's in-reply provider error, e.g. 502 provider_unavailable or
# 429 rate-limited; cut-off text, no tokens billed) is a transport error: retried in the loop, never stored. A stored
# reply of this kind is resent (mode transport-retry) as the same request with the cache key `_transport_retry` (not
# sent to the API), and the first complete reply is the one analysed (`answered`).
TRANSPORT_RETRY_KEY = "_transport_retry"
MAX_TRANSPORT_RETRIES = 3


def transport_failed(resp: dict | None) -> bool:
    """True for a stored reply that holds no complete model answer: its choice carries an error object."""
    if not isinstance(resp, dict):
        return False
    ch = (resp.get("choices") or [{}])[0]
    return bool(ch.get("error")) or ch.get("finish_reason") == "error"


def retry_req(req: dict, k: int) -> dict:
    return req if k == 0 else {**req, TRANSPORT_RETRY_KEY: k}


def answered(store: RawStore, req: dict) -> tuple[dict | None, int, dict | None]:
    """(first complete stored reply or None, number of the next try to send, last transport-failed reply or None)."""
    failed = None
    for k in range(MAX_TRANSPORT_RETRIES + 1):
        c = store.get(retry_req(req, k))
        if c is None:
            return None, k, failed
        if not transport_failed(c.get("response")):
            return c, k, failed
        failed = c
    return None, MAX_TRANSPORT_RETRIES + 1, failed


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def price(system: str) -> tuple[float, float]:
    f = MODELS["families"][system]
    return float(f["price_in"]), float(f["price_out"])


def budget_price(system: str) -> tuple[float, float]:
    """For the budget stop only: GPT-5.6 Sol at US$5 in and US$30 out per million tokens, the others at the config
    price."""
    return (5.0, 30.0) if system == "gpt" else price(system)


def list_usd(system: str, usage: dict | None, guard: bool = False) -> float:
    u = usage or {}
    tin = float(u.get("prompt_tokens", u.get("input_tokens")) or 0)
    tout = float(u.get("completion_tokens", u.get("output_tokens")) or 0)
    pi, po = budget_price(system) if guard else price(system)
    return (tin * pi + tout * po) / 1e6


# ------------------------------------------------------------------------------------------------ plan
def smoke_ids() -> list[str]:
    ids = sorted(LF.pool().report_id)
    return [ids[i] for i in np.random.default_rng(SMOKE_SEED).permutation(len(ids))[:SMOKE_N]]


def report_order() -> dict[str, int]:
    ids = sorted(LF.pool().report_id)
    return {ids[i]: k for k, i in enumerate(np.random.default_rng(SMOKE_SEED + 1).permutation(len(ids)))}


def gaps(systems: list[str], mode: str) -> list[tuple[str, str, str]]:
    cov = pd.read_parquet(COVERAGE)
    g = cov[(cov.status == "gap") & cov.system.isin(systems)]
    if mode == "smoke":
        g = g[g.item_id.isin(smoke_ids())]
    order = report_order()
    fo = {f: i for i, f in enumerate(LF.QUEUE_ORDER)}
    cells = sorted(zip(g.system, g.form, g.item_id), key=lambda c: (fo[c[1]], order[c[2]], LF.SYSTEMS.index(c[0])))
    return cells


# ------------------------------------------------------------------------------------------------ limits
class Adaptive:
    """In-flight cap per provider: halved on 429, +1 after `limit` successes in a row; a 429 pauses the provider."""

    def __init__(self, name: str, start: int, maximum: int, per_min: int | None):
        self.name, self.limit, self.max, self.per_min = name, start, maximum, per_min
        self.inflight, self.ok_run, self.pause_until, self.backoff, self.hits = 0, 0, 0.0, 5.0, 0
        self.starts: list[float] = []
        self.cond = asyncio.Condition()

    async def acquire(self):
        async with self.cond:
            while True:
                t = time.time()
                if self.per_min:
                    self.starts = [s for s in self.starts if t - s < 60.0]
                wait = max(self.pause_until - t, 0.0)
                if self.per_min and len(self.starts) >= self.per_min:
                    wait = max(wait, 60.0 - (t - self.starts[0]) + 0.05)
                if wait <= 0 and self.inflight < self.limit:
                    self.inflight += 1
                    if self.per_min:
                        self.starts.append(t)
                    return
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout=max(min(wait, 5.0), 0.05) if wait > 0 else 5.0)
                except asyncio.TimeoutError:
                    pass

    async def release(self, ok: bool):
        async with self.cond:
            self.inflight -= 1
            if ok:
                self.ok_run += 1
                self.backoff = max(5.0, self.backoff / 2)
                if self.ok_run >= self.limit and self.limit < self.max:
                    self.limit += 1
                    self.ok_run = 0
            self.cond.notify_all()

    async def rate_limited(self, retry_after: float | None):
        async with self.cond:
            self.hits += 1
            self.limit = max(1, self.limit // 2)
            self.ok_run = 0
            d = retry_after if retry_after and retry_after > 0 else self.backoff * (1 + random.random())
            self.backoff = min(self.backoff * 2, 120.0)
            self.pause_until = max(self.pause_until, time.time() + d)
            self.cond.notify_all()
            return d


# ------------------------------------------------------------------------------------------------ run state
@dataclass
class Run:
    group: str
    mode: str
    key: str
    url: str
    systems: list[str]
    store: RawStore
    run_id: str
    stopped: dict[str, str] = field(default_factory=dict)       # system -> reason (hard stop)
    all_stop: str | None = None
    counts: dict = field(default_factory=dict)
    inflight_usd: float = 0.0
    errors: list = field(default_factory=list)
    guard_keys: dict = field(default_factory=dict)      # .env name -> key, OpenRouter keys checked every minute
    disabled: set = field(default_factory=set)           # limiter (pool) names switched off: key refused or account low
    pool_key: dict = field(default_factory=dict)         # limiter name -> .env key name

    def log(self, msg: str):
        line = f"{now()} [{self.group}] {msg}"
        print(line, flush=True)
        with open(STATE / f"{self.group}.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def ledger(self, rec: dict):
        with open(LEDGER / f"{self.group}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def spent_all() -> float:
    tot = 0.0
    for p in LEDGER.glob("*.jsonl"):
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            tot += max(float(r.get("usd_guard") or 0), float(r.get("usd_reported") or 0))
    return tot


async def account_balance(client: httpx.AsyncClient, key: str) -> float | None:
    try:
        r = await client.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        d = r.json().get("data") or {}
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:
        return None


async def key_usage(client: httpx.AsyncClient, key: str) -> tuple[float | None, float | None]:
    try:
        r = await client.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        d = r.json().get("data") or {}
        return float(d.get("usage")), (float(d["limit"]) if d.get("limit") is not None else None)
    except Exception:
        return None, None


# ------------------------------------------------------------------------------------------------ one call
def body_of(req: dict) -> dict:
    return {k: v for k, v in req.items() if not k.startswith("_")}


async def call(run: Run, client: httpx.AsyncClient, lim: Adaptive, key: str, url: str, system: str, form: str,
               item: str, req: dict, est_usd: float):
    if run.store.get(req) is not None:
        run.counts["already"] = run.counts.get("already", 0) + 1
        return
    attempt, n429, delay = 0, 0, 2.0
    body = body_of(req)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    while attempt < MAX_ATTEMPTS:
        if run.all_stop or system in run.stopped:
            return
        await lim.acquire()
        ok = False
        run.inflight_usd += est_usd
        try:
            t0 = time.time()
            try:
                r = await client.post(url, headers=headers, json=body)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                attempt += 1
                run.log(f"transport error {type(e).__name__} {system}/{form}/{item} attempt {attempt}")
                await asyncio.sleep(delay * (1 + random.random())); delay = min(delay * 2, 120)
                continue
            lat = time.time() - t0
            if r.status_code == 429:
                n429 += 1
                if n429 > MAX_429_FREE:
                    attempt += 1
                ra = None
                try:
                    ra = float(r.headers.get("retry-after"))
                except (TypeError, ValueError):
                    pass
                d = await lim.rate_limited(ra)
                run.counts["429"] = run.counts.get("429", 0) + 1
                if lim.hits % 10 == 1:
                    run.log(f"429 on {lim.name}: limit now {lim.limit}, pause {d:.0f} s")
                continue
            if r.status_code in RETRY_STATUS:
                attempt += 1
                run.log(f"HTTP {r.status_code} {system}/{form}/{item} attempt {attempt}")
                await asyncio.sleep(delay * (1 + random.random())); delay = min(delay * 2, 120)
                continue
            if r.status_code != 200:
                txt = r.text[:300].replace("\n", " ")
                run.log(f"HTTP {r.status_code} {system}/{form}/{item}: not stored, not retried: {txt}")
                run.errors.append({"system": system, "form": form, "item": item, "status": r.status_code, "text": txt})
                if r.status_code in (401, 402, 403):
                    run.disabled.add(lim.name)
                    run.log(f"key refused ({r.status_code}) on pool {lim.name}: that pool stops; the request goes back "
                            f"to the queue for the group's other keys")
                    return "requeue"
                return
            try:
                resp = r.json()
            except ValueError:
                attempt += 1
                run.log(f"non-JSON 200 {system}/{form}/{item} attempt {attempt}")
                await asyncio.sleep(delay); delay = min(delay * 2, 120)
                continue
            if isinstance(resp, dict) and resp.get("error") and not resp.get("choices"):
                attempt += 1
                run.log(f"200 with an error object, no choices {system}/{form}/{item} attempt {attempt}: "
                        f"{json.dumps(resp.get('error'))[:200]}")
                await asyncio.sleep(delay * (1 + random.random())); delay = min(delay * 2, 120)
                continue
            if transport_failed(resp):          # in-reply provider error, no model answer
                attempt += 1
                err = (resp.get("choices") or [{}])[0].get("error")
                run.log(f"200 with an in-reply provider error {system}/{form}/{item} attempt {attempt}: "
                        f"{json.dumps(err)[:200]}")
                await asyncio.sleep(delay * (1 + random.random())); delay = min(delay * 2, 120)
                continue
            ok = True
            meta = {"ts": time.time(), "latency_s": lat, "family": system, "route": "standard",
                    "transport": "hf_router" if url == HF_URL else "openrouter", "endpoint": "real-time",
                    "key": lim.name,
                    "provider_reported": resp.get("provider"), "model_reported": resp.get("model"),
                    "usage": resp.get("usage"), "attempt": attempt + 1, "run_id": run.run_id, "group": run.group,
                    "form": form, "item": item}
            first = run.store.put(req, resp, meta)
            u = resp.get("usage") or {}
            rec = {"ts": meta["ts"], "system": system, "form": form, "item": item, "sha": _sha(req), "stored": first,
                   "usd_list": list_usd(system, u), "usd_guard": list_usd(system, u, guard=True),
                   "usd_reported": u.get("cost"), "tokens_in": u.get("prompt_tokens"),
                   "tokens_out": u.get("completion_tokens"), "model": resp.get("model"),
                   "provider": resp.get("provider"), "latency_s": lat,
                   "transport_retry": req.get(TRANSPORT_RETRY_KEY, 0)}
            run.ledger(rec)
            run.counts["stored"] = run.counts.get("stored", 0) + 1
            model = resp.get("model")
            if model != EXPECTED_MODEL[system]:
                run.stopped[system] = f"reply names model {model!r}, the main runs {EXPECTED_MODEL[system]!r}"
                run.log(f"HARD STOP {system}: {run.stopped[system]} ({form}/{item})")
            return
        finally:
            run.inflight_usd -= est_usd
            await lim.release(ok)
    run.log(f"attempts exhausted {system}/{form}/{item}: not stored")
    run.errors.append({"system": system, "form": form, "item": item, "status": "exhausted"})


# ------------------------------------------------------------------------------------------------ guards
async def guard_loop(run: Run, client: httpx.AsyncClient, done: asyncio.Event):
    while not done.is_set():
        s = spent_all() + run.inflight_usd
        if BUDGET_USD is not None and s >= BUDGET_USD - 2.0 and not run.all_stop:
            run.all_stop = f"budget: US${s:.2f} spent or in flight of US${BUDGET_USD:.0f}"
            run.log(f"HARD STOP {run.all_stop}")
        if run.url == OPENROUTER_URL and (KEY_MARGIN_USD is not None or ACCOUNT_MARGIN_USD is not None):
            for kname, kval in run.guard_keys.items():
                use, limit = await key_usage(client, kval)
                bal = await account_balance(client, kval)
                if use is None:
                    continue
                cap = (limit - KEY_MARGIN_USD) if (limit and KEY_MARGIN_USD is not None) else None
                low = (cap is not None and use >= cap) or (ACCOUNT_MARGIN_USD is not None and bal is not None
                                                           and bal < ACCOUNT_MARGIN_USD)
                if not low:
                    continue
                why = f"{kname}: key usage US${use:.2f} (guard {cap}), account balance US${bal}"
                for lname, kn in run.pool_key.items():
                    if kn == kname and lname not in run.disabled:
                        run.disabled.add(lname)
                        run.log(f"pool {lname} stopped: {why}")
        try:
            await asyncio.wait_for(done.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass


# ------------------------------------------------------------------------------------------------ main
def estimate(system: str, req: dict) -> float:
    """Worst-case cost of one request for the in-flight budget reserve: input characters / 3 tokens, 4,000 out."""
    chars = sum(len(m["content"]) for m in req["messages"])
    pi, po = budget_price(system)
    return (chars / 3 * pi + 4000 * po) / 1e6


async def main_async(a):
    g = GROUPS[a.group]
    systems = [s for s in g["systems"] if not a.systems or s in a.systems]
    env = dotenv_values(ENV)
    key = env.get(g["key"])
    if not key:
        raise SystemExit(f"{g['key']} missing from .env")
    for d in (STORE, LEDGER, STATE):
        d.mkdir(parents=True, exist_ok=True)
    run = Run(a.group, a.mode, key, g["url"], systems, RawStore(STORE),
              run_id=f"{a.group}-{a.mode}-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    cells = gaps(systems, "smoke" if a.mode in ("smoke", "dry-smoke") else "full")
    reqs = []
    for s, f, i in cells:
        req = LF.build(s, f, i)            # content assertion inside; raises on any difference
        reqs.append((s, f, i, req))
    if a.mode.endswith("transport-retry"):  # resend only cells whose stored replies all failed in transport
        pending = []
        for s, f, i, r in reqs:
            got, k, failed = answered(run.store, r)
            if got is None and failed is not None and k <= MAX_TRANSPORT_RETRIES:
                pending.append((s, f, i, retry_req(r, k)))
    else:
        pending = [(s, f, i, r) for s, f, i, r in reqs if run.store.get(r) is None]
    run.log(f"mode {a.mode}: {len(cells)} gap cells for {systems}; {len(cells) - len(pending)} already stored; "
            f"{len(pending)} to send")
    if a.mode.startswith("dry"):
        by = pd.DataFrame([(s, f) for s, f, _, _ in pending], columns=["system", "form"]).value_counts().sort_index()
        print(by.to_string())
        return
    qnames = sorted({PROVIDER_OF[s] for s in systems})
    key_vars = list(getattr(a, "keys", None) or [g["key"]])
    pools = {q: ([(f"{q}_key{n + 1}", kv, *PER_KEY[q]) for n, kv in enumerate(key_vars)] if q in PER_KEY
                 else [(q, g["key"], *LIMITS[q])]) for q in qnames}
    keys = {kn: env.get(kn) for q in qnames for _, kn, *_ in pools[q]}
    if not all(keys.values()):
        raise SystemExit(f"a key is missing from .env: {[k for k, v in keys.items() if not v]}")
    limits = {name: Adaptive(name, start, mx, pm) for q in qnames for name, _, start, mx, pm in pools[q]}
    queues: dict[str, asyncio.Queue] = {q: asyncio.Queue() for q in qnames}
    run.pool_key = {name: kn for q in qnames for name, kn, *_ in pools[q]}
    if g["url"] == OPENROUTER_URL:
        run.guard_keys = dict(keys)
    for s, f, i, r in pending:
        queues[PROVIDER_OF[s]].put_nowait((s, f, i, r))
    done = asyncio.Event()
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=600.0)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=300, max_keepalive_connections=100)) as client:
        for kname, kval in run.guard_keys.items():
            use, lim = await key_usage(client, kval)
            run.log(f"key {kname}: usage US${use} of limit US${lim}")

        async def worker(qn: str, lname: str, kname: str):
            q = queues[qn]
            while True:
                try:
                    s, f, i, r = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if run.all_stop or s in run.stopped:
                    continue
                if lname in run.disabled:
                    q.put_nowait((s, f, i, r))
                    return
                try:
                    res = await call(run, client, limits[lname], keys[kname], g["url"], s, f, i, r, estimate(s, r))
                    if res == "requeue":
                        q.put_nowait((s, f, i, r))
                        return
                except Exception as e:
                    run.log(f"worker error {s}/{f}/{i}: {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")

        tasks = [asyncio.create_task(worker(q, name, kn)) for q in qnames for name, kn, _, mx, _ in pools[q]
                 for _ in range(mx)]
        side = [asyncio.create_task(guard_loop(run, client, done))]
        await asyncio.gather(*tasks)
        done.set()
        await asyncio.gather(*side, return_exceptions=True)
    summary = {"group": a.group, "mode": a.mode, "run_id": run.run_id, "finished": now(), "counts": run.counts,
               "stopped": run.stopped, "all_stop": run.all_stop, "errors": run.errors[-200:],
               "n_errors": len(run.errors), "rate_limit_hits": {p: l.hits for p, l in limits.items()},
               "final_limits": {p: l.limit for p, l in limits.items()}, "spent_all_usd": spent_all()}
    (STATE / f"{a.group}_{a.mode}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    run.log(f"finished: {json.dumps({k: summary[k] for k in ('counts', 'stopped', 'all_stop', 'n_errors', 'spent_all_usd')})}")


def main():
    global BUDGET_USD, KEY_MARGIN_USD, ACCOUNT_MARGIN_USD
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--mode", required=True, choices=["dry", "dry-smoke", "smoke", "full", "transport-retry",
                                                      "dry-transport-retry"])
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--keys", nargs="+", default=None,
                    help=".env variables for the Muse or Hugging Face queue, one worker pool each (default: the "
                         "group's key)")
    ap.add_argument("--max-usd", type=float, default=None, help="stop every process at this spend on new calls")
    ap.add_argument("--key-margin-usd", type=float, default=None,
                    help="stop a key's pool this far short of the key's own limit")
    ap.add_argument("--account-margin-usd", type=float, default=None,
                    help="stop a key's pool when its account has less than this left")
    a = ap.parse_args()
    BUDGET_USD, KEY_MARGIN_USD, ACCOUNT_MARGIN_USD = a.max_usd, a.key_margin_usd, a.account_margin_usd
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
