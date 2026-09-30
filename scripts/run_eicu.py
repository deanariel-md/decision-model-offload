"""eICU replication: baseline prediction of death before hospital discharge on the open eICU-CRD demo only (population
eicu_demo; the credentialed database never goes to any model). Jev and the five primary chatbots on the unedited
records of the evaluation stays, statement "This patient will die before hospital discharge." (config/prompts.yaml);
GPT, Opus and Gemini through the OpenRouter Batch API, Jev, Muse and GLM on the standard route. No edits, no repeats.
Answers in runs/eicu_demo; calls in results/eicu/calls_eval.parquet (the standard-route calls also in
calls_eval_sync.parquet, written while the batches run).
The cost at standard list price for the evaluation stays and the masked-field test (scripts/eicu_exposure.py) is printed
before any call: input tokens estimated from the record text (data/eicu/states_eval.parquet, 4 characters per token
plus the prompt), --tokens_out output tokens assumed per call (default 400); with --stop_usd, nothing is sent when the
total is above that amount.
  python scripts/run_eicu.py estimate [--tokens_out 400]
  python scripts/run_eicu.py eval --yes

Secondary systems (--tiers free,pair; the default tier, primary, is the run above): GPT-5.6 Luna, Claude Sonnet 5 and
Gemini 3.5 Flash-Lite (free plans, through OpenRouter), MedGemma and Gemma (Hugging Face router) answer the same
question on the same baseline records of the evaluation stays, standard route, one call per stay; the first stored
reply stands (RawStore, runs/eicu_demo). Before any call every request is compared with the stored GPT-5.6 Sol request
for the same stay (results/eicu/calls_eval.parquet): only the model, the provider block and the seed may differ, and
those must equal the family's own requests on the NHANES evaluation baselines (results/eval/calls.parquet): its pinned
provider with allow_fallbacks false on the standard route, no provider block on the Hugging Face router, the seed where
the NHANES requests sent one, 16,000 output tokens, no temperature or reasoning parameter. Any other difference stops.
A reply naming another model or provider stops every system. Optional stops: --stop_usd (the projected list cost of the
five systems, high case, and the list spend of the replies) and --usage_cap_usd (the OpenRouter key's recorded usage
plus that projection; asks OpenRouter for the key's usage). The OpenRouter families run in one process (--workers
each), MedGemma and Gemma in another (--tiers pair; --hf_workers calls in flight on the token across both, halved on
HTTP 429 or on the provider's concurrency refusal; --hf_key names the .env variable that holds the token).
  python scripts/run_eicu.py check --tiers free,pair      # build and compare every request; projection; no call
  python scripts/run_eicu.py stub --tiers free,pair       # whole path on simulated replies, scratch store; no call
  python scripts/run_eicu.py eval --tiers free --yes      # the evaluation stays x 3 systems (OpenRouter)
  python scripts/run_eicu.py eval --tiers pair --yes      # the evaluation stays x MedGemma and Gemma
  python scripts/run_eicu.py tables --tiers free,pair     # results/eicu/calls_eval_secondary.parquet"""
import argparse, json, os, sys, tempfile, threading, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import batch as BT                                                     # noqa: E402
from jevity import clients as CL                                                   # noqa: E402
from jevity.clients import (BATCH_MISSING, MODELS, PROMPTS, LLMClient, RawStore, active_families,  # noqa: E402
                            estimate_tokens, split_system)
from jevity.runner import Call, execute, make_client_factory                      # noqa: E402

POP = "eicu_demo"
D = ROOT / "data" / "eicu"
OUT = ROOT / "results" / "eicu"
MASKED_N = 200                      # masked-field test (scripts/eicu_exposure.py)
TOKENS_OUT = 400.0                  # output tokens per call assumed by the cost estimate (--tokens_out)


def systems(tiers: tuple = ("primary",)) -> list[str]:
    if tuple(tiers) == ("primary",):
        return ["jev"] + active_families(("primary",))
    return active_families(tuple(tiers))


def states(split: str) -> pd.DataFrame:
    st = pd.read_parquet(D / f"states_{split}.parquet")
    if set(st.edit) != {"baseline"}:
        raise SystemExit(f"states_{split}.parquet holds edited states: the replication asks baselines only")
    return st


def plan(split: str) -> list[Call]:
    st = states(split)
    return [Call(m, int(r.profile), "baseline", False, 0, r.text) for m in systems() for r in st.itertuples(index=False)]


def estimate(quiet: bool = False, stop_usd: float | None = None, tokens_out: float = TOKENS_OUT) -> dict:
    """List-price projection per system for the evaluation stays and the masked-field test: input tokens from the record
    length (as cost_estimate), tokens_out output tokens assumed per call (Jev bills input only)."""
    import cost_estimate as CE
    out, total = {}, 0.0
    st = states("eval")
    tok = st.text.map(estimate_tokens).mean() + CE.PROMPT_OVERHEAD
    per = {s: (tok * MODELS["jev"]["price_per_mtok_input"] / 1e6 if s == "jev"
               else CE.per_call(s, tok, population=POP, standard=True, tokens_out=tokens_out)["usd"]) for s in systems()}
    out["eval"] = {"records": len(st), "input_tokens": round(tok), "usd": {s: v * len(st) for s, v in per.items()}}
    total += sum(out["eval"]["usd"].values())
    tok = out["eval"]["input_tokens"]
    out["masked"] = {"records": MASKED_N, "usd": {s: MASKED_N * CE.per_call(s, tok, population=POP, standard=True,
                                                                            tokens_out=tokens_out)["usd"]
                                                  for s in active_families(("primary",))}}
    total += sum(out["masked"]["usd"].values())
    out["total_usd"] = total
    if not quiet:
        for part, v in out.items():
            if part == "total_usd":
                continue
            print(f"  {part:7s} {v['records']:5,} stays: " + ", ".join(f"{s} ${u:.2f}" for s, u in v["usd"].items())
                  + f"  = ${sum(v['usd'].values()):.2f}")
        print(f"PROJECTED eICU total at standard list price: ${total:.2f} ({tokens_out:.0f} output tokens per call "
              f"assumed)" + (f" (stop above ${stop_usd:.2f})" if stop_usd is not None else ""))
    return out


def batch_loop(calls: list[Call], store: RawStore, interval: int) -> None:
    """Submit and collect the batch-route requests until none is pending and none waiting: every round collects the
    finished batches, then submits each family's pending requests (in chunks of config/models.yaml batch.max_requests)."""
    fams = sorted({split_system(c.model)[0] for c in calls} & set(MODELS["batch"]["families"]))
    n0 = int(MODELS["batch"].get("max_requests", 1000))
    while True:
        t = BT.collect(store)
        man = BT.Manifest(store)
        busy = man.pending_ids()
        left = {}
        for f in fams:
            fc = [c for c in calls if split_system(c.model)[0] == f]
            left[f] = (fc, len([x for x in BT.pending_requests(fc, store, POP) if x[1] not in busy]))
        print(time.strftime("%H:%M"), f"batch: written {t['written']}, failed {t['failed']}, waiting "
              f"{t['waiting_batches']}; pending "
              + ", ".join(f"{f} {k}" for f, (_, k) in left.items()), flush=True)
        if not any(k for _, k in left.values()) and t["waiting_batches"] == 0:
            return
        for f in fams:
            fc, k = left[f]
            if not k:
                continue
            try:
                BT.submit(fc, store, POP, max_requests=min(n0, k), max_total=k)
            except BT.Refused as e:
                print(f"  {f}: refused ({str(e)[:160]}); waiting", flush=True)
                break
        time.sleep(interval)


# ---------------------------------------------------------------- secondary systems (--tiers free,pair)
SOL = "openai/gpt-5.6-sol"
VARY = ("model", "provider", "seed", "_route")          # the only keys a secondary request may differ in from Sol's
PROVIDER = {"gpt_free": {"OpenAI"}, "claude_free": {"Anthropic"}, "gemini_free": {"Google"},
            "medgemma": {None, "featherless-ai", "Featherless"}, "gemma": {None, "featherless-ai", "Featherless"}}


def local(p: str) -> Path:
    """A stored raw path: absolute as the runner wrote it, or relative to the repository root."""
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def base_model(f: str) -> str:
    s = MODELS["families"][f]
    return s["hf_model"].split(":")[0] if s.get("transport") == "hf_router" else s["slug"]


def sec_states(split: str) -> pd.DataFrame:
    """The split's baseline states in stay order (the evaluation stays)."""
    return states(split).sort_values("profile")


def stored_requests(split: str, model: str = "gpt") -> dict[int, dict]:
    c = pd.read_parquet(OUT / f"calls_{split}.parquet")
    c = c[(c.model == model) & (c.edit == "baseline") & (c.repeat == 0)]
    return {int(r.profile): json.loads(local(r.raw_path).read_text(encoding="utf-8"))["request"] for r in c.itertuples()}


def nhanes_request(f: str) -> dict:
    """The family's stored request on its first NHANES evaluation baseline (settings only; results/eval/calls.parquet)."""
    c = pd.read_parquet(ROOT / "results" / "eval" / "calls.parquet", columns=["model", "edit", "repeat", "variant", "raw_path"])
    c = c[(c.model == f) & (c.edit == "baseline") & (c.repeat == 0) & (c.variant == "raw")]
    return json.loads(local(c.raw_path.iloc[0]).read_text(encoding="utf-8"))["request"]


class SecClient(LLMClient):
    """The project's chatbot client on the standard route; `sender` replaces the network (stub, or refuse in tables)."""
    sender = None

    def _send(self, req: dict) -> dict:
        if self.sender is None:
            return super()._send(req)
        resp = self.sender(self.family, req)
        meta = {"ts": time.time(), "latency_s": 0.0, "family": self.family, "route": "standard",
                "transport": self.transport, "provider_reported": resp.get("provider"),
                "model_reported": resp.get("model"), "usage": resp.get("usage"), "simulated": True}
        if not self.store.put(req, resp, meta):
            return self.store.get(req)
        return {"request": req, "response": resp, "meta": meta}


def check_request(f: str, profile: int, req: dict, sol: dict, ref: dict) -> None:
    """Stops with the differing keys unless the request equals Sol's apart from VARY, and VARY is the family's own."""
    bad = []
    if sol.get("model") != SOL or sol.get("_route") != "batch":
        bad.append("stored Sol request is not the batch-route Sol request")
    same = lambda q: {k: v for k, v in q.items() if k not in VARY}
    a, b = same(req), same(sol)
    bad += [f"{k} differs from Sol" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    spec = MODELS["families"][f]
    hf = spec.get("transport") == "hf_router"
    if req["model"] != (spec["hf_model"] if hf else spec["slug"]) or req["model"] != ref["model"]:
        bad.append(f"model {req['model']} (arm 1: {ref['model']})")
    if not (req["max_tokens"] == ref["max_tokens"] == 16000):
        bad.append("max_tokens")
    if req.get("seed") != ref.get("seed"):
        bad.append(f"seed {req.get('seed')} (arm 1: {ref.get('seed')})")
    if req.get("response_format") != ref.get("response_format"):
        bad.append("response_format differs from arm 1")
    if any(k in q for q in (req, ref) for k in ("temperature", "reasoning")):
        bad.append("temperature or reasoning parameter")
    if "_route" in req:
        bad.append("request marked for the batch route")
    if hf:
        if "provider" in req or "provider" in ref:
            bad.append("provider block on the Hugging Face router")
    else:
        want = {"only": list(ref["provider"]["only"]), "allow_fallbacks": False}
        if req.get("provider") != want or ("_route" not in ref and ref["provider"] != want):
            bad.append(f"provider {req.get('provider')} (arm 1: {ref.get('provider')})")
    if bad:
        raise SystemExit(f"request check stopped: {f} stay {profile}: " + "; ".join(bad))


def sec_build(split: str, fams: list[str], store: RawStore, sender=None) -> list[tuple]:
    """(family, stay, client, request, Sol's stored input tokens) for every call; stops on any difference."""
    st, sol, stmt = sec_states(split), stored_requests(split), PROMPTS["statements"][POP]
    tin = pd.read_parquet(OUT / f"calls_{split}.parquet")
    tin = tin[tin.model == "gpt"].set_index("profile")["tokens_in"]
    jobs = []
    for f in fams:
        ref = nhanes_request(f)
        cl = SecClient(f, store, POP, standard=True)
        cl.sender = sender
        for r in st.itertuples(index=False):
            req = cl.request(r.text, stmt)
            check_request(f, int(r.profile), req, sol[int(r.profile)], ref)
            jobs.append((f, int(r.profile), cl, req, float(tin.loc[int(r.profile)])))
    return jobs


def sec_projection(jobs: list[tuple]) -> dict:
    """List price. Input: Sol's reported input tokens for the stay, times the family's ratio of reported input tokens to
    Sol's on the NHANES evaluation baselines (tokenizers differ). Output: the family's mean on those baselines (and the
    90th centile as a high case)."""
    c = pd.read_parquet(ROOT / "results" / "eval" / "calls.parquet",
                        columns=["model", "edit", "repeat", "variant", "tokens_in", "tokens_out"])
    c = c[(c.edit == "baseline") & (c.repeat == 0) & (c.variant == "raw")]
    g = c.groupby("model")
    out = {}
    for f in dict.fromkeys(j[0] for j in jobs):
        s = MODELS["families"][f]
        ratio = g.tokens_in.mean()[f] / g.tokens_in.mean()["gpt"]
        ti = sum(j[4] for j in jobs if j[0] == f) * ratio
        n = sum(1 for j in jobs if j[0] == f)
        mean_o, p90_o = float(g.tokens_out.mean()[f]), float(g.tokens_out.quantile(0.9)[f])
        out[f] = {"calls": n, "usd": (ti * s["price_in"] + n * mean_o * s["price_out"]) / 1e6,
                  "usd_high": (ti * s["price_in"] + n * p90_o * s["price_out"]) / 1e6}
    return out


def key_usage() -> float:
    import httpx
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing from .env")
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


def list_usd(f: str, row: dict) -> float:
    u = row.get("usage") or {}
    s = MODELS["families"][f]
    return ((u.get("prompt_tokens") or 0) * s["price_in"] + (u.get("completion_tokens") or 0) * s["price_out"]) / 1e6


class Gate:
    """At most n calls in flight; halve() on HTTP 429 or a concurrency refusal, at most once per `cool` seconds."""

    def __init__(self, name: str, n: int, cool: float = 45.0):
        self.name, self.n, self.busy, self.cool, self.last = name, n, 0, cool, 0.0
        self.cv = threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.busy >= self.n:
                self.cv.wait()
            self.busy += 1

    def __exit__(self, *exc):
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()

    def halve(self, why: str) -> None:
        with self.cv:
            if time.time() - self.last < self.cool or self.n == 1:
                return
            self.n, self.last = max(1, self.n // 2), time.time()
            print(time.strftime("%H:%M:%S"), f"{self.name}: {why}; now {self.n} in flight", flush=True)


def concurrency_refusal(e: Exception) -> bool:
    """A refusal that carries no reply: Featherless's concurrency 400, or clients._post giving up on HTTP 429/5xx."""
    import httpx
    if isinstance(e, RuntimeError) and str(e) == "retries exhausted":
        return True
    return (isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 429)
            and ("concurren" in e.response.text.lower() or e.response.status_code == 429))


def sec_answer(jobs: list[tuple], workers: dict, hf: bool, cap_usd: float | None, max_tries: int = 30) -> list[dict]:
    """One call per job (a stored reply is read, never resent). A concurrency refusal carries no reply and is sent again
    after the gate halves; any other failure is recorded and left for a later pass. Stops every system on a reply naming
    another model or provider, or, when cap_usd is given, when the replies' list spend passes it. Hugging Face router:
    one gate for the token, shared by both models (Featherless serves 5 requests at a time per token)."""
    if hf:
        g = Gate("Hugging Face token (" + ", ".join(workers) + ")", max(workers.values()))
        gates = {f: g for f in workers}
    else:
        gates = {f: Gate(f, n) for f, n in workers.items()}
    if hf:
        by_model = {MODELS["families"][f]["hf_model"]: f for f in workers}
        CL.ON_RETRY_STATUS = lambda status, url, body: (status == 429 and body.get("model") in by_model
                                                        and gates[by_model[body["model"]]].halve("HTTP 429"))
    stop, lock, rows, spent, t0 = threading.Event(), threading.Lock(), [], [0.0], time.time()
    done = {f: 0 for f in workers}
    total = {f: sum(1 for j in jobs if j[0] == f) for f in workers}

    def one(job):
        f, prof, cl, req, _ = job
        for k in range(max_tries):
            if stop.is_set():
                return None
            try:
                if cl.store.get(req) is not None:
                    r = cl.probability_req(req)
                else:
                    with gates[f]:
                        r = cl.probability_req(req)
                break
            except Exception as e:  # noqa: BLE001
                if concurrency_refusal(e):
                    gates[f].halve(f"HTTP {e.response.status_code} concurrency refusal")
                    time.sleep(5 + 5 * k)
                    continue
                r = {"p": None, "valid": False, "error": repr(e)[:300]}
                break
        else:
            r = {"p": None, "valid": False, "error": "concurrency refusals: tries exhausted"}
        rep = (r.get("model_reported") or "").split("-2026")[0]
        bad = r.get("error") is None and (rep != base_model(f) or r.get("provider") not in PROVIDER[f])
        with lock:
            done[f] += 1
            spent[0] += list_usd(f, r)
            if bad or (cap_usd is not None and spent[0] > cap_usd):
                stop.set()
            if done[f] % 100 == 0 or done[f] == total[f]:
                el = time.time() - t0
                eta = (total[f] - done[f]) * el / done[f] / 3600
                print(time.strftime("%H:%M:%S"), f"{f}: {done[f]}/{total[f]} ({3600 * done[f] / el:.0f}/h; "
                      f"{eta:.1f} h left; in flight max {gates[f].n}); list spend so far US${spent[0]:.3f}", flush=True)
        return {"family": f, "profile": prof, **r, "stopped_here": bool(bad)}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    pools = {f: ThreadPoolExecutor(max_workers=n, thread_name_prefix=f) for f, n in workers.items()}
    try:
        futs = [pools[j[0]].submit(one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            if r is not None:
                rows.append(r)
    finally:
        for p in pools.values():
            p.shutdown(wait=True)
    if stop.is_set():
        why = [r for r in rows if r.get("stopped_here")]
        raise SystemExit(f"stopped: {'a reply named another model or provider: ' + str(why[:3]) if why else 'list spend over US$' + str(cap_usd)}")
    return rows


def _prob_req(self, req: dict) -> dict:
    """LLMClient.probability for an already-built request (the same code path: store first, then send)."""
    cached = self.store.get(req)
    if cached is None:
        with self.store.lock(req):
            cached = self.store.get(req) or self._send(req)
    resp = cached["response"]
    try:
        content = resp["choices"][0]["message"].get("content")
    except Exception:
        content = None
    return self._result(req, cached, resp, content)


SecClient.probability_req = _prob_req


def stub_sender(family: str, req: dict) -> dict:
    import random
    rng = random.Random(json.dumps(req["messages"], sort_keys=True) + family)
    return {"id": "sim", "model": base_model(family), "provider": sorted(PROVIDER[family], key=str)[0] if None not in PROVIDER[family] else None,
            "choices": [{"message": {"content": json.dumps({"p": round(rng.random(), 3)})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 60}}


def refuse(family: str, req: dict) -> dict:
    raise RuntimeError("no stored reply")


def sec_tables(fams: list[str], store: RawStore) -> None:
    """calls_eval_secondary.parquet with the primary run's columns (runner.execute on the store; never sends)."""
    split = "eval"
    jobs = sec_build(split, fams, store, sender=refuse)
    cls = {f: cl for f, _, cl, _, _ in jobs}
    calls = [Call(f, p, "baseline", False, 0, sec_states(split).set_index("profile").text[p]) for f, p, *_ in jobs]
    df = execute(calls, lambda m, r, v: cls[m], POP, OUT / f"calls_{split}_secondary.parquet", workers=8)
    print(f"calls_{split}_secondary.parquet: {len(df)} rows")
    print(df.groupby("model").agg(rows=("valid", "size"), usable=("valid", "sum"),
                                  errors=("error", lambda s: int(s.notna().sum()))).to_string())
    print(df.groupby(["model", "model_reported", "provider", "route"]).size().to_string())


def secondary(a) -> None:
    tiers = tuple(t.strip() for t in a.tiers.split(","))
    fams = systems(tiers)
    if a.systems:                                        # one family per process, e.g. one per Hugging Face token
        fams = [f for f in fams if f in a.systems.split(",")]
    if not fams or any(MODELS["families"][f].get("tier") not in ("free", "pair") for f in fams):
        raise SystemExit(f"--tiers {a.tiers}: secondary tiers are free and pair")
    from dotenv import load_dotenv, dotenv_values
    load_dotenv(ROOT / ".env")                           # keys read by the clients; never printed
    store = RawStore(ROOT / "runs" / "eicu_demo")
    if a.what == "stub":
        with tempfile.TemporaryDirectory() as d:
            jobs = sec_build("eval", fams, RawStore(Path(d)), sender=stub_sender)
            rows = sec_answer(jobs, {f: 8 for f in fams}, hf=False, cap_usd=1e9)
            assert len(rows) == len(jobs) and all(r["valid"] for r in rows), "stub: every simulated reply parses"
            print(f"stub eval: {len(rows)} simulated replies, all parsed; " +
                  ", ".join(f"{f} {sum(r['family'] == f for r in rows)}" for f in fams))
        return
    if a.what == "tables":
        sec_tables(fams, store)
        return
    split = "eval"
    all5 = active_families(("free", "pair"))
    jobs = sec_build(split, all5, store)                 # every request of the task is built and checked each time
    proj = sec_projection(jobs)
    unanswered = sum(1 for j in jobs if store.get(j[3]) is None)
    print(f"{split}: {len(jobs)} requests ({len(jobs) // len(all5)} stays x {', '.join(all5)}), {unanswered} without a "
          "stored reply; every one equals the stored GPT-5.6 Sol request for its stay apart from the model, the "
          "provider block and the seed, which equal the family's arm 1 requests")
    for f, v in proj.items():
        print(f"  {f:12s} {v['calls']:5,} calls  projected US${v['usd']:.3f} (high US${v['usd_high']:.3f}) at list price")
    task = sum(v["usd_high"] for v in proj.values())
    print(f"projected list cost of this task (five systems, evaluation stays, high case) US${task:.2f}"
          + (f" (stop above US${a.stop_usd:.2f})" if a.stop_usd is not None else ""))
    if a.what == "check":
        return
    if not a.yes:
        raise SystemExit("run needs --yes")
    if a.stop_usd is not None and task > a.stop_usd:
        raise SystemExit("projection over --stop_usd: nothing sent")
    if a.usage_cap_usd is not None:
        used = key_usage()
        print(f"key usage US${used:,.2f}; with the projection US${used + task:,.2f} (cap US${a.usage_cap_usd:,.2f})")
        if used + task > a.usage_cap_usd:
            raise SystemExit("over --usage_cap_usd: nothing sent")
    hf = any(MODELS["families"][f].get("transport") == "hf_router" for f in fams)
    if hf:
        if any(MODELS["families"][f].get("transport") != "hf_router" for f in fams):
            raise SystemExit("the Hugging Face pair runs in its own process (--tiers pair)")
        tok = dotenv_values(ROOT / ".env").get(a.hf_key)
        if not tok:
            raise SystemExit(f"{a.hf_key} missing from .env")
        os.environ["HF_TOKEN"] = tok                     # the client reads HF_TOKEN; never printed
        print(f"Hugging Face router token: {a.hf_key}")
    jobs = sec_build(split, fams, store)
    workers = {f: (a.hf_workers if hf else a.workers) for f in fams}
    rows = sec_answer(jobs, workers, hf=hf, cap_usd=a.stop_usd)
    df = pd.DataFrame([{"error": None, **r} for r in rows])
    print(df.groupby("family").agg(answered=("valid", "size"), usable=("valid", "sum"),
                                   errors=("error", lambda s: int(s.notna().sum()))).to_string())
    err = df[df.error.notna()]
    if len(err):
        print("errors (no reply stored; running the command again sends only these):\n",
              err.groupby(["family", err.error.str[:120]]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["estimate", "eval", "check", "stub", "tables"])
    ap.add_argument("--tiers", default="primary", help="primary (Jev and the five primary chatbots) or a comma list of "
                                                       "free, pair")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--stop_usd", type=float, default=None,
                    help="send nothing when the projected cost (USD, standard list price) is above this; secondary "
                         "tiers: also stop when the replies' list spend passes it; default: no stop")
    ap.add_argument("--usage_cap_usd", type=float, default=None,
                    help="secondary: send nothing when the OpenRouter key's recorded usage plus the projection is above "
                         "this (asks OpenRouter for the key's usage); default: no cap")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--glm_workers", type=int, default=48)
    ap.add_argument("--hf_workers", type=int, default=5, help="secondary: calls in flight on the Hugging Face token, both models")
    ap.add_argument("--hf_key", default="HF_TOKEN", help="secondary: the .env variable that holds the Hugging Face router token")
    ap.add_argument("--systems", default=None, help="secondary: only these families (comma list) of the tiers")
    ap.add_argument("--interval", type=int, default=120, help="seconds between batch collect/submit rounds")
    ap.add_argument("--tokens_out", type=float, default=TOKENS_OUT, help="output tokens per call assumed by the cost "
                                                                         "estimate")
    a = ap.parse_args()
    if a.tiers != "primary":
        secondary(a)
        sys.exit(0)
    if a.what in ("check", "stub", "tables"):
        raise SystemExit(f"{a.what} is for the secondary tiers (--tiers free,pair)")
    est = estimate(stop_usd=a.stop_usd, tokens_out=a.tokens_out)
    if a.what == "estimate":
        sys.exit(0)
    if a.stop_usd is not None and est["total_usd"] > a.stop_usd:
        raise SystemExit(f"projected ${est['total_usd']:.2f} is above --stop_usd ${a.stop_usd:.2f}: nothing sent")
    calls = plan(a.what)
    print(f"{a.what}: {len(calls)} calls ({len(calls) // len(systems())} stays x {', '.join(systems())})")
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    store = RawStore(ROOT / "runs" / "eicu_demo")
    OUT.mkdir(parents=True, exist_ok=True)
    factory = make_client_factory(store, POP)
    batch_fams = set(MODELS["batch"]["families"])
    sync = [c for c in calls if split_system(c.model)[0] not in batch_fams]
    workers = {"default": a.workers, "glm": a.glm_workers}
    th = threading.Thread(target=execute, args=(sync, factory, POP, OUT / f"calls_{a.what}_sync.parquet", workers),
                          name="sync")
    th.start()
    batch_loop([c for c in calls if split_system(c.model)[0] in batch_fams], store, a.interval)
    th.join()
    df = execute(calls, factory, POP, OUT / f"calls_{a.what}.parquet", workers=workers)     # every answer from the store
    print(df.groupby("model")["valid"].agg(["size", "sum", "mean"]).to_string())
    print("providers:\n", df.groupby(["model", "provider"]).size().to_string())
    print("routes:\n", df.groupby(["model", "route"]).size().to_string())
    miss = df[df.error == BATCH_MISSING]
    print(f"wrote {OUT / f'calls_{a.what}.parquet'}; batch answers not collected: {len(miss)}")
