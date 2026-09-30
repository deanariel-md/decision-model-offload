"""Timing sample (cost, speed and value): the runtime comparison between systems.
The same 100 evaluation records (the first 100 in the seeded block order, as in route_bridge.py) are asked as baseline
questions to every system on the standard route (batch families included), one question at a time per system. All
systems run in the same window: one worker per system, and question i of every system starts no earlier than
window_start + i x (window / 100), so every system is sampled across the same hour instead of in turn. Latency is the
client-side seconds from request to reply, retries on 429/5xx included (what a caller waits). Answers are stored in their
own raw store (runs/timing_<date>), never in the evaluation store, and never enter D or any other analysis.
Run it while no other calls are in flight. Prints its cost estimate and asks before sending (--yes skips the question).
  python scripts/timing_sample.py [--n 100] [--window_min 60] [--date YYYYMMDD] [--yes]
  python scripts/timing_sample.py --simulate               # dry run only: invented latencies, no API
Writes results/eval/timing_sample.json and results/eval/timing_sample_calls.parquet."""
import argparse, json, sys, threading, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.clients import JevClient, LLMClient, RawStore, MODELS, PROMPTS, active_families, estimate_tokens
from jevity.runner import Call, block_order

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("primary", "free", "pair")


def systems() -> list[str]:
    return ["jev"] + active_families(TIERS)


def first_records(states: pd.DataFrame, n: int) -> pd.DataFrame:
    base = states[(states.edit == "baseline") & ~states.annotated]
    order = [c.profile for c in block_order([Call("x", int(p), "baseline", False, 0, "") for p in base.profile])]
    first = list(dict.fromkeys(order))[:n]
    return base.set_index("profile").loc[first].reset_index()


def client(system: str, store: RawStore):
    if system == "jev":
        return JevClient(store, "nhanes")
    return LLMClient(system, store, "nhanes", standard=True)       # standard route even for batch families


def list_price(system: str, usage: dict | None) -> float | None:
    """Standard list price of one answer from its reported tokens (usage.cost when the route reports it)."""
    u = usage or {}
    if u.get("cost") is not None:
        return float(u["cost"])
    if system == "jev":
        tin = u.get("input_tokens") or u.get("prompt_tokens")
        return None if tin is None else tin * MODELS["jev"]["price_per_mtok_input"] / 1e6
    f = MODELS["families"][system]
    tin, tout = u.get("prompt_tokens"), u.get("completion_tokens")
    return None if tin is None or tout is None else (tin * f["price_in"] + tout * f["price_out"]) / 1e6


def run_system(system: str, recs: pd.DataFrame, store: RawStore, t0: float, gap: float, rows: list, lock,
               make_client=client) -> None:
    stmt = PROMPTS["statements"]["nhanes"]
    cl = make_client(system, store)
    for i, r in enumerate(recs.itertuples(index=False)):
        wait = t0 + i * gap - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.time()
        try:
            res = cl.probability(r.text, stmt)
            err = None
        except Exception as e:                    # recorded, never dropped; counts as a failed call
            res, err = {}, repr(e)[:300]
        row = {"system": system, "i": i, "profile": int(r.profile), "started": start, "latency_s": res.get("latency_s"),
               "valid": bool(res.get("valid")), "provider": res.get("provider"), "model_reported": res.get("model_reported"),
               "route": res.get("route"), "tokens_in": (res.get("usage") or {}).get("prompt_tokens",
                                                                                   (res.get("usage") or {}).get("input_tokens")),
               "tokens_out": (res.get("usage") or {}).get("completion_tokens", (res.get("usage") or {}).get("output_tokens")),
               "usd_list": list_price(system, res.get("usage")), "error": err or res.get("error")}
        with lock:
            rows.append(row)


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for s, g in df.groupby("system", sort=False):
        lat = g.latency_s.dropna().astype(float)
        out[s] = {"n": int(len(g)), "usable": int(g.valid.sum()), "errors": int(g.error.notna().sum()),
                  "median_s": float(lat.median()) if len(lat) else None,
                  "p90_s": float(np.percentile(lat, 90)) if len(lat) else None,
                  "mean_s": float(lat.mean()) if len(lat) else None,
                  "min_s": float(lat.min()) if len(lat) else None, "max_s": float(lat.max()) if len(lat) else None,
                  "mean_tokens_in": float(g.tokens_in.dropna().mean()) if g.tokens_in.notna().any() else None,
                  "mean_tokens_out": float(g.tokens_out.dropna().mean()) if g.tokens_out.notna().any() else None,
                  "usd_list_total": float(g.usd_list.fillna(0).sum()),
                  "providers": {str(k): int(v) for k, v in g.provider.value_counts(dropna=False).items()},
                  "first_started": float(g.started.min()), "last_started": float(g.started.max())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window_min", type=float, default=60.0)
    ap.add_argument("--date", default=time.strftime("%Y%m%d"), help="names the raw store runs/timing_<date>")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--simulate", action="store_true",
                    help="dry run only: simulated systems with invented latency and tokens; no API, no waiting")
    a = ap.parse_args()
    if a.simulate:
        from jevity.simulate import SimulatedModels
        states = pd.read_parquet(ROOT / "data" / "states_eval.parquet")
        recs, syss = first_records(states, a.n), systems()
        sim = SimulatedModels(pd.read_parquet(ROOT / "data" / "cohort.parquet"), states)
        rows, lock = [], threading.Lock()
        t0 = time.time()
        for s in syss:
            run_system(s, recs, None, t0, 0.0, rows, lock, make_client=lambda s_, _store: sim.client(s_, 0))
        df = pd.DataFrame(rows).sort_values(["system", "i"])
        res_dir = ROOT / "results" / "eval"; res_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(res_dir / "timing_sample_calls.parquet", index=False)
        out = {"n_records": len(recs), "window_min": 0, "store": None, "simulated": True,
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "SIMULATED (dry run): invented latencies and tokens", "systems": summarise(df)}
        (res_dir / "timing_sample.json").write_text(json.dumps(out, indent=1))
        print(f"simulated timing sample: {len(recs)} records x {len(syss)} systems -> {res_dir / 'timing_sample.json'}")
        sys.exit(0)
    import importlib.util
    spec = importlib.util.spec_from_file_location("cost_estimate", ROOT / "scripts" / "cost_estimate.py")
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    recs = first_records(pd.read_parquet(ROOT / "data" / "states_eval.parquet"), a.n)
    tok = recs.text.map(estimate_tokens).mean() + ce.PROMPT_OVERHEAD
    syss = systems()
    usd = {s: (len(recs) * tok * MODELS["jev"]["price_per_mtok_input"] / 1e6 if s == "jev"
               else len(recs) * ce.per_call(s, tok, standard=True)["usd"]) for s in syss}
    print(f"timing sample: {len(recs)} evaluation records x {len(syss)} systems ({', '.join(syss)}), standard route, "
          f"one question at a time per system over ~{a.window_min:.0f} min; about ${sum(usd.values()):.2f} "
          + "(" + ", ".join(f"{s} ${v:.2f}" for s, v in usd.items()) + ")")
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    store = RawStore(ROOT / "runs" / f"timing_{a.date}")
    rows, lock = [], threading.Lock()
    t0, gap = time.time() + 2, a.window_min * 60 / len(recs)
    threads = [threading.Thread(target=run_system, args=(s, recs, store, t0, gap, rows, lock), name=s) for s in syss]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    df = pd.DataFrame(rows).sort_values(["system", "i"])
    res_dir = ROOT / "results" / "eval"; res_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in ("p",) if c in df]).to_parquet(res_dir / "timing_sample_calls.parquet", index=False)
    out = {"n_records": len(recs), "window_min": a.window_min, "store": f"runs/timing_{a.date}",
           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "one question at a time per system, systems in the "
           "same window, standard route; latency includes client retries on 429/5xx", "systems": summarise(df)}
    (res_dir / "timing_sample.json").write_text(json.dumps(out, indent=1))
    for s, v in out["systems"].items():
        print(f"{s:14s} n {v['n']:3d} usable {v['usable']:3d} median {v['median_s'] or float('nan'):6.1f} s  "
              f"p90 {v['p90_s'] or float('nan'):6.1f} s  ${v['usd_list_total']:.3f}")
