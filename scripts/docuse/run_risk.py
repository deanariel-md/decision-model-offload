"""Ten-year risk, Jev used as documented: build, check and send the requests (config/docuse_risk.yaml,
src/jevity/docuse_risk.py). One call per record, the original primary question, the record as a JSON object of named
bands in `state`; Jev through clients.JevClient (config/models.yaml slug, allow_fallbacks false), raw answers in
runs/wide_docuse (first write wins, never overwritten).
  python scripts/docuse/run_risk.py --dry-run          # every request built and validated; nothing sent, no key read
  python scripts/docuse/run_risk.py --full             # the 12,910 baseline records of the wide set
  python scripts/docuse/run_risk.py --timing           # 100 evaluation records, one at a time over 60 minutes
  python scripts/docuse/run_risk.py --calls            # rebuild results/wide_docuse/calls.parquet from the store; no send
The live modes print the counts and the projected cost, check the spending stops when they are set (config budget: the
run stop on the projection, then the key's usage plus open batch holds against the project stop), and ask before
sending. Writes results/wide_docuse/{dry_run.json, calls.parquet, run_log.json, timing_sample.*}."""
import argparse, importlib.util, json, sys, threading, time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import docuse_risk as D
from jevity.clients import MODELS, RawStore
from jevity.runner import Call, block_order

RESULTS = ROOT / "results" / D.CFG["name"]
RUNS = ROOT / "runs"
STORE = RUNS / D.CFG["name"]


def first_records(profiles: list[int], n: int) -> list[int]:
    """The first n records of the seeded block order (runner.block_order), as the timing sample takes them."""
    order = [c.profile for c in block_order([Call("x", int(p), "baseline", False, 0, "") for p in profiles])]
    return list(dict.fromkeys(order))[:n]


def timing_module():
    spec = importlib.util.spec_from_file_location("timing_sample", ROOT / "scripts" / "timing_sample.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def eval_timing_profiles(n: int) -> tuple[list[int], str]:
    """The records of the original timing sample: the first n evaluation baselines in block order
    (scripts/timing_sample.py first_records on data/states_eval.parquet); without that file, the evaluation records of
    data/cohort.parquet in the same order."""
    f = ROOT / "data" / "states_eval.parquet"
    if f.exists():
        recs = timing_module().first_records(pd.read_parquet(f), n)
        return [int(p) for p in recs.profile], "data/states_eval.parquet"
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    return first_records(sorted(co.loc[co.split == "eval", "SEQN"].astype(int)), n), "data/cohort.parquet"


def projection(states: pd.DataFrame) -> dict:
    r = D.dry_run(states)
    print(f"requests: {r['n_requests']:,} ({r['n_unique_requests']:,} distinct), invalid {r['n_invalid']}; "
          f"estimated input tokens {r['tokens_estimated_calibrated']:,.0f} (mean {r['mean_tokens_calibrated']:.0f}; "
          f"original run mean {r['original_mean_tokens_reported']:.0f}); list cost ${r['usd_list_estimated']:.4f} at "
          f"${r['price_per_mtok_input']}/M input tokens, output free")
    if r["n_invalid"]:
        raise SystemExit(f"invalid requests: {json.dumps(r['invalid'])[:1000]}")
    return r


def preflight(states: pd.DataFrame, yes: bool) -> dict:
    """Projection, spending stops and the question; returns the projection. Account reads only (no model call)."""
    print(f"config sha256 {D.config_sha256()}")
    r = projection(states)
    b = D.CFG["budget"]
    upper = r["n_requests"] * max(r["mean_tokens_calibrated"], r["original_mean_tokens_reported"]) \
        * r["price_per_mtok_input"] / 1e6
    if b.get("run_stop_usd") is not None and upper > float(b["run_stop_usd"]):
        raise SystemExit(f"projection ${upper:.2f} above budget.run_stop_usd ${b['run_stop_usd']}: not run")
    from jevity import categorical as K
    K.load_keys(SimpleNamespace(name="docuse_risk", config=D.CFG))
    acc = K.openrouter_account()
    holds = K.open_holds(K.manifest_files(b.get("manifest_roots") or [], [STORE]))
    chk = K.budget_check(acc, holds, upper, b, own=str(STORE))
    stop = f"project stop ${chk['stop']:,.0f}" if chk["stop"] is not None else "no project stop set"
    print(f"key usage ${chk['key_usage']:,.2f} + open holds ${chk['held']:,.2f} + this run ${upper:,.2f} = "
          f"${chk['total']:,.2f} ({stop}); balance ${chk['balance']:,.2f}")
    for w in chk["warnings"]:
        print(f"WARNING {w}")
    if not chk["ok"]:
        raise SystemExit("budget check failed: " + "; ".join(chk["reasons"]))
    if not yes and input("Send? [y/N] ").strip().lower() != "y":
        raise SystemExit(0)
    return r


def live(states: pd.DataFrame, out_calls: Path, workers: int, yes: bool, label: str) -> None:
    preflight(states, yes)
    store = RawStore(STORE)
    t0 = time.time()
    stop = D.CFG["budget"].get("run_stop_usd")
    summ = D.run(states, store, RUNS / f"{D.CFG['name']}_progress", None if stop is None else float(stop),
                 workers=workers, chunk_records=int(D.CFG["budget"]["chunk_records"]))
    df = D.calls_table(states, store, out_calls)
    log = RESULTS / "run_log.json"
    hist = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    hist.append({"mode": label, "records": int(len(states)), "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **summ, "calls_file": str(out_calls),
                 "usable": int(df.valid.sum()), "providers": df.provider.value_counts().to_dict(),
                 "models_reported": df.model_reported.value_counts().to_dict()})
    log.write_text(json.dumps(hist, indent=1, default=str), encoding="utf-8")
    print(f"{label}: {len(df)} records, usable {int(df.valid.sum())}; providers {df.provider.value_counts().to_dict()}; "
          f"-> {out_calls}")


def timing(n: int, window_min: float, date: str, yes: bool) -> None:
    prof, src = eval_timing_profiles(n)
    states = D.build_states(prof)
    preflight(states, yes)
    tm = timing_module()
    store = RawStore(RUNS / f"{D.CFG['name']}_timing_{date}")      # own store: every answer is a fresh call
    recs = pd.DataFrame({"profile": states.profile, "text": states.state})
    rows, lock = [], threading.Lock()
    t0, gap = time.time() + 2, window_min * 60 / len(recs)
    tm.run_system("jev", recs, store, t0, gap, rows, lock, make_client=lambda s, st: D.client(st))
    df = pd.DataFrame(rows).sort_values("i")
    df.to_parquet(RESULTS / "timing_sample_calls.parquet", index=False)
    out = {"n_records": len(recs), "records_from": src, "window_min": window_min, "store": f"runs/{D.CFG['name']}_timing_{date}",
           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "Jev used as documented, one question at a time; latency includes client retries on 429/5xx",
           "systems": {"jev_documented": tm.summarise(df.assign(system="jev"))["jev"]}}
    (RESULTS / "timing_sample.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    v = out["systems"]["jev_documented"]
    print(f"timing: n {v['n']} usable {v['usable']} median {v['median_s']} s p90 {v['p90_s']} s ${v['usd_list_total']:.4f}")


def print_distribution(dist: dict) -> None:
    """Records per label of every banded field (dry_run.json band_distribution)."""
    for name, d in dist.items():
        print(f"{d['key']}:")
        for lab, n in d["counts"].items():
            print(f"  {n:6,d}  {lab}")
        if d["missing"]:
            print(f"  {d['missing']:6,d}  (missing)")
        for sex, c in d.get("by_sex", {}).items():
            print(f"    {sex}: " + "; ".join(f"{lab} {n:,}" for lab, n in c.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--timing", action="store_true")
    g.add_argument("--calls", action="store_true", help="rebuild the full calls table from the raw store; no send")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=int(D.CFG["timing"]["n"]))
    ap.add_argument("--window_min", type=float, default=float(D.CFG["timing"]["window_min"]))
    ap.add_argument("--date", default=time.strftime("%Y%m%d"))
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    profiles = D.wide_profiles()
    if a.dry_run:
        st = D.build_states(profiles)
        r = projection(st)
        (RESULTS / "dry_run.json").write_text(json.dumps(D.jsonable(r), indent=1, ensure_ascii=False), encoding="utf-8")
        print_distribution(r["band_distribution"])
        print(f"config sha256 {r['config_sha256']}\nwrote {RESULTS / 'dry_run.json'}")
    elif a.full:
        live(D.build_states(profiles), RESULTS / "calls.parquet", a.workers, a.yes, "full")
    elif a.timing:
        timing(a.n, a.window_min, a.date, a.yes)
    elif a.calls:
        df = D.calls_table(D.build_states(profiles), RawStore(STORE), RESULTS / "calls.parquet")
        print(f"calls table: {len(df)} records, usable {int(df.valid.sum())} -> {RESULTS / 'calls.parquet'}")
