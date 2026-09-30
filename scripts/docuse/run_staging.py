"""Arm 3, Jev used as documented: build, check and send the requests (config/docuse_staging.yaml,
src/jevity/docuse_staging.py). Only Jev is called. Every live command prints the calls and the projected cost, checks
the spending stops when they are set (config budget: the arm stop on the projection, then the key's usage plus open
batch holds against the project stop) and asks before sending.
  python scripts/docuse/run_staging.py --dry-run            # builds and validates every request; sends nothing
  python scripts/docuse/run_staging.py --full               # 1,000 main + 100 misleading-feature reports
  python scripts/docuse/run_staging.py --repeats            # repeats 1 and 2 on arm 3's 40 repeat reports
  python scripts/docuse/run_staging.py --timing             # arm 3's 100 timing reports, one at a time, over 60 min
  python scripts/docuse/run_staging.py --full --simulate --out DIR   # synthetic Jev answers, no network, under DIR
--variant documented|documented_notes|both (default both) picks the versions for --full and --repeats; the timing
sample runs the documented version only.
Raw responses: runs/arm3_docuse/ (timing: runs/arm3_docuse_timing_<date>/). Tables, rebuilt from the raw store after
every command: results/arm3_docuse/calls.parquet (the columns of results/arm3/calls.parquet) and parts.parquet."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity import docuse_staging as D
from jevity.clients import MODELS, RawStore


def _run_categorical():
    """scripts/run_categorical.py (arm 3's runner) for its spending stop and timing order; imported, not copied."""
    spec = importlib.util.spec_from_file_location("run_categorical", ROOT / "scripts" / "run_categorical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC = _run_categorical()
TIMING_VARIANT = "documented"        # the timing sample runs one version only


def variants_of(arg: str | None) -> tuple[str, ...]:
    return D.VARIANTS if arg in (None, "both") else D.check_variants([arg])


class Paths:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.results = self.root / D.CFG["results"]
        self.runs = self.root / D.CFG["store"]

    def store(self) -> RawStore:
        return RawStore(self.runs)

    def timing_store(self, date: str) -> RawStore:
        return RawStore(self.runs.with_name(f"{self.runs.name}_timing_{date}"))


def timing_calls(items: list[K.Item]) -> list[K.CatCall]:
    """Arm 3's timing reports: run_categorical.timing_items on arm 3's names items (the same 100 as its sample)."""
    a3 = D.source_arm()
    n = int(a3.config["timing"]["n"])
    ids = [i.item_id for i in RC.timing_items(a3, K.load_items(a3), n)]
    return [K.CatCall("jev", i, 0, TIMING_VARIANT) for i in ids]


def projection(arm, items: list[K.Item], calls: list[K.CatCall]) -> pd.DataFrame:
    """Jev's projected input tokens (the request body, 4 characters per token) at list price; output is free."""
    by = {(i.item_id, i.variant): i for i in items}
    cl = D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm)
    tin = sum(D.request_tokens(cl.request(by[(c.item_id, c.variant)])) for c in calls)
    return pd.DataFrame([{"system": "jev", "calls": len(calls), "route": "standard", "tokens_in": tin, "tokens_out": 0,
                          "usd": tin * MODELS["jev"]["price_per_mtok_input"] / 1e6}])


def dry_run(root: Path = ROOT, variants=None) -> dict:
    """Every request of the run (full and repeats for the selected variants, default both; timing, documented) built
    and validated against its version's template; nothing sent, nothing stored. Also lists every sent example found in
    a report (D.example_overlaps), for review; that list does not make a request invalid."""
    arm, items = D.arm_spec(), D.load_items()
    vs = variants_of(None) if variants is None else D.check_variants(variants)
    by = {(i.item_id, i.variant): i for i in items}
    store = Paths(root).store()
    sets = {"full": D.plan("full", items, variants=vs), "repeats": D.plan("repeats", items, variants=vs),
            "timing": timing_calls(items)}
    problems, tokens, seen, per_variant = {}, {}, set(), {}
    for name, calls in sets.items():
        tok = 0
        for c in calls:
            cl = D.DocuseJevClient(store, arm, c.repeat)
            it = by[(c.item_id, c.variant)]
            req = cl.request(it)
            bad = D.validate_request(req, it)
            if bad:
                problems[f"{name}/{c.variant}/{c.item_id}/{c.repeat}"] = bad
            key = (name == "timing", K._sha(req))
            if key in seen:
                problems[f"{name}/{c.variant}/{c.item_id}/{c.repeat}"] = ["duplicate request"]
            seen.add(key)
            t = D.request_tokens(req)
            tok += t
            pv = per_variant.setdefault(f"{name}/{c.variant}", {"requests": 0, "tokens_in": 0})
            pv["requests"] += 1
            pv["tokens_in"] += t
        tokens[name] = tok
    price = MODELS["jev"]["price_per_mtok_input"]
    total = sum(tokens.values())
    first = D.seeded_ids(items)[0]
    out = {"variants": list(vs), "requests": {k: len(v) for k, v in sets.items()},
           "requests_total": sum(len(v) for v in sets.values()), "by_set_and_variant": per_variant,
           "invalid": len(problems), "problems": dict(list(problems.items())[:20]),
           "tokens_in": tokens, "tokens_in_total": total,
           "tokens_in_per_request": {k: tokens[k] / max(len(sets[k]), 1) for k in sets},
           "price_per_mtok_input": price, "usd_list": {k: tokens[k] * price / 1e6 for k in sets},
           "usd_list_total": total * price / 1e6, "config_sha256": D.config_sha256(),
           "model": MODELS["jev"]["slug"], "transport": MODELS["jev"]["transport"],
           "first_report_in_order": first,
           "examples_found_in_reports": D.example_overlaps([i.state for i in items if i.variant == D.VARIANTS[0]]),
           "example_request": {v: D.DocuseJevClient(store, arm).request(by[(first, v)]) for v in vs}}
    return out


def spending_stop(arm, proj: pd.DataFrame) -> None:
    """The shared runners' stops: run_categorical.check_budget on the projection (budget.arm_stop_usd; null: no stop),
    then the key's usage plus every open batch hold plus this projection against budget.project_stop_usd (null: no
    stop) and the balance (categorical.budget_check)."""
    usd = RC.check_budget(arm, proj)
    cfg = arm.config["budget"]
    holds = K.open_holds(K.manifest_files(cfg.get("manifest_roots") or []))
    chk = K.budget_check(K.openrouter_account(), holds, usd, cfg, own=arm.name)
    stop = f"stop ${chk['stop']:,.0f}" if chk["stop"] is not None else "no project stop set"
    print(f"key usage ${chk['key_usage']:,.2f} + open holds ${chk['held']:,.2f} + this run ${usd:,.4f} = "
          f"${chk['total']:,.2f} ({stop}); balance ${chk['balance']:,.2f}")
    for w in chk["warnings"]:
        print("WARNING", w)
    if not chk["ok"]:
        raise SystemExit("spending stop: " + "; ".join(chk["reasons"]))


def rebuild_tables(paths: Paths, arm, items: list[K.Item]) -> pd.DataFrame:
    """calls.parquet and parts.parquet from the raw store: every planned call (full and repeats, both variants) with a
    stored answer."""
    df, parts = D.calls_table(paths.store(), arm, items, D.all_planned(items))
    paths.results.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.results / "calls.parquet", index=False)
    parts.to_parquet(paths.results / "parts.parquet", index=False)
    return df


def write_design(paths: Paths, arm, items) -> None:
    paths.results.mkdir(parents=True, exist_ok=True)
    first = D.seeded_ids(items)[0]
    by = {(i.item_id, i.variant): i for i in items}
    d = {"arm": arm.name, "written": datetime.datetime.now().isoformat(timespec="seconds"),
         "config": "config/docuse_staging.yaml", "config_sha256": D.config_sha256(),
         "items": {k: str(D.CFG["items"][k]) for k in ("main", "confuser")},
         "items_sha256": {k: K.file_sha256(ROOT / D.CFG["items"][k]) for k in ("main", "confuser")},
         "n_items": len(items), "repeat_items": D.repeat_ids(), "n_repeats": D.n_repeats(),
         "seed": int(D.CFG["seed"]), "variants": list(D.VARIANTS), "timing_variant": TIMING_VARIANT,
         "templates": {v: D.template(v) for v in D.VARIANTS},
         "jev_request_example": {v: D.DocuseJevClient(paths.store(), arm).request(by[(first, v)]) for v in D.VARIANTS}}
    (paths.results / "design.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def run_calls(mode: str, paths: Paths, yes: bool = False, workers: int | None = None, sender=None,
              variants=None) -> pd.DataFrame | None:
    arm, items = D.arm_spec(), D.load_items()
    vs = variants_of(None) if variants is None else D.check_variants(variants)
    calls = D.plan(mode, items, variants=vs)
    live = sender is None
    print(f"{arm.name} {mode}: {len(calls)} Jev calls ({len({c.item_id for c in calls})} reports; "
          f"variants {', '.join(vs)})")
    if live:
        K.load_keys(arm)
        spending_stop(arm, projection(arm, items, calls))
        if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
            return None
    write_design(paths, arm, items)
    factory = D.make_factory(arm, paths.store(), sender=sender)
    done = K.execute(calls, items, factory, None, workers=int(workers or D.CFG["workers"]), arm_name=arm.name)
    print(done.groupby("parse", dropna=False).size().to_string())
    err = done[done.error.notna()]
    if len(err):
        print(f"WARNING {len(err)} calls failed (stored as unusable; run the command again to retry):",
              err.error.str[:120].value_counts().head(5).to_dict())
    df = rebuild_tables(paths, arm, items)
    print(f"calls.parquet: {len(df)} rows ({df.valid.sum()} usable)")
    return df


def run_timing(paths: Paths, date: str | None = None, window_min: float | None = None, yes: bool = False,
               sender=None) -> pd.DataFrame | None:
    """As run_categorical.cmd_timing (arm 3's sample): the same 100 reports, one at a time on the standard route,
    question i starting no earlier than start + i x window / n; its own raw store; never enters the answers' analysis."""
    arm, items = D.arm_spec(), D.load_items([TIMING_VARIANT])
    by = {i.item_id: i for i in items}
    calls = timing_calls(items)
    tcfg = D.source_arm().config["timing"]
    window_min = float(tcfg["window_min"] if window_min is None else window_min)
    if sender is None:
        K.load_keys(arm)
        spending_stop(arm, projection(arm, items, calls))
        if not yes and input(f"Timing: {len(calls)} reports over {window_min} min. Proceed? [y/N] ").strip().lower() != "y":
            return None
    date = date or datetime.date.today().strftime("%Y%m%d")
    cl = D.make_factory(arm, paths.timing_store(date), sender=sender)("jev", 0)
    gap, t0, rows = window_min * 60 / max(len(calls), 1), time.time(), []
    for i, c in enumerate(calls):
        wait = t0 + i * gap - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.time()
        try:
            r = cl.answer(by[c.item_id])
        except Exception as e:
            r = K.empty_row(repr(e)[:300])
        rows.append({"system": "jev", "i": i, "item_id": c.item_id, "started": start, "wall_s": time.time() - start,
                     **{k: r.get(k) for k in ("latency_s", "valid", "provider", "model_reported", "route", "tokens_in",
                                              "tokens_out", "usd_reported", "error")}})
    df = pd.DataFrame(rows).assign(window_start=t0, variant=TIMING_VARIANT)
    paths.results.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.results / "timing_sample.parquet", index=False)
    store = Path(paths.timing_store(date).run_dir)
    summ = {"n": len(calls), "variant": TIMING_VARIANT, "window_min": window_min, "date": date,
            "store": store.relative_to(paths.root).as_posix() if store.is_relative_to(paths.root) else str(store),
            "systems": KA.timing_summary(df)}
    (paths.results / "timing_sample.json").write_text(json.dumps(KA.clean(summ), indent=1), encoding="utf-8")
    print(json.dumps(KA.clean(summ["systems"]), indent=1))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--repeats", action="store_true")
    g.add_argument("--timing", action="store_true")
    ap.add_argument("--variant", choices=["documented", "documented_notes", "both"], default="both",
                    help="versions for --dry-run, --full, --repeats (default both); --timing: documented only")
    ap.add_argument("--simulate", action="store_true", help="synthetic Jev answers, no network (needs --out)")
    ap.add_argument("--out", type=Path, default=None, help="root for runs/ and results/ (default: the repository)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--window_min", type=float, default=None)
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    if a.simulate and a.out is None:
        raise SystemExit("--simulate writes under --out only, never into the repository's runs/ or results/")
    paths = Paths(a.out or ROOT)
    sender = D.synthetic_sender(D.truth_parts_of_frames(), 0) if a.simulate else None
    vs = variants_of(a.variant)
    if a.dry_run:
        out = dry_run(paths.root, vs)
        ex = out.pop("example_request")
        print(json.dumps(out, indent=1))
        for v, req in ex.items():
            print(f"example request, {v} (first report in the seeded order):")
            print(json.dumps(req, indent=1, ensure_ascii=False)[:6000])
        if out["examples_found_in_reports"]:
            print(f"NOTE {len(out['examples_found_in_reports'])} sent examples occur in a report (listed above); "
                  "they do not make a request invalid")
        if out["invalid"]:
            raise SystemExit(f"{out['invalid']} invalid requests")
        return out
    if a.timing:
        if a.variant not in ("both", TIMING_VARIANT):
            raise SystemExit(f"the timing sample runs the {TIMING_VARIANT} version only")
        return run_timing(paths, a.date, 0.0 if a.simulate and a.window_min is None else a.window_min, a.yes, sender)
    mode = "full" if a.full else "repeats"
    return run_calls(mode, paths, a.yes, a.workers, sender, vs)


if __name__ == "__main__":
    main()
