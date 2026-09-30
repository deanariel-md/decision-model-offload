"""Categorical arms (arm 2 multiple choice, arm 3 ordered levels): plan, calls, batch route, timing sample, analysis,
tables. The arm's settings are in config/<arm>.yaml; the code in src/jevity/categorical*.py. Every live command prints
the planned calls and the projected cost and asks before spending; keys come from the .env file named by config
keys.env_file. Every batch submission first checks the key's usage plus every open batch's worst-case hold (other
runs' included) against the project stop (config budget.project_stop_usd, when set) and the balance.
  python scripts/run_categorical.py arm2 plan                              # counts and projected cost; no call
  python scripts/run_categorical.py arm2 batch submit                      # GPT, Opus, Gemini, Luna, Flash-Lite
  python scripts/run_categorical.py arm2 batch collect                     # repeat until "waiting 0"
  python scripts/run_categorical.py arm2 run                               # Jev and the synchronous families; batch
                                                                           # answers read from the raw cache
  python scripts/run_categorical.py arm2 timing [--n 100 --window_min 60]  # one at a time, interleaved, same hour
  python scripts/run_categorical.py arm2 analyze
  python scripts/run_categorical.py arm2 tables                            # supplement tables from analysis*.json
  python scripts/run_categorical.py arm2 dryrun [--out DIR]                # simulated systems end to end; no network
Writes results/<arm>/{design.json, calls.parquet, timing_sample.*, analysis*.json, options*.json, table_*} and raw
replies under runs/<arm>/ (timing: runs/<arm>_timing_<date>/)."""
from __future__ import annotations

import argparse, contextlib, datetime, json, os, shutil, sys, threading, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity.clients import BATCH_MISSING, MODELS, RawStore, estimate_tokens, split_system, system_tier


class Paths:
    def __init__(self, arm: K.ArmSpec, root: Path = ROOT, suffix: str = ""):
        self.root = Path(root)
        self.results = self.root / "results" / f"{arm.name}{suffix}"
        self.runs = self.root / "runs" / f"{arm.name}{suffix}"

    def store(self) -> RawStore:
        return RawStore(self.runs)

    def calls(self) -> Path:
        return self.results / "calls.parquet"


# ------------------------------------------------------------------------------------------------ design and cost
def design(arm: K.ArmSpec, *, systems: list[str] | None = None):
    """(items, systems, repeat item ids, planned calls) of the arm; `systems` keeps only those systems."""
    items = K.load_items(arm)
    allsys = K.arm_systems(arm)
    sys_ = [s for s in allsys if systems is None or s in systems]
    rep = K.repeat_ids(arm)
    calls = K.plan_calls(items, sys_, rep, int(arm.config["repeats"]["n_repeats"]), int(arm.config["seed"]))
    return items, sys_, rep, calls


def uses_batch(arm: K.ArmSpec, system: str) -> bool:
    b = MODELS.get("batch") or {}
    return system != "jev" and arm.batch and bool(b.get("enabled")) and split_system(system)[0] in (b.get("families") or {})


def project_cost(arm: K.ArmSpec, items: list[K.Item], calls: list[K.CatCall], standard: bool = False) -> pd.DataFrame:
    """Per system: calls, input tokens estimated from the request text (4 characters per token), output tokens assumed
    per call from config budget.tokens_out, priced by route."""
    by = {(it.item_id, it.variant): it for it in items}
    tout = arm.config["budget"]["tokens_out"]
    rows = []
    present = {c.system for c in calls}
    for s in [x for x in K.arm_systems(arm) if x in present] + sorted(present - set(K.arm_systems(arm))):
        cs = [c for c in calls if c.system == s]
        if s == "jev":
            tin = sum(estimate_tokens(by[(c.item_id, c.variant)].state + json.dumps(K.jev_question(arm, by[(c.item_id, c.variant)])))
                      for c in cs)
            usd = tin * MODELS["jev"]["price_per_mtok_input"] / 1e6
            rows.append({"system": s, "calls": len(cs), "route": "standard", "tokens_in": tin, "tokens_out": 0, "usd": usd})
            continue
        fam = split_system(s)[0]
        tin = sum(estimate_tokens(K.system_text(arm) + K.user_text(arm, by[(c.item_id, c.variant)])) for c in cs)
        to = len(cs) * int(tout.get(fam, tout["default"]))
        batch = uses_batch(arm, s) and not standard
        f = MODELS["families"][fam]
        p = MODELS["batch"]["families"][fam] if batch else f
        rows.append({"system": s, "calls": len(cs), "route": "batch" if batch else "standard", "tokens_in": tin,
                     "tokens_out": to, "usd": (tin * p["price_in"] + to * p["price_out"]) / 1e6})
    return pd.DataFrame(rows)


def check_budget(arm: K.ArmSpec, proj: pd.DataFrame) -> float:
    """Prints the projection; refuses it above config budget.arm_stop_usd (null: no stop)."""
    tot = float(proj.usd.sum())
    stop = arm.config["budget"].get("arm_stop_usd")
    print(proj.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"projected {arm.name}: ${tot:,.2f} " + (f"(stop above ${stop})" if stop is not None else "(no stop set)"))
    if stop is not None and tot > float(stop):
        raise SystemExit("projection above budget.arm_stop_usd: not run")
    return tot


def write_design(paths: Paths, arm: K.ArmSpec, items, systems, rep, calls) -> dict:
    paths.results.mkdir(parents=True, exist_ok=True)
    items_file = K.ROOT / arm.config["items"]["main"]
    it0 = items[0]
    d = {"arm": arm.name, "set": "main", "kind": arm.kind, "written": datetime.datetime.now().isoformat(timespec="seconds"),
         "items_file": str(arm.config["items"]["main"]), "items_sha256": K.file_sha256(items_file),
         "n_items": len({i.item_id for i in items}), "variants": sorted({i.variant for i in items}),
         "n_items_by_variant": pd.Series([i.variant for i in items]).value_counts().sort_index().to_dict(),
         "systems": {s: ("jev" if s == "jev" else system_tier(s)) for s in systems},
         "repeat_items": rep, "calls_per_system": pd.Series([c.system for c in calls]).value_counts().to_dict(),
         "seed": int(arm.config["seed"]), "route_batch": arm.batch,
         "chatbot_system_prompt": K.system_text(arm), "chatbot_user_example": K.user_text(arm, it0),
         "chatbot_schema": K.reply_schema(arm, it0.presented), "jev_question_example": K.jev_question(arm, it0)}
    (paths.results / "design.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return d


def worker_plan(arm: K.ArmSpec, systems: list[str], workers: int = 16, hf_workers: int = 8) -> dict:
    """Parallel requests per system: the command-line default, hf_workers for the Hugging Face router, and never more
    than the arm's route.workers cap."""
    fams = MODELS["families"]
    w = {"default": workers, **{s: hf_workers for s in systems
                                if (fams.get(split_system(s)[0]) or {}).get("transport") == "hf_router"}}
    for s, cap in ((arm.config.get("route") or {}).get("workers") or {}).items():
        w[s] = min(int(cap), w.get(s, w["default"]))
    return w


# ------------------------------------------------------------------------------------------------ commands
def cmd_plan(arm, *, systems=None, standard=False):
    items, sys_, rep, calls = design(arm, systems=systems)
    print(f"{arm.name}: {len({i.item_id for i in items})} items, systems {sys_}")
    print("calls by system:", pd.Series([c.system for c in calls]).value_counts().to_dict())
    return check_budget(arm, project_cost(arm, items, calls, standard))


def cmd_run(arm, paths: Paths, *, systems=None, yes=False, workers=16, hf_workers=8, sender=None):
    items, sys_, rep, calls = design(arm, systems=systems)
    simulate = sender is not None
    if not simulate:
        K.load_keys(arm)
    print(f"{arm.name}: {len(calls)} calls", pd.Series([c.system for c in calls]).value_counts().to_dict())
    if not simulate:
        check_budget(arm, project_cost(arm, items, calls))
        if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
            return None
    part = systems is not None and set(sys_) != set(K.arm_systems(arm))
    if not part:                                  # a run of some systems only (two processes at once, one per provider)
        write_design(paths, arm, items, sys_, rep, calls)       # writes its own table; the full run rebuilds
    store = paths.store()                                        # calls.parquet
    factory = K.make_factory(arm, store, sender=sender, standard=simulate)
    out = paths.results / f"calls_part_{'_'.join(sys_)}.parquet" if part else paths.calls()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = K.execute(calls, items, factory, out, workers=worker_plan(arm, sys_, workers, hf_workers)
                   if not simulate else 2, arm_name=arm.name)
    print(df.groupby("system")["valid"].agg(["size", "mean"]).to_string())
    miss = df[df.error == BATCH_MISSING]
    if len(miss):
        print(f"WARNING {len(miss)} batch-route calls have no collected answer ({miss.system.value_counts().to_dict()}): "
              "run batch collect / batch submit, then run again before analysis.")
    return df


def cmd_batch(arm, paths: Paths, action, args, yes=False, post=None, get=None, account=None, systems=None):
    """submit takes --models to send some batch systems only, so a submission can go in pieces whose holds fit the
    balance and the key limit; the design file always lists every system."""
    from jevity import batch as BT
    store = paths.store()
    if action in ("status", "collect") and get is None:
        K.load_keys(arm)
    if action == "status":
        rows = BT.status(store, get=get); print(pd.DataFrame(rows).to_string()); return rows
    if action == "adopt":
        return BT.adopt(store, *args)
    if action == "drop":
        return BT.drop(store, *args)
    if action == "collect":
        t = BT.collect(store, get=get)
        print(f"written {t['written']}, failed {t['failed']} (resubmit with batch submit), waiting {t['waiting_batches']}, "
              f"errors {t['errors']} (retried next collect)")
        return t
    if post is None or account is None:          # tests inject both; a live submission reads the keys
        K.load_keys(arm)
    items, sys_, rep, calls = design(arm)
    if systems:
        off = sorted(set(systems) - {s for s in sys_ if uses_batch(arm, s)})
        if off:
            raise SystemExit(f"not batch-route systems of {arm.name}: {off}")
    chosen = [c for c in calls if not systems or c.system in systems]
    todo = K.pending_batch(chosen, items, arm, store)
    print(f"batch route: {len(todo)} requests not yet answered", pd.Series([f for f, _, _ in todo]).value_counts().to_dict())
    check_budget(arm, project_cost(arm, items, [c for c in chosen if uses_batch(arm, c.system)]))
    if not yes and input("Submit? [y/N] ").strip().lower() != "y":
        return []
    write_design(paths, arm, items, sys_, rep, calls)
    return K.submit_batch(todo, store, arm, post=post, account=account)


def timing_items(arm, items: list[K.Item], n: int) -> list[K.Item]:
    """The first n items of the primary variant in the seeded block order."""
    v = arm.config.get("primary_variant", "main")
    pool = {it.item_id: it for it in items if it.variant == v}
    order = [c.item_id for c in K.block_order([K.CatCall("x", i) for i in pool], int(arm.config["seed"]))]
    return [pool[i] for i in order[:n]]


def cmd_timing(arm, paths: Paths, n=None, window_min=None, date=None, yes=False, sender=None, systems=None):
    """Every system answers the same n items on the standard route, one question at a time, question i starting no
    earlier than start + i x window/n, all systems of one command in the same window. Own raw store; never enters the
    analysis of answers. With --models a later window adds its systems to the table (replacing theirs, keeping the
    others'); each row keeps its window's start, and the JSON lists each system's. Config timing.hold: no live timing
    sample for the arm."""
    tcfg = arm.config["timing"]
    if sender is None and tcfg.get("hold"):
        raise SystemExit(f"{arm.name}: timing sample on hold ({tcfg['hold']}); remove timing.hold in the config first")
    n, window_min = int(n or tcfg["n"]), float(tcfg["window_min"] if window_min is None else window_min)
    items, sys_, *_ = design(arm, systems=systems)
    recs = timing_items(arm, items, n)
    if sender is None:
        K.load_keys(arm)
        calls = [K.CatCall(s, it.item_id, 0, it.variant) for s in sys_ for it in recs]
        check_budget(arm, project_cost(arm, items, calls, standard=True))
        if not yes and input(f"Timing sample: {len(recs)} items x {len(sys_)} systems over {window_min} min. Proceed? [y/N] ").strip().lower() != "y":
            return None
    date = date or datetime.date.today().strftime("%Y%m%d")
    store = RawStore(paths.runs.with_name(f"{paths.runs.name}_timing_{date}"))
    factory = K.make_factory(arm, store, sender=sender, standard=True)
    gap, t0, rows, lock = window_min * 60 / max(len(recs), 1), time.time(), [], threading.Lock()

    def run_system(s):
        cl = factory(s, 0)
        for i, it in enumerate(recs):
            wait = t0 + i * gap - time.time()
            if wait > 0:
                time.sleep(wait)
            start = time.time()
            try:
                r = cl.answer(it)
            except Exception as e:
                r = K.empty_row(repr(e)[:300])
            with lock:
                rows.append({"system": s, "i": i, "item_id": it.item_id, "started": start,
                             "wall_s": time.time() - start, **{k: r.get(k) for k in (
                                 "latency_s", "valid", "provider", "model_reported", "route", "tokens_in",
                                 "tokens_out", "usd_reported", "error")}})

    threads = [threading.Thread(target=run_system, args=(s,), name=s) for s in sys_]
    [t.start() for t in threads]
    [t.join() for t in threads]
    df = pd.DataFrame(rows).assign(window_start=t0)
    paths.results.mkdir(parents=True, exist_ok=True)
    tp = paths.results / "timing_sample.parquet"
    with table_lock(paths.results / "timing_sample.lock"):   # two processes in one window (Muse on its own key)
        if systems and tp.exists():
            old = pd.read_parquet(tp)
            df = pd.concat([old[~old.system.isin(sys_)], df], ignore_index=True)
        df = df.sort_values(["system", "i"]).reset_index(drop=True)
        df.to_parquet(tp, index=False)
        starts = df.groupby("system").window_start.first().dropna()
        summ = {"n": len(recs), "window_min": window_min, "date": date,
                "windows": {s: datetime.datetime.fromtimestamp(float(w)).isoformat(timespec="seconds")
                            for s, w in starts.items()},
                "systems": KA.timing_summary(df)}
        (paths.results / "timing_sample.json").write_text(json.dumps(KA.clean(summ), indent=1), encoding="utf-8")
    print(json.dumps(KA.clean(summ["systems"]), indent=1))
    return df


@contextlib.contextmanager
def table_lock(path: Path, poll_s: float = 0.5, timeout_s: float = 600):
    """An exclusive lock file around a read-modify-write of a shared results table."""
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"{path} held for over {timeout_s:.0f} s; remove it if no timing process is running")
            time.sleep(poll_s)
    try:
        yield
    finally:
        os.close(fd)
        os.remove(path)


def items_frame(items: list[K.Item], variant: str, arm=None) -> pd.DataFrame:
    """One row per item of the variant; with `arm`, the item-file columns the post hoc analyses name (config
    analysis.post_hoc: rates_by_group.label_column, by_category) joined by item id."""
    f = pd.DataFrame([{"item_id": i.item_id, "truth": i.truth, "stratum": i.stratum, "group": i.group, "state": i.state}
                      for i in items if i.variant == variant]).sort_values("item_id").reset_index(drop=True)
    ph = ((arm.config.get("analysis") or {}).get("post_hoc") or {}) if arm is not None else {}
    cols = {c for v in (ph.get("by_category") or {}).values() for c in v}
    cols |= {c for c in [(ph.get("rates_by_group") or {}).get("label_column")] if c}
    if cols:
        ic = arm.config["items"]
        src = pd.read_csv(K.ROOT / ic["main"], dtype=str, keep_default_na=False)
        cols = sorted(c for c in cols if c in src and c not in f)
        f = f.merge(src[[ic["id_column"], *cols]].rename(columns={ic["id_column"]: "item_id"}), on="item_id", how="left")
    return f


def subset_ids(arm) -> dict[str, set[str]]:
    """Item subsets (config analysis.subsets: name -> {column, equals}), from the items file."""
    subs = (arm.config.get("analysis") or {}).get("subsets") or {}
    if not subs:
        return {}
    ic = arm.config["items"]
    df = pd.read_csv(K.ROOT / ic["main"], dtype=str, keep_default_na=False)
    return {n: set(df.loc[df[s["column"]].astype(str) == str(s["equals"]), ic["id_column"]]) for n, s in subs.items()}


def extra_analyses(arm, paths: Paths, calls, items, sys_, tiers, rep, timing, variants, allow_incomplete) -> list[Path]:
    """Arm 2's second item set: each variant on each item subset of config analysis.subsets
    (analysis_<variant>_<subset>.json), and the option-count contrast of two variants on every item and on each subset
    (options.json, options_<subset>.json)."""
    made = []
    primary = arm.config.get("primary_variant", "main")
    subs = subset_ids(arm)
    for v in variants:
        for name, ids in subs.items():
            f = items_frame(items, v, arm)
            res = KA.analyze(calls, f[f.item_id.isin(ids)].reset_index(drop=True), arm.config, arm.labels, sys_, tiers,
                             MODELS, [r for r in rep if r in ids], timing, v)
            res["subset"] = {"name": name, **arm.config["analysis"]["subsets"][name], "n_items": res["n_items"]}
            res["variant"] = f"{v}, {name} subset"
            miss = {s: c["missing"] for s, c in res["completeness"].items() if c["missing"]}
            res["missing_treated_as_unusable"] = miss
            made.append(paths.results / f"analysis_{v}_{name}.json")
            made[-1].write_text(json.dumps(res, indent=1), encoding="utf-8")
    pair = (arm.config.get("analysis") or {}).get("option_count_contrast")
    if pair:
        n = {v: {it.item_id: len(it.presented) for it in items if it.variant == v} for v in pair}
        for name, ids in {"": None, **subs}.items():
            frames = {v: items_frame(items, v) for v in pair}
            if ids is not None:
                frames = {v: f[f.item_id.isin(ids)].reset_index(drop=True) for v, f in frames.items()}
            counts = {v: frames[v].item_id.map(n[v]).to_numpy(int) for v in pair}
            res = KA.option_count_analysis(calls, frames, counts, arm.config, arm.labels, sys_, MODELS)
            res["subset"] = name or None
            made.append(paths.results / (f"options_{name}.json" if name else "options.json"))
            made[-1].write_text(json.dumps(res, indent=1), encoding="utf-8")
        o = json.loads((paths.results / "options.json").read_text(encoding="utf-8"))
        print(f"  options (longer minus shorter list, same {o['n_items']:,} questions):")
        for s, r in o["systems"].items():
            print(f"    {s:12s} {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
    return made


def cmd_analyze(arm, paths: Paths, *, allow_incomplete=False) -> dict:
    calls = pd.read_parquet(paths.calls())
    items, sys_, rep, *_ = design(arm)
    tiers = {s: ("jev" if s == "jev" else system_tier(s)) for s in sys_}
    tp = paths.results / "timing_sample.parquet"
    timing = pd.read_parquet(tp) if tp.exists() else None
    variants = sorted({i.variant for i in items})
    primary = arm.config.get("primary_variant", "main")
    results = {}
    for v in variants:
        res = KA.analyze(calls, items_frame(items, v, arm), arm.config, arm.labels, sys_, tiers, MODELS, rep, timing, v)
        miss = {s: c["missing"] for s, c in res["completeness"].items() if c["missing"]}
        if miss and not allow_incomplete:
            raise SystemExit(f"{v}: answers missing {miss} (uncollected batches or calls not run); rerun, or "
                             "--allow_incomplete to analyse them as unusable")
        res["missing_treated_as_unusable"] = miss
        # any reply fabricated by the simulator (raw store meta "simulated") marks the whole analysis as a dry run
        res["simulated"] = bool("simulated" in calls and calls.simulated.fillna(False).astype(bool).any())
        results[v] = res
        if len(variants) > 1:
            (paths.results / f"analysis_{v}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    out = results[primary]
    (paths.results / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    extra_analyses(arm, paths, calls, items, sys_, tiers, rep, timing, variants, allow_incomplete)
    pm = out["primary_metric"]
    print(f"{arm.name} ({primary}): {pm}")
    for s, r in out["systems"].items():
        print(f"  {s:12s} {r[pm]:.3f} [{r[pm + '_ci'][0]:.3f}, {r[pm + '_ci'][1]:.3f}]  unusable {r['unusable']}/"
              f"{r['n_items']} ({r['unusable_rate']:.1%})  ${r['cost']['usd_list_per_answer']:.5f}/answer (list)")
    if "style_check" in out:
        st = out["style_check"]
        b = st["baseline"]
        print(f"  {'style only':12s} {b[pm]:.3f} [{b[pm + '_ci'][0]:.3f}, {b[pm + '_ci'][1]:.3f}]  reference: word and "
              f"negation counts, leave one recommendation out; {st['n_style_misleading']} style-misleading items")
    for s, c in out["comparisons"].items():
        v = out["acceptable_cheaper"][s]
        hb = (f"Holm bounds {c['holm_lower_bound']:+.3f} / {c['holm_upper_bound']:+.3f}" if c["confirmatory"]
              else f"bounds {c['lower_bound_unadjusted']:+.3f} / {c['upper_bound_unadjusted']:+.3f} (unadjusted, descriptive)")
        print(f"  jev - {s:12s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  {hb}  {c['outcome']}  "
              f"acceptable cheaper: {v['verdict']}")
    hf = out.get("hybrid_share")
    if hf:
        print(f"  hybrid: Jev answers {hf['comparisons'][next(iter(hf['comparisons']))]['n_jev']} of "
              f"{out['n_items']} items (its highest top probability), the chatbot the rest")
        for s, c in hf["comparisons"].items():
            k = c["cost_list"]
            print(f"  hybrid - {s:10s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  Holm bounds "
                  f"{c['holm_lower_bound']:+.3f} / {c['holm_upper_bound']:+.3f}  {c['outcome']}  list cost per 1,000 "
                  f"items ${k['hybrid_usd_per_1000_items']:.2f} vs ${k['chatbot_usd_per_1000_items']:.2f}")
    return out


# ------------------------------------------------------------------------------------------------ supplement tables
def cmd_tables(arm, paths: Paths) -> list[Path]:
    """Supplement tables from the analysis JSON of each version: repeatability, cost and speed, value (and the result,
    post hoc and option-count tables where the arm has them). Numbers as CSV (intervals in three columns) and formatted
    as Markdown, under results/<arm>/."""
    primary = arm.config.get("primary_variant", "main")
    files = {"": paths.results / "analysis.json"}
    if not files[""].exists():
        raise SystemExit(f"{files['']} does not exist: run analyze first")
    files |= {f"_{f.stem[len('analysis_'):]}": f for f in sorted(paths.results.glob("analysis_*.json"))
              if f.stem != f"analysis_{primary}"}
    made = []
    more = bool((arm.config.get("analysis") or {}).get("result_tables"))

    def write(name, tab):
        base = paths.results / f"table_{name}"
        KA.table_frame(tab).to_csv(base.with_suffix(".csv"), index=False, lineterminator="\n")
        base.with_suffix(".md").write_text(KA.table_markdown(tab), encoding="utf-8")
        made.extend([base.with_suffix(".csv"), base.with_suffix(".md")])
    for suffix, f in files.items():
        res = json.loads(f.read_text(encoding="utf-8"))
        tabs = {**(KA.result_tables(res) if more else {}), **KA.supplement_tables(res), **KA.post_hoc_tables(res)}
        for name, tab in tabs.items():
            write(f"{name}{suffix}", tab)
    for f in sorted(paths.results.glob("options*.json")):
        write(f.stem, KA.option_count_table(json.loads(f.read_text(encoding="utf-8"))))
    return made


# ------------------------------------------------------------------------------------------------ dry run
def cmd_dryrun(arm, out_root: Path, n_timing: int = 20) -> dict:
    """Simulated systems through the real clients, raw store and parsers, then timing (no waiting), analysis and
    tables, all under out_root. No network: the sender replaces the transport."""
    from jevity.categorical_sim import SimulatedSender
    paths = Paths(arm, out_root)
    if paths.runs.exists():
        shutil.rmtree(paths.runs)
    sender = SimulatedSender(arm, K.load_items(arm))
    cmd_run(arm, paths, sender=sender)
    cmd_timing(arm, paths, n=n_timing, window_min=0, date="dryrun", sender=sender)
    out = cmd_analyze(arm, paths)
    cmd_tables(arm, paths)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("command", choices=["plan", "run", "batch", "timing", "analyze", "tables", "dryrun"])
    ap.add_argument("args", nargs="*", help="batch: submit|collect|status|adopt <key> <id>|drop <key>")
    ap.add_argument("--models", default=None, help="comma list of systems")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hf_workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--window_min", type=float, default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--allow_incomplete", action="store_true")
    ap.add_argument("--out", default=None, help="dryrun: output root (default results|runs/<arm>_dryrun under the repo)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    arm = K.load_arm(a.arm)
    systems = a.models.split(",") if a.models else None
    paths = Paths(arm)
    if a.command == "plan":
        cmd_plan(arm, systems=systems)
    elif a.command == "run":
        cmd_run(arm, paths, systems=systems, yes=a.yes, workers=a.workers, hf_workers=a.hf_workers)
    elif a.command == "batch":
        cmd_batch(arm, paths, a.args[0], a.args[1:], a.yes, systems=systems)
    elif a.command == "timing":
        cmd_timing(arm, paths, a.n, a.window_min, a.date, a.yes, systems=systems)
    elif a.command == "analyze":
        cmd_analyze(arm, paths, allow_incomplete=a.allow_incomplete)
    elif a.command == "tables":
        print("\n".join(map(str, cmd_tables(arm, paths))))
    elif a.command == "dryrun":
        cmd_dryrun(arm, Path(a.out) if a.out else ROOT / "results" / f"{arm.name}_dryrun")
