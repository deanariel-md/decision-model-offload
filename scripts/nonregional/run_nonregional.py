"""Non-regional node sites the AJCC notes do not name. The misleading-feature set's 20 type 4 reports put the metastasis
in a separately submitted para-aortic or aortocaval node, the very words of the documented_notes version's
examples. Here 20 type 4 reports carry sites the notes do not name (scripts/nonregional/sites.yaml, each confirmed
non-regional by the CAP colon protocol), read from data/arm3_nonregional_sites/items.csv; the questions, criteria, notes
and examples and the request builders are unchanged.
  python scripts/nonregional/run_nonregional.py --dry-run   # counts and projected cost; sends nothing
  python scripts/nonregional/run_nonregional.py --run --yes [--systems jev,gpt,claude,gemini,glm]
  python scripts/nonregional/run_nonregional.py --run --yes --systems muse
  python scripts/nonregional/run_nonregional.py --analyze   # results/arm3_nonregional_sites/{summary.json, ...}
--run prints the calls and the projected cost, checks the spending stops when they are set (--max_usd on the projection;
config/arm3_confuser.yaml budget: the key's usage plus open batch holds against the project stop) and asks before
sending. --key names another .env variable to send with on OpenRouter (this process only).
Jev: structured input, both versions (documented, documented_notes), the documented run's request builder
(jevity.docuse_staging). Chatbots: the five main chatbots, level names only, the misleading-feature run's arm 3 request
(config/arm3_confuser.yaml, read unchanged; only the items path differs, in memory), on the standard route (the batch
route of that run sends the same body without allow_fallbacks, which the Batch API rejects).
Raw responses: runs/arm3_nonregional_sites/ (first write wins)."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import docuse_staging as D
from jevity.clients import BATCH_MISSING, RawStore

SPEC = yaml.safe_load((HERE / "sites.yaml").read_text(encoding="utf-8"))
SITES = [s["site"] for s in SPEC["sites"]]
SEED = int(SPEC["seed"])
TYPE = 4
DATA = ROOT / "data" / "arm3_nonregional_sites"
ITEMS = DATA / "items.csv"
STORE = ROOT / "runs" / "arm3_nonregional_sites"
RESULTS = ROOT / "results" / "arm3_nonregional_sites"
CHATBOTS = list(D.CFG["analysis"]["confuser_family"])          # gpt, claude, gemini, muse, glm
MAX_USD: float | None = None                                    # stop on the projection (USD; --max_usd); None: no stop


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ requests
def jev_items() -> list[K.Item]:
    df = pd.read_csv(ITEMS, dtype={"report_id": str}, keep_default_na=False)
    return [K.Item(r["report_id"], r["report"].strip(), str(r["level"]), D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                   str(r["level"]), str(r["substage"]), v)
            for v in D.VARIANTS for r in df.to_dict("records")]


def chatbot_arm() -> K.ArmSpec:
    """config/arm3_confuser.yaml as read (prompts, levels, names variant, keys, budget) with this run's items
    file. The arm keeps its name: the reply schema sent to the model is named after it (arm3_confuser_answer), so the
    request is the misleading-feature run's word for word apart from the report."""
    arm = K.load_arm("arm3_confuser")
    cfg = copy.deepcopy(arm.config)
    cfg["items"]["main"] = str(ITEMS.relative_to(ROOT)).replace("\\", "/")
    return dataclasses.replace(arm, config=cfg)


def plan(systems: list[str]) -> tuple[list, list, list, list]:
    """(Jev items, Jev calls, chatbot items, chatbot calls) for the systems asked."""
    ji = jev_items()
    jc = [K.CatCall("jev", i.item_id, 0, i.variant) for i in ji] if "jev" in systems else []
    ci = K.load_items(chatbot_arm())
    cc = [K.CatCall(s, i.item_id, 0, i.variant) for s in CHATBOTS if s in systems for i in ci]
    return ji, jc, ci, cc


def projection(systems: list[str]) -> pd.DataFrame:
    rs = _module("run_staging", ROOT / "scripts" / "docuse" / "run_staging.py")
    rc = _module("run_categorical", ROOT / "scripts" / "run_categorical.py")
    ji, jc, ci, cc = plan(systems)
    parts = []
    if jc:
        parts.append(rs.projection(D.arm_spec(), ji, jc))
    if cc:
        parts.append(rc.project_cost(chatbot_arm(), ci, cc, standard=True))
    return pd.concat(parts, ignore_index=True)


def spending(proj: pd.DataFrame, max_usd: float | None = None) -> dict:
    """Key usage plus open holds plus this projection against config/arm3_confuser.yaml budget.project_stop_usd (null:
    no stop; the balance must cover them either way, categorical.budget_check), and the projection against max_usd
    (default MAX_USD; None: no stop)."""
    max_usd = MAX_USD if max_usd is None else max_usd
    arm = chatbot_arm()
    K.load_keys(arm)
    cfg = arm.config["budget"]
    usd = float(proj.usd.sum())
    holds = K.open_holds(K.manifest_files(cfg.get("manifest_roots") or []))
    chk = K.budget_check(K.openrouter_account(), holds, usd, cfg, own=arm.name)
    over = max_usd is not None and usd > max_usd
    return {"projection_usd": usd, "key_usage": chk["key_usage"], "open_holds": chk["held"], "total": chk["total"],
            "stop": chk["stop"], "max_usd": max_usd, "ok": bool(chk["ok"] and not over),
            "reasons": chk["reasons"] + ([f"projection above US${max_usd:,.2f}"] if over else [])}


def dry_run(systems: list[str]) -> dict:
    ji, jc, ci, cc = plan(systems)
    arm = D.arm_spec()
    cl = D.DocuseJevClient(RawStore(STORE), arm)
    bad = {f"{i.variant}/{i.item_id}": p for i in ji if (p := D.validate_request(cl.request(i), i))}
    proj = projection(systems)
    sp = spending(proj)
    out = {"calls": {"jev": len(jc), **pd.Series([x.system for x in cc]).value_counts().to_dict()},
           "calls_total": len(jc) + len(cc), "jev_invalid_requests": bad,
           "projection": proj.to_dict(orient="records"), "spending": sp,
           "config_sha256": D.config_sha256(), "items_sha256": K.file_sha256(ITEMS),
           "chatbot_user_example": K.user_text(chatbot_arm(), ci[0]), "chatbot_system": K.system_text(chatbot_arm())}
    print(json.dumps(out, indent=1, default=str))
    return out


def run(systems: list[str], yes: bool, key: str | None, workers: int, max_usd: float | None = None) -> None:
    ji, jc, ci, cc = plan(systems)
    arm_j, arm_c = D.arm_spec(), chatbot_arm()
    proj = projection(systems)
    print(proj.to_string(index=False))
    sp = spending(proj, max_usd)
    stop = f"stop ${sp['stop']:,.0f}" if sp["stop"] is not None else "no project stop set"
    pstop = f"projection stop ${sp['max_usd']:,.2f}" if sp["max_usd"] is not None else "no projection stop set"
    print(f"key usage ${sp['key_usage']:,.2f} + open holds ${sp['open_holds']:,.2f} + this run ${sp['projection_usd']:,.4f}"
          f" = ${sp['total']:,.2f} ({stop}; {pstop})")
    if not sp["ok"]:
        raise SystemExit("spending stop: " + "; ".join(sp["reasons"]))
    if key:                               # this process only
        os.environ["OPENROUTER_API_KEY"] = os.environ[key]
    if not yes and input(f"{len(jc) + len(cc)} calls. Proceed? [y/N] ").strip().lower() != "y":
        return
    store = RawStore(STORE)
    frames = []
    if jc:
        frames.append(K.execute(jc, ji, D.make_factory(arm_j, store), None, workers=min(workers, 4), arm_name=arm_j.name))
    if cc:
        frames.append(K.execute(cc, ci, K.make_factory(arm_c, store, standard=True), None, workers=workers,
                                arm_name=arm_c.name))
    df = pd.concat(frames, ignore_index=True)
    print(df.groupby(["system", "variant"])["valid"].agg(["size", "mean"]).to_string())
    err = df[df.error.notna()]
    if len(err):
        print(f"WARNING {len(err)} calls failed (run again to retry):", err.error.str[:120].value_counts().to_dict())


# ------------------------------------------------------------------------------------------------ analysis
def chatbot_calls(ci: list[K.Item]) -> pd.DataFrame:
    """The chatbots' calls table from the store only: refuses if any planned request has no stored answer."""
    arm = chatbot_arm()
    store = RawStore(STORE)
    f = K.make_factory(arm, store, standard=True)
    missing = [(s, i.item_id) for s in CHATBOTS for i in ci if store.get(f(s).request(i)) is None]
    if missing:
        raise SystemExit(f"{len(missing)} chatbot calls have no stored answer: run --run first")
    cc = [K.CatCall(s, i.item_id, 0, i.variant) for s in CHATBOTS for i in ci]
    return K.execute(cc, ci, f, None, workers=4, arm_name=arm.name)


def _counts(s: pd.Series, order=None) -> dict:
    v = s.fillna("(none)").astype(str).value_counts()
    keys = [k for k in (order or []) if k in v.index] + sorted(k for k in v.index if k not in (order or []))
    return {k: int(v[k]) for k in keys}


def _conf(x: pd.Series) -> dict | None:
    x = x.dropna()
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "min": float(x.min()),
            "max": float(x.max())}


def parts_correct(p: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Per question, the reports whose answer is the report's own truth (D.parts_of_truth: T, the regional node band
    without the separately submitted node, deposits, M1a)."""
    out = {q: 0 for q in D.QUESTION_IDS}
    for rid, t in truth.iterrows():
        want = D.parts_of_truth(t.t, int(t.nodes_involved), int(t.tumour_deposits), t.m)
        for q in D.QUESTION_IDS:
            out[q] += int(rid in p.index and p.loc[rid, f"{q}_choice"] == want[q])
    return out


def joint_confidence(parts: pd.DataFrame) -> pd.Series:
    """The product of the four top probabilities (config/docuse_staging.yaml combination.routing_confidence)."""
    return parts[[f"{q}_top" for q in D.QUESTION_IDS]].astype(float).prod(axis=1, min_count=4)


def analyze() -> dict:
    items = pd.read_csv(ITEMS, dtype=str, keep_default_na=False).set_index("report_id")
    ji, jc, ci, _ = plan(["jev"] + CHATBOTS)
    arm = D.arm_spec()
    jdf, parts = D.calls_table(RawStore(STORE), arm, ji, jc)
    if len(jdf) < len(jc):
        raise SystemExit(f"{len(jc) - len(jdf)} Jev calls have no stored answer: run --run first")
    cdf = chatbot_calls(ci)
    # the para-aortic / aortocaval reports: the stored answers of the documented run and the misleading-feature run
    conf = pd.read_csv(ROOT / D.CFG["items"]["confuser"], dtype=str, keep_default_na=False)
    old_ids = set(conf.loc[conf.confuser_type == str(TYPE), "report_id"])
    old_misread = conf.set_index("report_id").misread_level
    oj = pd.read_parquet(ROOT / D.CFG["results"] / "calls.parquet")
    oj = oj[oj.item_id.isin(old_ids) & (oj.repeat == 0)]
    op = pd.read_parquet(ROOT / D.CFG["results"] / "parts.parquet")
    op = op[op.item_id.isin(old_ids) & (op.repeat == 0)]
    oc = pd.read_parquet(ROOT / D.CFG["analysis"]["chatbot_confuser_calls"])
    oc = oc[oc.item_id.isin(old_ids) & (oc.repeat == 0) & (oc.variant == "names")]
    levels = list(D.LEVELS)
    truth = "IVA-IVB"
    rows, systems = [], {}

    def block(calls: pd.DataFrame, ids, misread) -> dict:
        a = calls.set_index("item_id").answer.reindex(sorted(ids))
        return {"n_reports": len(ids), "answered": int(a.notna().sum()),
                "usable": int(calls.valid.sum()), "correct": int((a == truth).sum()),
                "misread_level": int(sum(a.get(i) == misread.get(i) for i in ids)),
                "answers": _counts(a, levels)}

    for v in D.VARIANTS:
        d, p = jdf[jdf.variant == v], parts[parts.variant == v].set_index("item_id")
        od, opv = oj[oj.variant == v], op[op.variant == v].set_index("item_id")
        s = block(d, items.index, items.misread_level)
        s["four_questions"] = {q: _counts(p[f"{q}_choice"], list(D.options(q))) for q in D.QUESTION_IDS}
        s["four_questions_correct"] = parts_correct(p, items)
        jc_new = joint_confidence(p)
        s["joint_confidence"] = _conf(jc_new)
        s["by_site"] = {site: {"n": int((items.node_site == site).sum()),
                               "correct": int(sum(d.set_index("item_id").answer.get(i) == truth
                                                  for i in items.index[items.node_site == site])),
                               "distant_metastasis": _counts(p.loc[items.index[items.node_site == site],
                                                                   "distant_metastasis_choice"])} for site in SITES}
        o = block(od, old_ids, old_misread)
        o["four_questions"] = {q: _counts(opv[f"{q}_choice"], list(D.options(q))) for q in D.QUESTION_IDS}
        o["four_questions_correct"] = parts_correct(opv, conf.set_index("report_id").loc[sorted(old_ids)])
        o["joint_confidence"] = _conf(joint_confidence(opv))
        o["source"] = f"{D.CFG['results']}/calls.parquet and parts.parquet (repeat 0), the 20 type {TYPE} reports"
        s["para_aortic_aortocaval_reports"] = o
        systems[f"jev/{v}"] = s
        for rid, r in items.iterrows():
            rows.append({"system": "jev", "variant": v, "report_id": rid, "node_site": r.node_site, "truth": truth,
                         "misread_level": r.misread_level, "answer": d.set_index("item_id").answer.get(rid),
                         "joint_confidence": jc_new.get(rid),
                         **{f"{q}_choice": p[f"{q}_choice"].get(rid) for q in D.QUESTION_IDS}})
    for m in CHATBOTS:
        d = cdf[cdf.system == m]
        s = block(d, items.index, items.misread_level)
        s["by_site"] = {site: {"n": int((items.node_site == site).sum()),
                               "correct": int(sum(d.set_index("item_id").answer.get(i) == truth
                                                  for i in items.index[items.node_site == site]))} for site in SITES}
        o = block(oc[oc.system == m], old_ids, old_misread)
        o["source"] = f"{D.CFG['analysis']['chatbot_confuser_calls']} (names, repeat 0), the 20 type {TYPE} reports"
        s["para_aortic_aortocaval_reports"] = o
        systems[f"{m}/names"] = s
        for rid, r in items.iterrows():
            rows.append({"system": m, "variant": "names", "report_id": rid, "node_site": r.node_site, "truth": truth,
                         "misread_level": r.misread_level, "answer": d.set_index("item_id").answer.get(rid)})
    allc = pd.concat([jdf, cdf], ignore_index=True)
    summ = {"run": "non-regional node sites the notes do not name", "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "sites": SPEC["sites"], "site_source": SPEC["source"],
            "items": str(ITEMS.relative_to(ROOT)).replace("\\", "/"), "items_sha256": K.file_sha256(ITEMS),
            "config_sha256": D.config_sha256(), "seed": SEED, "truth": {"level": truth, "substage": "IVA", "m": "M1a"},
            "calls": int(len(allc)), "calls_by_system": allc.groupby("system").size().to_dict(),
            "usable": int(allc.valid.sum()),
            "providers": allc.groupby("system").provider.agg(lambda x: x.value_counts().to_dict()).to_dict(),
            "models_reported": allc.groupby("system").model_reported.agg(lambda x: x.value_counts().to_dict()).to_dict(),
            "routes": allc.groupby("system").route.agg(lambda x: x.value_counts().to_dict()).to_dict(),
            "usd_reported": float(allc.usd_reported.fillna(0).sum()),
            "usd_reported_by_system": allc.groupby("system").usd_reported.sum().to_dict(),
            "errors": allc.error.dropna().astype(str).str[:120].value_counts().to_dict(),
            "systems": systems}
    RESULTS.mkdir(parents=True, exist_ok=True)
    jdf.to_parquet(RESULTS / "calls_jev.parquet", index=False)
    parts.to_parquet(RESULTS / "parts.parquet", index=False)
    cdf.to_parquet(RESULTS / "calls_chatbots.parquet", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "per_report.csv", index=False, lineterminator="\n")
    (RESULTS / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    for k, s in systems.items():
        o = s["para_aortic_aortocaval_reports"]
        jc = (f"; joint confidence median {s['joint_confidence']['median']:.3f} (para-aortic/aortocaval "
              f"{o['joint_confidence']['median']:.3f})" if s.get("joint_confidence") else "")
        print(f"{k:28s} correct {s['correct']:2d}/{s['n_reports']} (para-aortic/aortocaval {o['correct']:2d}/"
              f"{o['n_reports']}); misread {s['misread_level']}{jc}")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--dry-run", "--run", "--analyze"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--systems", default="jev,gpt,claude,gemini,glm")
    ap.add_argument("--key", default=None, help="run: the .env name of the OpenRouter key for this process")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_usd", type=float, default=None, help="run: refuse a projection above this (USD); default: "
                                                                "no stop")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    systems = [s for s in a.systems.split(",") if s]
    if a.dry_run:
        dry_run(["jev"] + CHATBOTS)
    elif a.run:
        run(systems, a.yes, a.key, a.workers, a.max_usd)
    else:
        analyze()


if __name__ == "__main__":
    main()
