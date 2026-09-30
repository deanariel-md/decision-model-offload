"""Arm 3, Jev used as documented: the structure-only ablation (src/jevity/docuse_structure_only.py). Only Jev is called,
only in this variant, repeat 0, on the report sets the documented version was run on (--set main|confuser|reworded|
nonregional|heldout|all; default main). Every live command prints the calls and the projected cost, checks the spending
stops when they are set (config budget, as run_staging.py; --max_usd: the projection, and a running guard that refuses
any request once spent + in flight + that request would pass the cap, every set's stored answers counted) and asks
before sending.
  python scripts/docuse/run_structure_only.py --dry-run --set all      # builds and checks every request; sends nothing
  python scripts/docuse/run_structure_only.py --full --set all         # every report of every set (stored reused)
  python scripts/docuse/run_structure_only.py --analyze --set all      # tables, summary, ablation_staging.json
  python scripts/docuse/run_structure_only.py --full --set all --simulate --out DIR   # synthetic answers, under DIR
--dry-run also checks each request against the one the set's documented run stored (its raw store must be present).
Raw responses: each set's store (docuse_structure_only.SETS; main and confuser share runs/arm3_docuse_structure_only as
the documented run shares runs/arm3_docuse). Tables: results/arm3_docuse_structure_only/<set>/calls.parquet and
parts.parquet; results/arm3_docuse_structure_only/summary.json and results/summaries/ablation_staging.json (under DIR
when simulated). --analyze rebuilds the tables from the raw stores and needs the main set; it also reads the documented
runs' tables and analyses and results/summaries/docuse_summary.json and hybrid_explore.json."""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "summaries"))
from jevity import categorical as K
from jevity import docuse_staging as D
from jevity import docuse_structure_only as S
from jevity.clients import MODELS, RawStore

import run_staging as R
from sens_spec import prop, wilson

FULL_CUT = 1 - 1e-9          # full confidence: joint confidence 1.00 (hybrid_explore: below the cut by more than 1e-9)
QKEYS = {"t_category": "t", "regional_nodes": "nodes", "tumor_deposits": "deposits", "distant_metastasis": "m"}
SET_FIELDS = ("exact_accuracy", "ci", "n", "wrong", "full_conf_share", "sensitivity", "specificity", "usd_per_1000",
              "per_question")


def sets_of(arg: str) -> list[str]:
    names = list(S.SET_NAMES) if arg == "all" else [arg]
    missing = [n for n in names if not S.SETS[n].available()]
    if missing:
        raise SystemExit(f"items file missing for {missing}: {[str(S.SETS[n].items_csv) for n in missing]}")
    return names


class Paths:
    """Where the stores, tables and summary file go: the repository, or DIR for a simulated run (stores and tables
    re-rooted under DIR, the summary file at DIR/results/summaries)."""

    def __init__(self, root: Path = ROOT, simulated: bool = False):
        self.root, self.simulated = Path(root), simulated
        self.results = self.root / S.RESULTS_REL
        self.writing = (self.root if simulated else ROOT) / "results" / "summaries" / "ablation_staging.json"

    def store(self, set_name: str) -> RawStore:
        return RawStore(self.root / S.SETS[set_name].store.relative_to(ROOT))

    def spent(self) -> float:
        return sum(S.stored_spend(RawStore(d)) for d in {self.store(n).run_dir for n in S.SET_NAMES})


def arm_spec():
    arm = D.arm_spec()
    arm.config["budget"] = {**arm.config["budget"], "arm_stop_usd": S.SPEND_CAP_USD}
    return arm


def plan(set_name: str, mode: str, items: list[K.Item]) -> list[K.CatCall]:
    if mode != "full":
        raise ValueError(mode)
    return [K.CatCall("jev", i, 0, S.VARIANT) for i in S.ordered_ids(set_name, items)]


def projection(arm, items: list[K.Item], calls: list[K.CatCall], store: RawStore) -> pd.DataFrame:
    """Projected list cost of the calls without a stored answer (body tokens at 4 characters per token)."""
    by = {i.item_id: i for i in items}
    cl = S.StructureOnlyJevClient(store, arm)
    reqs = [cl.request(by[c.item_id]) for c in calls]
    new = [r for r in reqs if store.get(r) is None]
    tin = sum(D.request_tokens(r) for r in new)
    return pd.DataFrame([{"system": "jev", "calls": len(new), "stored": len(reqs) - len(new), "route": "standard",
                          "tokens_in": tin, "tokens_out": 0, "usd": tin * MODELS["jev"]["price_per_mtok_input"] / 1e6}])


def check_set(set_name: str, arm, items: list[K.Item], against_stored: bool = True) -> dict[str, list[str]]:
    """Every request of a set: S.validate_request, no duplicate, and (against_stored) the stored documented request
    for the report with null criteria."""
    cl = S.StructureOnlyJevClient(RawStore(ROOT / "runs" / "_unused"), arm)
    problems, seen = {}, set()
    for it in items:
        req = cl.request(it)
        bad = S.validate_request(req, it, arm)
        if K._sha(req) in seen:
            bad.append("duplicate request")
        seen.add(K._sha(req))
        if against_stored:
            bad += S.check_against_stored(it, req, arm, set_name)
        if bad:
            problems[it.item_id] = bad
    return problems


def dry_run(paths: Paths, set_names: list[str]) -> dict:
    arm = arm_spec()
    price = MODELS["jev"]["price_per_mtok_input"]
    per, tot_req, tot_usd = {}, 0, 0.0
    for sn in set_names:
        items = S.load_items(sn)
        problems = check_set(sn, arm, items)
        store = paths.store(sn)
        proj = projection(arm, items, plan(sn, "full", items), store)
        cl = S.StructureOnlyJevClient(store, arm)
        tok = sum(D.request_tokens(cl.request(i)) for i in items)
        per[sn] = {"reports": len(items), "requests": len(items), "invalid": len(problems),
                   "problems": dict(list(problems.items())[:5]), "checked_against_stored_documented": len(items),
                   "tokens_in_per_request": tok / len(items), "usd_list_all": tok * price / 1e6,
                   "already_stored": int(proj.stored.sum()), "usd_list_to_send": float(proj.usd.sum()),
                   "store": str(store.run_dir.relative_to(paths.root)).replace("\\", "/"),
                   "workers": S.SETS[sn].workers, "first_report": S.ordered_ids(sn, items)[0]}
        tot_req += len(items)
        tot_usd += per[sn]["usd_list_to_send"]
    return {"variant": S.VARIANT, "sets": per, "requests_total": tot_req, "usd_list_to_send_total": tot_usd,
            "spent_so_far_reported": paths.spent(), "spend_cap_usd": S.SPEND_CAP_USD,
            "config_sha256": D.config_sha256(), "model": MODELS["jev"]["slug"],
            "invalid_total": sum(v["invalid"] for v in per.values())}


def rebuild_tables(paths: Paths, arm, set_name: str, items) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, parts = S.calls_table(paths.store(set_name), arm, items)
    d = paths.results / set_name
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "calls.parquet", index=False)
    parts.to_parquet(d / "parts.parquet", index=False)
    return df, parts


def run_calls(mode: str, paths: Paths, set_names: list[str], yes: bool = False, workers: int | None = None,
              sender=None) -> dict | None:
    arm = arm_spec()
    runs_by_set = []
    for sn in set_names:
        items = S.load_items(sn)
        calls = plan(sn, mode, items)
        bad = check_set(sn, arm, [i for i in items if i.item_id in {c.item_id for c in calls}], against_stored=False)
        if bad:
            raise SystemExit(f"{sn}: invalid requests: {dict(list(bad.items())[:3])}")
        proj = projection(arm, items, calls, paths.store(sn))
        print(f"{arm.name}/{S.VARIANT} {mode} set {sn}: {len(calls)} Jev calls ({int(proj.calls.sum())} to send, "
              f"{int(proj.stored.sum())} stored); projected ${float(proj.usd.sum()):.6f}")
        runs_by_set.append((sn, items, calls, proj))
    proj_all = pd.concat([p for *_, p in runs_by_set], ignore_index=True).groupby(
        ["system", "route"], as_index=False).sum(numeric_only=True)
    guard = None
    if sender is None:
        K.load_keys(arm)
        R.spending_stop(arm, proj_all)
        guard = S.SpendGuard(S.SPEND_CAP_USD, paths.spent())
        cap = f"cap ${S.SPEND_CAP_USD:.2f}" if S.SPEND_CAP_USD is not None else "no cap set"
        print(f"spend so far (this variant, every set, reported): ${guard.spent:.6f}; projected new "
              f"${float(proj_all.usd.sum()):.6f}; {cap}")
        if S.SPEND_CAP_USD is not None and guard.spent + float(proj_all.usd.sum()) > S.SPEND_CAP_USD:
            raise SystemExit(f"spend cap: projection passes US${S.SPEND_CAP_USD:.2f}")
        if not yes and input(f"{int(proj_all.calls.sum())} calls. Proceed? [y/N] ").strip().lower() != "y":
            return None
    out = {}
    for sn, items, calls, _ in runs_by_set:
        factory = S.make_factory(arm, paths.store(sn), sender=sender, spend=guard)
        done = K.execute(calls, items, factory, None, workers=int(workers or S.SETS[sn].workers), arm_name=arm.name)
        print(f"set {sn}:", done.groupby("parse", dropna=False).size().to_dict())
        err = done[done.error.notna()]
        if len(err):
            print(f"WARNING set {sn}: {len(err)} calls failed (stored as unusable; run the command again to retry):",
                  err.error.str[:160].value_counts().head(5).to_dict())
        df, _ = rebuild_tables(paths, arm, sn, items)
        print(f"set {sn}: calls.parquet {len(df)} rows ({int(df.valid.sum())} usable); tokens_in mean "
              f"{df.tokens_in.mean():.1f}; usd_reported sum {df.usd_reported.sum():.6f}")
        out[sn] = df
    if guard is not None:
        print(f"spend after run (this variant, every set, reported): ${paths.spent():.6f}")
    return out


# ------------------------------------------------------------------------------------------------ analysis
def metrics(calls: pd.DataFrame, parts: pd.DataFrame, items: list[K.Item], truth_parts: dict) -> dict:
    """Stage group from the four answers (docuse_staging.parse_answers, the key's AJCC table); a missing or unusable
    answer counts as wrong. Full confidence: joint confidence (product of the four top probabilities, calls.top_prob)
    at 1.00. Sensitivity: wrong answers short of full confidence; specificity: right answers at full confidence. Cost:
    mean reported input tokens at Jev's list input price (output free), per 1,000 reports."""
    ids = sorted(i.item_id for i in items)
    truth = pd.Series({i.item_id: i.truth for i in items}).reindex(ids)
    c = calls[calls["repeat"] == 0]
    assert not c.item_id.duplicated().any()
    c = c.set_index("item_id").reindex(ids)
    valid = c["valid"].fillna(False).astype(bool)
    correct = (valid & (c["answer"] == truth)).to_numpy()
    top = pd.to_numeric(c["top_prob"], errors="coerce").to_numpy(dtype=float)
    full = np.nan_to_num(top, nan=-1.0) >= FULL_CUT
    n, k = len(ids), int(correct.sum())
    p = parts[parts["repeat"] == 0].set_index("item_id").reindex(ids)
    perq, wrong_by_q = {}, {}
    tq = {q: pd.Series({i: truth_parts[i][q] for i in ids}) for q in QKEYS}
    for q, short in QKEYS.items():
        ok = (p[f"{q}_choice"] == tq[q]).to_numpy()
        perq[short] = float(ok.mean())
        wrong_by_q[short] = int((~ok).sum())
    n_wrong_parts = sum((p[f"{q}_choice"] != tq[q]).astype(int) for q in QKEYS).to_numpy()
    price = float(MODELS["jev"]["price_per_mtok_input"])
    tin = pd.to_numeric(c["tokens_in"], errors="coerce")
    out = {"exact_accuracy": k / n, "ci": wilson(k, n), "per_question": perq, "wrong": n - k,
           "full_conf_share": float(full.mean()),
           "sensitivity": prop(int((~correct & ~full).sum()), int((~correct).sum())),
           "specificity": prop(int((correct & full).sum()), int(correct.sum())),
           "usd_per_1000": float(tin.mean()) * price / 1e6 * 1000, "n": n}
    extra = {"n": n, "answered": int(c["system"].notna().sum()), "usable": int(valid.sum()),
             "unusable_or_missing": int((~valid).sum()), "wrong_by_question": wrong_by_q,
             "stage_wrong_by_number_of_wrong_parts": {str(m): int(((n_wrong_parts == m) & ~correct).sum())
                                                     for m in range(5)},
             "tokens_in_per_answer": float(tin.mean()),
             "usd_reported_total": float(pd.to_numeric(c["usd_reported"], errors="coerce").sum()),
             "confusion_by_question": {short: {str(a): {str(b): int(v) for b, v in row.items() if v}
                                               for a, row in pd.crosstab(tq[q], p[f"{q}_choice"].fillna("unusable")).iterrows()}
                                       for q, short in QKEYS.items()}}
    return {"writing": out, "extra": extra}


def documented_frames(set_name: str, items: list[K.Item]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rs, ids = S.SETS[set_name], {i.item_id for i in items}
    dc, dp = pd.read_parquet(rs.doc_calls), pd.read_parquet(rs.doc_parts)
    dc = dc[(dc.variant == S.BASE_VARIANT) & (dc["repeat"] == 0) & dc.item_id.isin(ids)]
    dp = dp[(dp.variant == S.BASE_VARIANT) & (dp["repeat"] == 0) & dp.item_id.isin(ids)]
    assert set(dc.item_id) == ids, (set_name, len(ids - set(dc.item_id)))
    return dc, dp


def _close(a, b, what, tol=1e-12):
    assert a is not None and b is not None and abs(float(a) - float(b)) < tol, (what, a, b)


def check_documented(set_name: str, w: dict, simulated: bool) -> list[str]:
    """The documented metrics recomputed here against the stored documented analyses of each set."""
    J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    dres, done = ROOT / D.CFG["results"], []
    if set_name == "main":
        an = J(dres / S.BASE_VARIANT / "analysis_names.json")["systems"]["jev"]
        pj = J(dres / S.BASE_VARIANT / "parts.json")["main_reports"]
        _close(w["exact_accuracy"], an["exact_accuracy"], "main accuracy")
        _close(w["usd_per_1000"], an["cost"]["usd_list_per_1000_answers"], "main cost", 1e-9)
        for q, short in QKEYS.items():
            _close(w["per_question"][short], pj[q]["accuracy"], f"main {q}")
        done += ["results/arm3_docuse/documented/analysis_names.json systems.jev (exact_accuracy, cost)",
                 "results/arm3_docuse/documented/parts.json main_reports.*.accuracy"]
    elif set_name == "confuser":
        an = J(dres / S.BASE_VARIANT / "analysis_confuser.json")["systems"]["jev"]
        _close(w["exact_accuracy"], an["exact_accuracy"], "confuser accuracy")
        _close(w["usd_per_1000"], an["cost"]["usd_list_per_1000_answers"], "confuser cost", 1e-9)
        done.append("results/arm3_docuse/documented/analysis_confuser.json systems.jev (exact_accuracy, cost)")
    elif set_name == "reworded":
        sm = J(ROOT / "results/arm3_docuse_reworded/summary.json")["per_variant"][S.BASE_VARIANT]
        _close(w["exact_accuracy"], sm["exact_accuracy"], "reworded accuracy")
        for q, short in QKEYS.items():
            _close(w["per_question"][short], sm["part_accuracy"][q], f"reworded {q}")
        done.append("results/arm3_docuse_reworded/summary.json per_variant.documented (exact_accuracy, part_accuracy)")
    elif set_name == "nonregional":
        sm = J(ROOT / "results/arm3_nonregional_sites/summary.json")["systems"][f"jev/{S.BASE_VARIANT}"]
        _close(w["exact_accuracy"], sm["correct"] / sm["n_reports"], "nonregional accuracy")
        for q, short in QKEYS.items():
            _close(w["per_question"][short], sm["four_questions_correct"][q] / sm["n_reports"], f"nonregional {q}")
        done.append("results/arm3_nonregional_sites/summary.json systems['jev/documented'] (correct, four_questions_correct)")
    elif set_name == "heldout":
        hr = ROOT / "results" / "arm3_heldout"
        an = J(hr / "analysis_documented_names.json")["systems"]["jev"]
        _close(w["exact_accuracy"], an["exact_accuracy"], "heldout accuracy")
        _close(w["usd_per_1000"], an["cost"]["usd_list_per_1000_answers"], "heldout cost", 1e-9)
        done.append(f"{hr.relative_to(ROOT).as_posix()}/analysis_documented_names.json systems.jev "
                    "(exact_accuracy, cost)")
    he = ROOT / "results" / "summaries" / "hybrid_explore.json"
    key = {"main": "main", "confuser": "misleading_feature", "nonregional": "nonregional_sites"}.get(set_name)
    if key and he.exists():
        m = J(he)["structured_safety_net"]["primary"]["documented|median"]
        assert abs(m["cut"] - 1.0) < 1e-12
        x = m["sets"][key]
        _close(w["full_conf_share"], 1 - x["share_sent"], f"{set_name} full-confidence share")
        assert x["jev_wrong"] == w["wrong"] and x["wrong_sent"] == w["sensitivity"]["k"] \
            and x["right_kept"] == w["specificity"]["k"], (set_name, x, w)
        done.append("results/summaries/hybrid_explore.json structured_safety_net.primary['documented|median']"
                    f".sets.{key}")
    return done


def analyze(paths: Paths, set_names: list[str], allow_incomplete: bool = False) -> dict:
    if "main" not in set_names:
        raise SystemExit("--analyze needs the main set (the top-level fields of ablation_staging.json)")
    arm = arm_spec()
    blocks, extras, checks = {"structure_only": {}, "structure_definitions": {}}, {}, {}
    for sn in set_names:
        items = S.load_items(sn)
        so_calls, so_parts = rebuild_tables(paths, arm, sn, items)
        missing = sorted({i.item_id for i in items} - set(so_calls.item_id))
        if missing and not allow_incomplete:
            raise SystemExit(f"set {sn}: {len(missing)} reports have no stored structure_only answer (e.g. "
                             f"{missing[:5]}); run --full --set {sn}, or --allow_incomplete to score them as wrong")
        tp = S.truth_parts(sn)
        so = metrics(so_calls, so_parts, items, tp)
        doc = metrics(*documented_frames(sn, items), items, tp)
        checks[sn] = check_documented(sn, doc["writing"], paths.simulated)
        blocks["structure_only"][sn], blocks["structure_definitions"][sn] = so["writing"], doc["writing"]
        extras[sn] = {"structure_only": so["extra"], "structure_definitions": doc["extra"],
                      "structure_only_missing_scored_wrong": len(missing)}
    ds = json.loads((ROOT / "results" / "summaries" / "docuse_summary.json").read_text(encoding="utf-8"))[
        "staging"]["original"]
    n = blocks["structure_only"]["main"]["n"]
    orig = {}
    for v in ("names", "definitions"):
        acc = float(ds[f"{v}_exact_accuracy"])
        kk = int(round(acc * n))
        assert abs(kk / n - acc) < 1e-12, (v, acc)
        orig[v] = {"exact_accuracy": acc, "ci": wilson(kk, n)}
    top = lambda b: {k: b["main"][k] for k in ("exact_accuracy", "ci", "per_question", "wrong", "full_conf_share",
                                               "sensitivity", "specificity", "usd_per_1000")}
    out = {"n": n, "names": orig["names"], "definitions": orig["definitions"]}
    for key in ("structure_only", "structure_definitions"):
        out[key] = {**top(blocks[key]), "sets": {sn: {f: blocks[key][sn][f] for f in SET_FIELDS} for sn in set_names}}
    summ = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "variant": S.VARIANT, "sets": set_names,
            "sources": {"structure_only": {sn: f"{S.RESULTS_REL}/{sn}/calls.parquet, parts.parquet (raw: "
                                               f"{S.SETS[sn].store.relative_to(ROOT).as_posix()})" for sn in set_names},
                        "structure_definitions": {sn: f"{S.SETS[sn].doc_calls.relative_to(ROOT).as_posix()}, "
                                                      f"{S.SETS[sn].doc_parts.name} (variant documented, repeat 0)"
                                                  for sn in set_names},
                        "names, definitions": "results/summaries/docuse_summary.json staging.original.{names,definitions}"
                                              "_exact_accuracy; Wilson CI here"},
            "definitions_of_measures": metrics.__doc__, "checks_documented_equals": checks, "ablation": out,
            "extra": extras}
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    paths.writing.parent.mkdir(parents=True, exist_ok=True)
    paths.writing.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"wrote {paths.results / 'summary.json'} and {paths.writing}")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--set", default="main", choices=list(S.SET_NAMES) + ["all"])
    ap.add_argument("--simulate", action="store_true", help="synthetic Jev answers, no network (needs --out)")
    ap.add_argument("--out", type=Path, default=None, help="root for runs/ and results/ (default: the repository)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max_usd", type=float, default=None,
                    help="spending cap in US$ on every call of this variant in every set, stored answers included "
                         "(default: no cap)")
    ap.add_argument("--allow_incomplete", action="store_true")
    a = ap.parse_args(argv)
    if a.simulate and a.out is None:
        raise SystemExit("--simulate writes under --out only, never into the repository's runs/ or results/")
    S.SPEND_CAP_USD = a.max_usd
    paths = Paths(a.out or ROOT, simulated=a.simulate)
    names = sets_of(a.set)
    if a.dry_run:
        out = dry_run(paths, names)
        print(json.dumps(out, indent=1))
        if out["invalid_total"]:
            raise SystemExit(f"{out['invalid_total']} invalid requests")
        return out
    if a.analyze:
        return analyze(paths, names, a.allow_incomplete)
    sender = None
    if a.simulate:
        tp = {k: v for sn in names for k, v in S.truth_parts(sn).items()}
        sender = D.synthetic_sender(tp, 0)
    return run_calls("full", paths, names, a.yes, a.workers, sender)


if __name__ == "__main__":
    main()
