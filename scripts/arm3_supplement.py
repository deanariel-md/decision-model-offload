"""Arm 3 supplement tables: repeatability on the repeat set and cost per system (src/jevity/arm3_supplement.py).
Descriptive: the confirmatory outcomes stay those of the shared analysis.

Run after the shared analysis, on the same calls (and again once the timing sample has run):

    python scripts\\run_categorical.py arm3 analyze
    python scripts\\arm3_supplement.py

Reads results/arm3/calls.parquet, the raw replies it points to (reasoning tokens), results/arm3/timing_sample.parquet
if present and the shared analysis of each version (required: costs checked for agreement, and its coverage curve is
the one printed; its routing table is results/arm3/table_routing.md, from run_categorical.py arm3 tables); writes
results/arm3/supplement_<version>.json and results/arm3/supplement_tables.md. No network, no model call."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import run_categorical as RC
from jevity import arm3_supplement as S
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity.clients import MODELS


def _raw(path, paths: RC.Paths) -> dict | None:
    """A stored raw reply, at its recorded path or, if the store was moved, by file name under the arm's stores."""
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    for q in (p, paths.runs / p.name):
        if q.exists():
            return json.loads(q.read_text(encoding="utf-8"))
    return None


def _levels(calls: pd.DataFrame, labels: tuple[str, ...], s: str, v: str, rep_ids: list[str], n_rep: int) -> np.ndarray:
    """Repeat set x repeats: the level index of each answer, NaN where unusable, missing or not collected."""
    c = calls[(calls.system == s) & (calls.variant == v) & calls.item_id.isin(rep_ids) & (calls["repeat"] < n_rep)]
    x = np.full((len(rep_ids), n_rep), np.nan)
    pos = {i: k for k, i in enumerate(rep_ids)}
    for r in c.itertuples():
        got = getattr(r, "error", None) != KA.BATCH_MISSING
        if got and pd.notna(r.valid) and bool(r.valid) and r.answer in labels:
            x[pos[r.item_id], int(r.repeat)] = labels.index(r.answer)
    return x


def _mean(x) -> float | None:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else None


def _report_fields(arm: K.ArmSpec) -> pd.DataFrame:
    """The items file's truth cell and format per report (T, N, M, layout), indexed by report id."""
    p = Path(arm.config["items"]["main"])
    df = pd.read_csv(p if p.is_absolute() else ROOT / p, dtype=str, keep_default_na=False)
    return df.set_index(arm.config["items"]["id_column"])[["t", "n", "m", "layout"]]


def crc_extras(arm: K.ArmSpec, t: KA.Table, n_boot: int, seed: int, ci: float) -> dict:
    """Arm 3's primary-version additions: exact accuracy by report format and the component behind each wrong
    answer."""
    f = _report_fields(arm).loc[t.items.item_id]
    fmt = f["layout"].to_numpy()
    fidx = KA.strat_index(np.char.add(np.char.add(t.strata.astype(str), "|"), fmt.astype(str)), n_boot, seed)
    tnm = list(zip(f["t"], f["n"], f["m"]))
    truth = [arm.labels[i] for i in t.truth]
    return {"settings": {"by_format": ("exact accuracy per report format; reports resampled within level x format; "
                                       "difference synoptic minus narrative"),
                         "components": ("each wrong answer: the fewest of T, N and M that must change, from the "
                                        "true cell, to reach a cell the AJCC 8th table places in the answered level; "
                                        "ties listed together")},
            "by_format": {s: S.accuracy_by_format(t.correct[:, t.col(s)], fmt, fidx, ci) for s in t.systems},
            "components": {s: S.wrong_answer_components(
                tnm, [arm.labels[i] if i >= 0 else None for i in t.answer[:, t.col(s)]], truth) for s in t.systems}}


def run(arm: K.ArmSpec, paths: RC.Paths, primary_extras=crc_extras, every_extras=None,
        title: str = "Arm 3 supplement tables", unit: str = "reports",
        script: str = "scripts/arm3_supplement.py") -> dict:
    """`primary_extras(arm, t, n_boot, seed, ci) -> dict` adds keys to the primary version's results (arm 3: report
    format and components); `every_extras` the same for every version (the eICU AKI set: Jev's confident half and the
    confusion by stage). Their "settings" merge into the version's settings."""
    calls = pd.read_parquet(paths.calls())
    items, sys_, rep, *_ = RC.design(arm)
    a = {"n_boot": 2000, "ci": 0.95, "margin": 0.05, **arm.config["analysis"]}
    seed, n_boot, ci = int(arm.config["seed"]), int(a["n_boot"]), float(a["ci"])
    n_rep, share = int(arm.config["repeats"]["n_repeats"]), float(a["hybrid_share"]["jev_share"])
    tp = paths.results / "timing_sample.parquet"
    timing = KA.timing_summary(pd.read_parquet(tp)) if tp.exists() else {}
    speed = {s: timing[s] for s in sys_ if s in timing}                     # the tables' system order
    primary = arm.config.get("primary_variant", "main")
    variants = sorted({i.variant for i in items}, key=lambda x: (x != primary, x))   # the primary version first
    rep_ids = sorted(rep)
    out, shared_by = {}, {}
    for v in variants:
        t = KA.Table(calls, RC.items_frame(items, v), sys_, arm.labels, MODELS, v)
        idx = KA.strat_index(t.strata, n_boot, seed)
        f = paths.results / (f"analysis_{v}.json" if len(variants) > 1 else "analysis.json")
        if not f.exists():
            raise SystemExit(f"{f} does not exist: run the shared analysis first (its routing and coverage curve are "
                             "the ones printed)")
        shared = shared_by[v] = json.loads(f.read_text(encoding="utf-8"))
        main = calls[(calls["repeat"] == 0) & (calls.variant == v) & calls.item_id.isin(set(t.items.item_id))]
        res = {"arm": arm.name, "variant": v, "n_items": len(t.items), "repeat_set": len(rep_ids),
               "simulated": bool("simulated" in calls and calls.simulated.fillna(False).astype(bool).any()),
               "note": "descriptive; the confirmatory outcomes are the shared analysis's",
               "settings": {"n_boot": n_boot, "seed": seed, "ci": ci, "hybrid_jev_share": share,
                            "repeatability": "ICC(2,1) of the level index and the share answered identically, on the "
                                             "reports with three usable answers; reports resampled",
                            "cost": "main pass; list and billed prices as the shared analysis; reasoning tokens as "
                                    "reported in the raw reply"},
               "repeatability": {}, "cost": {}, "value": {}, "hybrid": {}, "speed": speed}
        for s in sys_:
            res["repeatability"][s] = S.repeatability(_levels(calls, arm.labels, s, v, rep_ids, n_rep), n_boot, seed, ci)
        for j, s in enumerate(sys_):
            g = main[(main.system == s) & (main.error.fillna("") != KA.BATCH_MISSING)]
            rt = [S.reasoning_tokens(_raw(p, paths)) for p in g.raw_path]
            present = t.present[:, j]
            lst, bil, cor = t.cost_list[:, j], t.cost_billed[:, j], t.correct[:, j]
            per_list = float(lst[present].mean()) if present.any() else None
            bk, bc = lst[idx].sum(axis=1), cor[idx].sum(axis=1)
            res["cost"][s] = {
                "answers": int(present.sum()),
                "input_tokens_per_answer": _mean(g.tokens_in), "output_tokens_per_answer": _mean(g.tokens_out),
                "reasoning_tokens_per_answer": _mean([x for x in rt if x is not None]),
                "answers_reporting_reasoning": int(sum(x is not None for x in rt)),
                "usd_billed_per_1000_answers": float(bil[present].mean() * 1000) if present.any() else None,
                "usd_list_per_1000_answers": per_list * 1000 if per_list is not None else None,
                "usd_list_per_million_answers": per_list * 1e6 if per_list is not None else None,
                "usd_list_per_correct": float(lst.sum() / cor.sum()) if cor.sum() else None,
                "usd_list_per_correct_ci": KA.percentile_ci(bk[bc > 0] / bc[bc > 0], ci) if (bc > 0).any() else None}
            sc = shared["systems"][s]["cost"]["usd_list_per_correct"]
            assert sc is None or abs(sc - res["cost"][s]["usd_list_per_correct"]) < 1e-12, (v, s)
        jj = t.col("jev")
        use = KA.hybrid_split(t, share)
        res["hybrid"] = {"jev_share": share, "n_jev": int(use.sum()), "systems": {}}
        for s in [x for x in sys_ if x != "jev"]:
            jc = t.col(s)
            nj, nc = int(t.correct[:, jj].sum()), int(t.correct[:, jc].sum())
            vl = KA.value_vs_jev(t, s, idx, ci)["list"]
            more = nc > nj
            res["value"][s] = {"accuracy_jev": float(t.correct[:, jj].mean()), "accuracy_chatbot": float(t.correct[:, jc].mean()),
                               "correct_jev": nj, "correct_chatbot": nc, "chatbot_more_accurate": more,
                               "verdict": S.value_verdict(nj, nc, res["cost"]["jev"]["usd_list_per_1000_answers"] or 0.0,
                                                          res["cost"][s]["usd_list_per_1000_answers"] or 0.0),
                               "usd_per_extra_correct": vl["usd_per_extra_correct"] if more else None,
                               "usd_per_extra_correct_ci": vl["usd_per_extra_correct_ci"] if more else None,
                               "share_resamples_chatbot_not_more_correct": vl["share_resamples_chatbot_not_more_correct"],
                               "extra_usd_per_1000_items": vl["extra_usd_per_1000_items"],
                               "extra_correct_per_1000_items": vl["extra_correct_per_1000_items"]}
            hb, _ = KA.hybrid_share(t, s, a, idx, use)
            k = hb["cost_list"]
            save = k["chatbot_usd_per_1000_items"] - k["hybrid_usd_per_1000_items"]
            res["hybrid"]["systems"][s] = {
                "accuracy_hybrid": hb[a["primary_metric"]], "accuracy_hybrid_ci": hb[f"{a['primary_metric']}_ci"],
                "accuracy_chatbot": hb["chatbot_alone"], "diff": hb["diff"], "diff_ci": hb["ci"],
                "usd_hybrid_per_1000": k["hybrid_usd_per_1000_items"], "usd_chatbot_per_1000": k["chatbot_usd_per_1000_items"],
                "saving_usd_per_1000": save,
                "saving_share": save / k["chatbot_usd_per_1000_items"] if k["chatbot_usd_per_1000_items"] > 0 else None}
        for fn in ([primary_extras] if v == primary and primary_extras else []) + ([every_extras] if every_extras else []):
            extra = fn(arm, t, n_boot, seed, ci)
            res["settings"].update(extra.pop("settings", {}))
            res.update(extra)
        res = KA.clean(res)
        (paths.results / f"supplement_{v}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        out[v] = res
    (paths.results / "supplement_tables.md").write_text(
        S.tables_markdown(out, shared_by, title=title, script=script, results_dir=f"results/{arm.name}", unit=unit),
        encoding="utf-8")
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    arm = K.load_arm("arm3")
    paths = RC.Paths(arm)
    out = run(arm, paths)
    print(f"wrote {', '.join(f'supplement_{v}.json' for v in out)} and supplement_tables.md under {paths.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
