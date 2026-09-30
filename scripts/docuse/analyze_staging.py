"""Arm 3, Jev used as documented: the arm 3 analysis with Jev's documented answers in place of its original ones.
No model call; stored outputs only. The chatbots' answers are arm 3's, as stored (results/arm3/calls.parquet); Jev's are
results/arm3_docuse/calls.parquet (scripts/docuse/run_staging.py), two versions (documented, documented_notes). The
scoring is categorical_analysis.analyze with arm 3's settings (config/arm3.yaml: seed, 2,000 resamples, margin 5
points, Jev on its most confident half).
  python scripts/docuse/analyze_staging.py [--root DIR] [--allow_incomplete] [--n_boot N]
Writes one set per Jev version V (documented, documented_notes) in results/arm3_docuse/V/:
  analysis_names.json, analysis_definitions.json   Jev version V against the chatbots' names / definitions answers;
                                                   the five main chatbots are the Holm family
  cheap_family_names.json, cheap_family_definitions.json   the same with the five cheaper systems as the Holm family
  staging_names.json, staging_definitions.json     main-stage accuracy, stage III answered 0 to II, IVC answered
                                                   IVA-IVB, confusion matrices, sensitivity / specificity / AUROC for
                                                   stage III-IV and IV, the error filter (Jev keeps its most confident
                                                   half), cost per 1,000
  parts.json                                       Jev's accuracy on each of the four questions
  analysis_confuser.json, misread.json             the 100 misleading-feature reports (arm3_confuser format)
and results/arm3_docuse/summary.json: the headline numbers with the file and key each comes from.
Checker: before writing, every chatbot's exact accuracy must equal the value in results/arm3/analysis*.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity import crc
from jevity import docuse_staging as D
from jevity.clients import MODELS, system_tier

import run_staging as R

A = D.CFG["analysis"]
LEVELS = D.LEVELS
MAIN_OF = [crc.main_stage(l) for l in LEVELS]


# ------------------------------------------------------------------------------------------------ frames
def combined(chat_calls: pd.DataFrame, jev_calls: pd.DataFrame, jev_variant: str, variant: str) -> pd.DataFrame:
    """The chatbots' stored answers of one version (names, definitions) with the answers of one Jev version
    (documented, documented_notes) in place of Jev's, relabelled to the chatbots' version so the shared Table reads
    them side by side."""
    assert jev_variant in D.VARIANTS, jev_variant
    chat = chat_calls[(chat_calls.variant == variant) & (chat_calls.system != "jev")]
    jev = jev_calls[(jev_calls.system == "jev") & (jev_calls.variant == jev_variant)].assign(variant=variant)
    return pd.concat([chat, jev[chat.columns.intersection(jev.columns)]], ignore_index=True)


def main_items_frame(variant: str) -> pd.DataFrame:
    """Arm 3's item frame (run_categorical.items_frame): item_id, truth, stratum (level), group (substage)."""
    a3 = D.source_arm()
    return R.RC.items_frame(K.load_items(a3), variant, a3)


def confuser_items() -> pd.DataFrame:
    f = pd.read_csv(ROOT / D.CFG["items"]["confuser"], dtype={"report_id": str}, keep_default_na=False)
    return f.rename(columns={"report_id": "item_id"}).assign(
        truth=lambda d: d.level, stratum=lambda d: d.confuser_type.astype(str), group=lambda d: d.substage)


def table(calls: pd.DataFrame, frame: pd.DataFrame, systems: list[str], variant: str) -> KA.Table:
    return KA.Table(calls, frame, systems, LEVELS, MODELS, variant)


# ------------------------------------------------------------------------------------------------ staging measures
def _boot_share(hit: np.ndarray, among: np.ndarray, idx: np.ndarray, ci: float) -> list | None:
    if not among.any():
        return None
    H, M = hit[idx].astype(float), among[idx].astype(float)
    ok = M.sum(axis=1) > 0
    return list(KA.percentile_ci((H * M).sum(axis=1)[ok] / M.sum(axis=1)[ok], ci))


def staging_extra(calls: pd.DataFrame, frame: pd.DataFrame, systems: list[str], variant: str = "names",
                  n_boot: int = 2000, seed: int = int(D.CFG["seed"]), ci: float = 0.95) -> dict:
    """Per system, main pass: main-stage accuracy (0, I, II, III, IV; unusable counts wrong); stage III reports answered
    0 to II and IVC reports answered IVA-IVB (counts among the usable answers, as results/summaries/arm3_summary.json);
    the 10 x 10 confusion matrix of usable answers; for stage III-IV and stage IV: sensitivity (an unusable answer is
    a miss), specificity (an unusable answer is not a correct negative) and the AUROC of the probability on the set.
    Percentile intervals from arm 3's resamples (stratified by level)."""
    t = table(calls, frame, systems, variant)
    idx = KA.strat_index(t.strata, n_boot, seed)
    main_t = np.array([MAIN_OF[k] for k in t.truth])
    out = {}
    for s in systems:
        j = t.col(s)
        ans, ok = t.answer[:, j], t.usable[:, j]
        main_a = np.array([MAIN_OF[k] if k >= 0 else None for k in ans], dtype=object)
        ms = ok & (main_a == main_t)
        iii = main_t == "III"
        ivc = t.truth == LEVELS.index("IVC")
        conf = np.zeros((len(LEVELS), len(LEVELS)), int)
        np.add.at(conf, (t.truth[ok], ans[ok]), 1)
        r = {"n_items": len(t.items), "main_stage_accuracy": float(ms.mean()),
             "main_stage_accuracy_ci": list(KA.percentile_ci(ms[idx].mean(axis=1), ci)),
             "stage_iii_n": int((iii & ok).sum()),
             "stage_iii_to_0_ii": int((iii & ok & np.isin(main_a, ["0", "I", "II"])).sum()),
             "ivc_n": int((ivc & ok).sum()),
             "ivc_to_iva_ivb": int((ivc & ok & (ans == LEVELS.index("IVA-IVB"))).sum()),
             "confusion": {"rows": "truth", "columns": "answer", "labels": list(LEVELS), "counts": conf.tolist(),
                           "n_usable": int(ok.sum()),
                           "unusable_by_truth": {l: int((~ok & (t.truth == k)).sum()) for k, l in enumerate(LEVELS)}}}
        r["stage_iii_to_0_ii_share"] = r["stage_iii_to_0_ii"] / r["stage_iii_n"] if r["stage_iii_n"] else None
        r["ivc_to_iva_ivb_share"] = r["ivc_to_iva_ivb"] / r["ivc_n"] if r["ivc_n"] else None
        for name, labs in A["positive_sets"].items():
            ks = [LEVELS.index(l) for l in labs]
            pos = np.isin(t.truth, ks)
            call = ok & np.isin(ans, ks)
            sens, spec = call & pos, ok & ~call & ~pos
            p = np.nansum(t.P[:, j, ks], axis=1)
            has = ~np.isnan(t.P[:, j, 0]) & ok
            au = KA.auroc(p[has], pos[has])
            boots = [KA.auroc(p[ix][has[ix]], pos[ix][has[ix]]) for ix in idx]
            boots = np.array([b for b in boots if b is not None])
            r[name] = {"levels": labs, "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()),
                       "sensitivity": float(sens[pos].mean()), "sensitivity_ci": _boot_share(sens, pos, idx, ci),
                       "specificity": float(spec[~pos].mean()), "specificity_ci": _boot_share(spec, ~pos, idx, ci),
                       "auroc": au, "auroc_ci": list(KA.percentile_ci(boots, ci)) if len(boots) else None,
                       "auroc_n": int(has.sum()),
                       "score": "sum of the system's probabilities on the set's levels"}
        out[s] = r
    return KA.clean(out)


def jev_share() -> float:
    """Arm 3's hybrid share (config/arm3.yaml analysis.hybrid_share.jev_share): the half Jev keeps."""
    return float(D.source_arm().config["analysis"]["hybrid_share"]["jev_share"])


def error_filter(calls: pd.DataFrame, frame: pd.DataFrame, variant: str = "names") -> dict:
    """Jev's error filter (config error_filter): Jev keeps its most confident half by the routing confidence (the
    product of the four top probabilities), with the hybrid's own split (categorical_analysis.hybrid_split: ties by
    report id, an unusable answer last); the other half goes to a person. Counts and accuracy on the kept and on the
    routed reports."""
    t = table(calls, frame, ["jev"], variant)
    corr = t.correct[:, 0]
    keep = KA.hybrid_split(t, jev_share())
    return KA.clean({"rule": D.CFG["error_filter"]["rule"], "source": D.CFG["error_filter"]["source"],
                     "share": jev_share(), "confidence": "top_prob (product of the four top probabilities)",
                     "n_kept": int(keep.sum()), "share_kept": float(keep.mean()),
                     "accuracy_kept": float(corr[keep].mean()) if keep.any() else None,
                     "accuracy_routed": float(corr[~keep].mean()) if (~keep).any() else None,
                     "errors_kept": int((keep & ~corr).sum()), "errors_routed": int((~keep & ~corr).sum()),
                     "min_confidence_kept": float(np.nanmin(np.where(keep, t.top[:, 0], np.nan)))
                     if keep.any() and not np.isnan(np.where(keep, t.top[:, 0], np.nan)).all() else None})


def parts_accuracy(parts: pd.DataFrame, truth: dict[str, dict[str, str]]) -> dict:
    """Per question, Jev's choice against the true part (the answer a reader right on that part gives), with the
    truth x choice counts."""
    out = {}
    for q in D.QUESTION_IDS:
        tr = parts.item_id.map(lambda i: truth[i][q])
        ch = parts[f"{q}_choice"]
        cnt = pd.crosstab(tr, ch.fillna("unusable"))
        out[q] = {"n": int(len(parts)), "accuracy": float((ch == tr).mean()) if len(parts) else None,
                  "unusable": int(ch.isna().sum()),
                  "counts": {str(a): {str(b): int(v) for b, v in row.items() if v} for a, row in cnt.iterrows()}}
    return out


# ------------------------------------------------------------------------------------------------ misleading features
def _answers(calls: pd.DataFrame | None, ids: set[str], variant: str | None) -> dict[str, pd.Series]:
    if calls is None:
        return {}
    c = calls[(calls["repeat"] == 0) & calls.item_id.isin(ids)]
    if variant is not None:
        c = c[c.variant == variant]
    c = c[c.error.fillna("") != KA.BATCH_MISSING] if "error" in c else c
    return {s: g.set_index("item_id").apply(lambda r: r.answer if r.valid else None, axis=1)
            for s, g in c.groupby("system")}


def _count(truth: pd.Series, answer: pd.Series, misread: pd.Series | None) -> dict:
    a = answer.reindex(truth.index)
    use = a.notna()
    right = use & (a == truth)
    mis = use & ~right & (a == misread) if misread is not None else pd.Series(False, index=truth.index)
    out = {"n": int(len(truth)), "correct": int(right.sum())}
    if misread is not None:
        out["misread"] = int(mis.sum())
    out.update(other=int((use & ~right & ~mis).sum()), unusable=int((~use).sum()))
    return out


def misread_table(jev_calls: pd.DataFrame, chat_confuser_calls: pd.DataFrame | None, conf_items: pd.DataFrame,
                  main_frame: pd.DataFrame, main_calls: pd.DataFrame, systems: list[str]) -> dict:
    """Counts in the format of results/arm3_confuser/misread.json. Types 1-4 per type and system: correct, misread (the
    level a reader who falls for the feature gives, confuser_items.csv misread_level), other, unusable; and a
    same-level control: over the type's reports, the mean share of the main reports of the same true level that the
    system answered at the report's misread level. Type 5 per feature. Type 6, every system with main answers: the main
    reports with tumor deposits beside involved nodes, split by whether adding the deposits to the node count changes
    the level (misread: that level)."""
    ids = set(conf_items.item_id)
    ans = {**_answers(chat_confuser_calls, ids, "names"), **_answers(jev_calls, ids, None)}
    ans = {s: ans[s] for s in systems if s in ans}
    ci = conf_items.set_index("item_id")
    main_ans = _answers(main_calls, set(main_frame.item_id), None)
    mtruth = main_frame.set_index("item_id").truth
    out = {"note": "descriptive; counts of reports; systems without stored answers are left out",
           "types_1_4": {}, "type_5": {}, "type_6": {}}
    for k in (1, 2, 3, 4):
        g = ci[ci.confuser_type.astype(int) == k]
        sysd = {}
        for s, a in ans.items():
            r = _count(g.truth, a, g.misread_level)
            ma = main_ans.get(s)
            if ma is not None:
                shares = [float((ma.reindex(mtruth.index)[mtruth == lv] == mr).mean()) for lv, mr in zip(g.truth, g.misread_level)]
                r["control_same_level_main"] = float(np.mean(shares)) if shares else None
            sysd[s] = r
        out["types_1_4"][str(k)] = {"feature": str(g.confuser_feature.iloc[0]) if len(g) else None, "systems": sysd}
    g5 = ci[ci.confuser_type.astype(int) == 5]
    for f, g in g5.groupby("confuser_feature"):
        out["type_5"][str(f)] = {s: _count(g.truth, a, None) for s, a in ans.items()}
    items = pd.read_csv(ROOT / D.CFG["items"]["main"], dtype={"report_id": str}, keep_default_na=False).set_index("report_id")
    t6 = items[(items.nodes_involved > 0) & (items.tumour_deposits > 0)]
    added = pd.Series({i: crc.level_of(crc.stage_group(r.t, crc.n_category(r.nodes_involved + r.tumour_deposits, 0), r.m))
                       for i, r in t6.iterrows()}, dtype=object)
    changes = added != t6.level
    for name, m in (("changes_level", changes), ("level_unchanged", ~changes)):
        g = t6[m]
        out["type_6"][name] = {s: _count(g.level, main_ans[s], added[m])
                               for s in K.arm_systems(D.source_arm()) if s in main_ans}
    return KA.clean(out)


# ------------------------------------------------------------------------------------------------ main
def _timing(root: Path) -> pd.DataFrame | None:
    frames = []
    p = ROOT / A["chatbot_timing"]
    if p.exists():
        t = pd.read_parquet(p)
        frames.append(t[t.system != "jev"])
    q = root / D.CFG["results"] / "timing_sample.parquet"
    if q.exists():
        frames.append(pd.read_parquet(q))
    return pd.concat(frames, ignore_index=True) if frames else None


def check_reference(res: dict, variant: str) -> dict:
    """Every chatbot's exact accuracy in this analysis must equal arm 3's stored analysis (the chatbots' answers and the
    scoring are unchanged)."""
    ref = json.loads((ROOT / A["reference_analysis"][variant]).read_text(encoding="utf-8"))
    assert ref["variant"] == variant, (ref["variant"], variant)
    got = {}
    for s, r in res["systems"].items():
        if s == "jev" or s not in ref["systems"]:
            continue
        a, b = r["exact_accuracy"], ref["systems"][s]["exact_accuracy"]
        if abs(a - b) > 1e-12:
            raise SystemExit(f"checker: {s} {variant} exact accuracy {a} differs from arm 3's {b}")
        got[s] = a
    return got


def analyze_variant(jv: str, root: Path, jev_all: pd.DataFrame, parts_all: pd.DataFrame, chat: pd.DataFrame,
                    chat_c: pd.DataFrame | None, cpath: Path, cfg: dict, n_boot: int | None, timing, rep,
                    items: list[K.Item], allow_incomplete: bool) -> dict:
    """One Jev version's outputs in results/arm3_docuse/<jv>/, against the chatbots' names and definitions answers."""
    res_dir = root / D.CFG["results"] / jv
    res_dir.mkdir(parents=True, exist_ok=True)
    arm = D.arm_spec()
    jev = jev_all[jev_all.variant == jv]
    parts = parts_all[parts_all.variant == jv]
    main0 = jev[jev["repeat"] == 0]
    missing = sorted({i.item_id for i in items if i.variant == jv} - set(main0.item_id))
    if missing and not allow_incomplete:
        raise SystemExit(f"{jv}: {len(missing)} reports have no stored Jev answer (e.g. {missing[:5]}); run --full "
                         "again, or --allow_incomplete to score them as unusable")
    written, summary = [], {"checker": {}, "files": {}}
    for v in A["chatbot_variants"]:
        comb = combined(chat, jev, jv, v)
        frame = main_items_frame(v)
        present = [s for s in chat[chat.variant == v].system.unique() if s != "jev"]
        known = [s for s in K.arm_systems(D.source_arm()) if s != "jev"]
        order = ["jev"] + [s for s in known if s in present] + sorted(set(present) - set(known))
        tiers = {s: ("jev" if s == "jev" else system_tier(s)) for s in order}
        res = KA.analyze(comb, frame, cfg, LEVELS, order, tiers, MODELS, rep, timing, v)
        res.update(arm=arm.name, jev=f"documented use, version {jv} (config/docuse_staging.yaml)", jev_variant=jv,
                   chatbots=f"arm 3 {v} answers as stored",
                   missing_treated_as_unusable={s: c["missing"] for s, c in res["completeness"].items() if c["missing"]},
                   simulated=bool(comb.simulated.fillna(False).astype(bool).any()))
        summary["checker"][v] = check_reference(res, v)
        written.append(res_dir / f"analysis_{v}.json")
        written[-1].write_text(json.dumps(res, indent=1), encoding="utf-8")
        cheap = ["jev"] + [s for s in A["cheaper_family"] if s in present]
        rc = KA.analyze(comb, frame, cfg, LEVELS, cheap, {"jev": "jev", **{s: "primary" for s in cheap[1:]}}, MODELS,
                        rep, timing, v)
        rc.update(arm=arm.name, jev_variant=jv,
                  family="the five cheaper systems as the Holm family (tier 'primary' here means family)",
                  simulated=res["simulated"])
        written.append(res_dir / f"cheap_family_{v}.json")
        written[-1].write_text(json.dumps(rc, indent=1), encoding="utf-8")
        st = {"variant": v, "jev_variant": jv,
              "systems": staging_extra(comb, frame, order, v, int(cfg["analysis"]["n_boot"]), int(cfg["seed"])),
              "error_filter": error_filter(comb, frame, v),
              "cost_per_1000_list": {s: res["systems"][s]["cost"]["usd_list_per_1000_answers"] for s in order},
              "simulated": res["simulated"]}
        written.append(res_dir / f"staging_{v}.json")
        written[-1].write_text(json.dumps(st, indent=1), encoding="utf-8")
        j = res["systems"]["jev"]
        f = f"{jv}/analysis_{v}.json"
        summary["files"][v] = {
            "jev_exact_accuracy": [j["exact_accuracy"], f"{f} systems.jev.exact_accuracy"],
            "jev_exact_accuracy_ci": [j["exact_accuracy_ci"], f"{f} systems.jev.exact_accuracy_ci"],
            "jev_qwk": [j["qwk"], f"{f} systems.jev.qwk"],
            "jev_rps": [j["confidence"].get("rps"), f"{f} systems.jev.confidence.rps"],
            "jev_main_stage_accuracy": [st["systems"]["jev"]["main_stage_accuracy"],
                                        f"{jv}/staging_{v}.json systems.jev.main_stage_accuracy"],
            "jev_error_filter_accuracy_kept": [st["error_filter"]["accuracy_kept"],
                                               f"{jv}/staging_{v}.json error_filter.accuracy_kept"],
            "outcomes_main_family": {s: c["outcome"] for s, c in res["comparisons"].items() if c["confirmatory"]},
            "outcomes_cheap_family": {s: c["outcome"] for s, c in rc["comparisons"].items() if c["confirmatory"]},
            "jev_usd_list_per_1000": [j["cost"]["usd_list_per_1000_answers"],
                                      f"{f} systems.jev.cost.usd_list_per_1000_answers"]}
    truth = D.truth_parts_of_frames()
    pa = {"jev_variant": jv, "main_pass": parts_accuracy(parts[parts["repeat"] == 0], truth),
          "main_reports": parts_accuracy(parts[(parts["repeat"] == 0) & ~parts.item_id.isin(set(confuser_items().item_id))],
                                         truth)}
    written.append(res_dir / "parts.json")
    written[-1].write_text(json.dumps(KA.clean(pa), indent=1), encoding="utf-8")
    # misleading-feature reports: arm3_confuser's settings (strata: confuser type), Jev and the five primary chatbots
    ci = confuser_items()
    fam = [s for s in A["confuser_family"] if chat_c is not None and s in set(chat_c.system)]
    if fam:
        cc = combined(chat_c, jev[jev.item_id.isin(set(ci.item_id))], jv, "names")
        ccfg = K.load_arm("arm3_confuser").config
        ccfg = {**ccfg, "arm": "arm3_docuse_confuser",
                "analysis": {**ccfg["analysis"], **({"n_boot": n_boot} if n_boot else {})}}
        rcf = KA.analyze(cc, ci[["item_id", "truth", "stratum", "group"]].sort_values("item_id").reset_index(drop=True),
                         ccfg, LEVELS, ["jev"] + fam, {"jev": "jev", **{s: "primary" for s in fam}}, MODELS, None, None,
                         "names")
        rcf.update(jev_variant=jv)
        written.append(res_dir / "analysis_confuser.json")
        written[-1].write_text(json.dumps(rcf, indent=1), encoding="utf-8")
    else:
        print(f"{jv}: no stored chatbot answers for the misleading-feature reports at {cpath}: misread.json has Jev only "
              "and analysis_confuser.json is not written")
    comb_names = combined(chat, jev, jv, "names")
    mis = misread_table(jev, chat_c, ci, main_items_frame("names"), comb_names, ["jev"] + list(A["confuser_family"]))
    mis["jev_variant"] = jv
    written.append(res_dir / "misread.json")
    written[-1].write_text(json.dumps(mis, indent=1), encoding="utf-8")
    summary["written"] = [str(p.relative_to(root)).replace("\\", "/") for p in written]
    return summary


def analyze(root: Path = ROOT, allow_incomplete: bool = False, n_boot: int | None = None,
            confuser_calls: Path | None = None) -> dict:
    """Every Jev version, each against the chatbots' names and definitions answers; summary.json across them."""
    res_dir = root / D.CFG["results"]
    arm = D.arm_spec()
    cfg = arm.config if n_boot is None else {**arm.config, "analysis": {**arm.config["analysis"], "n_boot": n_boot}}
    jev = pd.read_parquet(res_dir / "calls.parquet")
    parts = pd.read_parquet(res_dir / "parts.parquet")
    chat = pd.read_parquet(ROOT / A["chatbot_calls"])
    items = D.load_items()
    timing = _timing(root)
    rep = D.repeat_ids()
    cpath = Path(confuser_calls) if confuser_calls else ROOT / A["chatbot_confuser_calls"]
    chat_c = pd.read_parquet(cpath) if cpath.exists() else None
    summary = {jv: analyze_variant(jv, root, jev, parts, chat, chat_c, cpath, cfg, n_boot, timing, rep, items,
                                   allow_incomplete) for jv in D.VARIANTS}
    (res_dir / "summary.json").write_text(json.dumps(KA.clean(summary), indent=1), encoding="utf-8")
    print(json.dumps(KA.clean(summary), indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="where results/arm3_docuse is (a --simulate --out DIR run)")
    ap.add_argument("--allow_incomplete", action="store_true")
    ap.add_argument("--n_boot", type=int, default=None, help="fewer resamples for a quick check only")
    ap.add_argument("--chatbot_confuser_calls", type=Path, default=None,
                    help="the chatbots' stored misleading-feature answers (default: config analysis.chatbot_confuser_calls)")
    a = ap.parse_args(argv)
    return analyze(a.root, a.allow_incomplete, a.n_boot, a.chatbot_confuser_calls)


if __name__ == "__main__":
    main()
