"""Arm 1, further descriptive analyses (no test in the confirmatory family; no model is called):
  1 calibration split: log-loss as returned and after cross-fitted recalibration, calibration component, mean
    predicted, AUROC (wide set, from results/wide/analysis.json; evaluation-set log-loss beside it)
  2 clinical slope with and without the smoking edit (analysis.class_beta_draws on the same descriptive draws as the
    evaluation analysis: profile resample and reference refit), with the age-equivalents in results/eval/analysis.json
  3 smoking gradient in real records: mean predicted log-odds, current minus never smokers, per system (wide set)
  5 subgroups: age 40-49/50-59/60-69/70-79, sex, race and ethnicity, family income below / at or above the
    poverty line, diabetes, smoking status, tertile of the reference's predicted risk (wide set)
  6 calibration by race and ethnicity (observed / expected; from results/wide/analysis.json)
  4 format dependence: irrelevant edits against repeat noise, Jev's Choice against Noul answers, the complement check;
  and, from results/eval/analysis.json, the M1 instruction's effect on the race edit, and two supplement tables: free
  and open tiers against their flagships; reasoning effort high against low.
  python scripts/arm1_extras.py [--B 1000]
Writes results/wide/arm1_extras.json and results/wide/arm1_extras.md."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import analysis as A                      # noqa: E402

SYSTEMS = ["jev", "gpt", "claude", "gemini", "muse", "glm"]
CHATBOTS = SYSTEMS[1:]
REFS = {"p_spline_logit": "reference_spline_logit", "p_lightgbm": "reference_lightgbm",
        "p_age_sex": "reference_age_sex", "p_phenoage_10y": "phenoage_10y"}
SEED = 20260922


def lg(p):
    p = np.clip(np.asarray(p, float), *A.CLIP)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(np.asarray(p, float), *A.CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def slope(t: pd.DataFrame, drop: tuple = ()) -> float:
    """analysis.class_beta_draws' point estimate: edit-level means, features weighted equally across their edits."""
    t = t[~t.feature.isin(drop)]
    m = t.groupby("edit").agg(z=("z_logit", "mean"), d=("d_logit", "mean"), feature=("feature", "first"))
    w = 1.0 / m.feature.map(m.feature.value_counts())
    return float(np.sum(w * m.d * m.z) / np.sum(w * m.d ** 2))


def class_weights(c: dict, t: pd.DataFrame) -> dict:
    """Each field's share of a class slope (analysis.class_beta_draws): edit weight 1/(edits in its field) x mean d^2."""
    feat = t.groupby("edit")["feature"].first()
    em = pd.DataFrame(c["edit_means"]).set_index("edit")
    f = feat.reindex(em.index)
    x = (1.0 / f.map(f.value_counts())) * em.d_logit ** 2
    return {k: float(v) for k, v in (x / x.sum()).groupby(f).sum().items()}


def complement_hundredths(calls: pd.DataFrame) -> dict:
    """analysis.jev_complement's share within 0.05 of 1, counted in whole hundredths. Jev answers on the 0.01 grid, and
    the float test |s - 1| <= 0.05 counts some sums of exactly 0.95 or 1.05 in and others out."""
    d = calls[(calls.model == "jev") & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid
              & A._variant(calls, "raw")]
    a = calls[(calls.model == "jev_complement") & calls.valid]
    m = d[["profile", "p"]].merge(a[["profile", "p"]].rename(columns={"p": "p_alive"}), on="profile")
    grid = lambda p: bool(np.allclose(p * 100, np.round(p * 100), atol=1e-9))
    h = (np.round(m.p.to_numpy(float) * 100) + np.round(m.p_alive.to_numpy(float) * 100)).astype(int)
    return {"n": int(len(h)), "on_0.01_grid": grid(m.p.to_numpy(float)) and grid(m.p_alive.to_numpy(float)),
            "within_0.05_inclusive": int((np.abs(h - 100) <= 5).sum()), "share_within_0.05_inclusive": float((np.abs(h - 100) <= 5).mean()),
            "strictly_within_0.05": int((np.abs(h - 100) < 5).sum()), "at_exactly_0.95_or_1.05": int((np.abs(h - 100) == 5).sum())}


def confirmatory_without_smoking(ev: dict, calls: pd.DataFrame, inst: pd.DataFrame, co: pd.DataFrame, refits) -> dict:
    """The confirmatory clinical slope (six family clinical edits, weights dbar^2) recomputed with the analysis's own
    primary draws (B draws, seed SEED), then without the smoking edit.
    Stops unless the all-edit slopes and simultaneous intervals reproduce results/eval/analysis.json."""
    fr = ev["meta"]["family_rules"]
    ms, fam = fr["systems_in_family"], fr["edits_in_family"]
    draws = A.Draws(sorted(co.loc[co.split == "eval", "SEQN"].astype(int)), ev["meta"]["B"], refits.R, seed=SEED)
    res = A.confirmatory_family({m: A.contrast_table(calls, inst, m, "spline_logit") for m in ms}, refits, draws, calls, co,
                                edits=fam, pairs=False, keep_draws=True)
    out = {}
    for m in ms:
        mine, reported = res["clinical_slope"][m], ev["confirmatory"]["clinical_slope"][m]
        if abs(mine["beta"] - reported["beta"]) > 1e-9 or np.max(np.abs(np.subtract(mine["ci_simultaneous"], reported["ci_simultaneous"]))) > 1e-9:
            raise SystemExit(f"confirmatory slope for {m} does not reproduce results/eval/analysis.json")
        es = [e for e in mine["fields"] if not e.startswith("C5")]
        z = np.log([res["cells"][e][m]["implied_or_model"] for e in es]); d = np.log([res["cells"][e][m]["or_reference"] for e in es])
        Z = np.column_stack([res["_draws"][(e, m)]["implied_log_or"] for e in es])
        D = np.column_stack([res["_draws"][(e, m)]["reference_log_or"] for e in es])
        bd = (Z * D).sum(1) / (D * D).sum(1)
        out[m] = {"all_edits": reported["beta"], "all_edits_ci_simultaneous": reported["ci_simultaneous"],
                  "without_smoking": float((z * d).sum() / (d * d).sum()), "without_smoking_ci": A._ci(bd),
                  "without_smoking_ci_wald": [float((z * d).sum() / (d * d).sum() - 1.96 * bd.std(ddof=1)),
                                              float((z * d).sum() / (d * d).sum() + 1.96 * bd.std(ddof=1))],
                  "fields": es}
    return out


def smoking_gradient(P: pd.DataFrame, d: pd.DataFrame, y: np.ndarray, B: int, seed: int = SEED) -> dict:
    cur, nev = np.where(d.smoking.to_numpy() == "current")[0], np.where(d.smoking.to_numpy() == "never")[0]
    rng = np.random.default_rng(seed)
    out = {"n_current": len(cur), "n_never": len(nev), "deaths_current": int(y[cur].sum()), "deaths_never": int(y[nev].sum()),
           "observed_risk_current": float(y[cur].mean()), "observed_risk_never": float(y[nev].mean()), "systems": {}}
    draws = [(rng.choice(cur, len(cur)), rng.choice(nev, len(nev))) for _ in range(B)]
    for m in P.columns:
        L = lg(P[m].to_numpy())
        out["systems"][m] = {"log_odds_gap": float(L[cur].mean() - L[nev].mean()),
                             "log_odds_gap_ci": A._ci([L[a].mean() - L[b].mean() for a, b in draws]),
                             "mean_predicted_current": float(P[m].to_numpy()[cur].mean()),
                             "mean_predicted_never": float(P[m].to_numpy()[nev].mean())}
    return out


def subgroups(d: pd.DataFrame) -> dict[str, np.ndarray]:
    g = {}
    for lo in (40, 50, 60, 70):
        g[f"age {lo}-{lo + 9}"] = ((d.age >= lo) & (d.age < lo + 10)).to_numpy()
    for s in ("male", "female"):
        g[f"sex {s}"] = (d.sex == s).to_numpy()
    for r in sorted(d.race_ethnicity.dropna().unique()):
        g[f"race {r}"] = (d.race_ethnicity == r).to_numpy()
    g["income below poverty line"] = (d.income_poverty_ratio < 1).to_numpy()
    g["income at or above poverty line"] = (d.income_poverty_ratio >= 1).to_numpy()
    dia = d.diabetes.astype(str).str.lower().isin(["yes", "1", "1.0", "true"]).to_numpy()
    g["diabetes"], g["no diabetes"] = dia, ~dia
    for s in ("never", "former", "current"):
        g[f"smoking {s}"] = (d.smoking == s).to_numpy()
    return g


def subgroup_block(P: pd.DataFrame, y: np.ndarray, groups: dict, B: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    L = {m: ll(P[m].to_numpy(), y) for m in P.columns}
    rows = []
    for name, mask in groups.items():
        ix = np.where(mask)[0]
        if len(ix) < 100:
            continue
        r = {"group": name, "n": len(ix), "deaths": int(y[ix].sum()), "observed_risk": float(y[ix].mean()), "systems": {}}
        bs = [rng.choice(ix, len(ix)) for _ in range(B)]
        for m in P.columns:
            p = P[m].to_numpy()[ix]
            r["systems"][m] = {"log_loss": float(L[m][ix].mean()), "mean_predicted": float(p.mean()),
                               "auroc": float(roc_auc_score(y[ix], p)) if 0 < y[ix].sum() < len(ix) else None}
        for c in CHATBOTS:
            dlt = L["jev"] - L[c]
            r["systems"][c]["jev_minus_system"] = float(dlt[ix].mean())
            r["systems"][c]["jev_minus_system_ci"] = A._ci([dlt[b].mean() for b in bs])
        mc = np.mean([L[c] for c in CHATBOTS], axis=0)
        r["jev_minus_mean_chatbot"] = float((L["jev"] - mc)[ix].mean())
        r["jev_minus_mean_chatbot_ci"] = A._ci([(L["jev"] - mc)[b].mean() for b in bs])
        rows.append(r)
    return rows


def wide_frame():
    calls = pd.read_parquet(ROOT / "results" / "wide" / "calls.parquet", columns=["model", "profile", "edit", "valid", "variant", "p"])
    c = calls[(calls.edit == "baseline") & calls.valid & (calls.variant == "raw")].drop_duplicates(["model", "profile"])
    P = c.pivot(index="profile", columns="model", values="p")[SYSTEMS]
    o = pd.read_parquet(ROOT / "data" / "reference_oof.parquet").set_index("SEQN")
    for col, name in REFS.items():
        P[name] = o[col]
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    co = co.set_index(co.SEQN.astype(int))
    P = P.dropna()
    P = P[P.index.isin(co.index) & co.death_10y.reindex(P.index).notna()]
    return P, co.loc[P.index], co.loc[P.index, "death_10y"].astype(float).to_numpy()


def main(B: int) -> dict:
    wide = json.loads((ROOT / "results" / "wide" / "analysis.json").read_text())
    ev = json.loads((ROOT / "results" / "eval" / "analysis.json").read_text())
    P, d, y = wide_frame()
    out = {"meta": {"added_after_results": True, "note": "descriptive; outside the fixed family; no model called",
                    "n_wide": len(P), "deaths_wide": int(y.sum()), "B": B, "seed": SEED}}
    # 1 calibration split
    ws, es = wide["prediction"]["systems"], ev["prediction"]["systems"]
    out["calibration_split"] = {m: {k: ws[m][k] for k in ("log_loss", "log_loss_recalibrated", "calibration_component",
                                                          "mean_p", "auroc", "calibration_slope", "calibration_intercept")}
                                | {"eval_log_loss": es.get(m, {}).get("log_loss")} for m in ws}
    out["calibration_split_meta"] = {"n": wide["prediction"]["n_records"], "deaths": wide["prediction"]["deaths"],
                                     "jev_vs_chatbots_recalibrated": wide["prediction"]["focal_vs_llms_recalibrated"]["pairs"],
                                     "jev_vs_age_sex": wide["prediction"]["focal_vs_age_sex"]["pairs"]}
    # 2 slope with and without smoking, and the age-equivalents
    calls = pd.read_parquet(ROOT / "results" / "eval" / "calls.parquet")
    inst = pd.read_parquet(ROOT / "data" / "instances_eval.parquet")
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    refits = A.Refits.load(ROOT / "data" / "refit_d_spline_logit_eval.npz")
    ddesc = A.Draws(sorted(co.loc[co.split == "eval", "SEQN"].astype(int)), ev["meta"]["B_descriptive"], refits.R, seed=17)
    out["slope"] = {}
    for m in SYSTEMS:        # the analysis's own descriptive draws (profile resample + reference refit), as in run_all
        t = A.contrast_table(calls, inst, m, "spline_logit")
        blk = {}
        for name, drop in (("all_clinical_edits", ()), ("without_smoking", ("smoking",)),
                           ("without_smoking_and_age", ("smoking", "age"))):
            td = t[~t.feature.isin(drop)]
            c = A.class_beta_draws(td, refits, ddesc)["clinical"]
            blk[name] = {"slope": c["beta"], "ci": c["beta_ci"], "field_weights": class_weights(c, td[td.klass == "clinical"])}
        blk["reported_class_slope"] = ev["classes"][m]["clinical"]["beta"]
        blk["reported_class_slope_ci"] = ev["classes"][m]["clinical"]["beta_ci"]
        out["slope"][m] = blk
    out["confirmatory_slope_without_smoking"] = cw = confirmatory_without_smoking(ev, calls, inst, co, refits)
    cb = [cw[m]["without_smoking"] for m in CHATBOTS if m in cw]
    out["confirmatory_slope_without_smoking_chatbots"] = {"min": min(cb), "max": max(cb), "n": len(cb)}
    eq = ev["headline_metrics"]["equivalents"]["cells"]
    out["age_equivalents"] = {e: {m: {"excess_in_years_of_age": eq[e][m]["excess_in_years_of_age"],
                                      "ci": eq[e][m]["excess_in_years_of_age_ci"]} for m in SYSTEMS if m in eq[e]}
                              for e in ("C5_smoking_never_to_current", "L5a_sex_male_to_female", "C6_age_plus5")}
    # 3 smoking gradient in real records
    out["smoking_gradient"] = smoking_gradient(P[SYSTEMS + ["reference_spline_logit"]], d, y, B)
    # 5 subgroups plus reference-risk tertiles
    g = subgroups(d)
    t3 = pd.qcut(P.reference_spline_logit, 3, labels=["low", "middle", "high"]).to_numpy()
    for k in ("low", "middle", "high"):
        g[f"reference risk tertile {k}"] = t3 == k
    out["subgroups"] = subgroup_block(P[SYSTEMS + ["reference_spline_logit", "reference_age_sex"]], y, g, B)
    dia = d.diabetes.astype(str).str.lower()
    out["subgroups_meta"] = {"no_diabetes_means_not_coded_yes": {"no": int((dia == "no").sum()), "borderline": int((dia == "borderline").sum()),
                                                                 "missing": int((~dia.isin(["yes", "no", "borderline"])).sum())},
                             "income_missing": int(d.income_poverty_ratio.isna().sum()), "smoking_missing": int(d.smoking.isna().sum())}
    # 6 calibration by race and ethnicity; M1 on the race edit
    out["race_calibration"] = wide["group_calibration"]
    out["m1_race_edit"] = ev["mitigation"]["M1_minus_raw"]["L1_race_nhw_to_nhb"]
    # 4 format dependence: irrelevant edits against repeat noise; Choice vs Noul; complement
    out["format_dependence"] = {
        "drift": {m: {k: ev["drift"][m][k] for k in ("mean_abs_drift_pts", "mean_abs_repeat_pts", "excess_pts",
                                                     "excess_pts_ci", "exceeds_operational_2pts", "by_edit")} for m in SYSTEMS},
        "jev_bands_vs_noul": {k: {c: {"beta": v["beta"], "beta_ci": v["beta_ci"]} for c, v in ev["jev_bands_vs_noul"][k].items()
                                  if isinstance(v, dict) and "beta" in v} for k in ("bands", "noul_same_instances")},
        "jev_complement": ev["jev_complement"], "jev_complement_hundredths": complement_hundredths(calls)}
    # supplement tables: free and open tiers against their flagships; reasoning effort high against low
    out["supp_tiers"] = {k: {"clinical_slope": v["clinical_slope"], "per_call_error_diff": v["per_call_error_diff"]}
                         for k, v in ev["tier_contrasts"].items()}
    out["supp_reasoning_effort"] = {m: {"n_profiles": ev["reasoning_arm"][m]["n_profiles"],
                                        "clinical_slope": ev["reasoning_arm"][m]["high_minus_low"]["clinical_slope"],
                                        "per_call_error_diff": ev["reasoning_arm"][m]["high_minus_low"]["per_call_error_diff"]}
                                    for m in ev["reasoning_arm"]}
    return out


def tables(r: dict) -> str:
    L = ["# Arm 1, further descriptive analyses (added after the results; outside the fixed family)", ""]
    L += ["## 1. Calibration split (wide set, n = %d, %d deaths)" % (r["calibration_split_meta"]["n"], r["calibration_split_meta"]["deaths"]), "",
          "| Predictor | Log-loss | Recalibrated | Calibration component | Mean predicted | AUROC | Evaluation-set log-loss |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for m, v in r["calibration_split"].items():
        e = v["eval_log_loss"]
        L.append(f"| {m} | {v['log_loss']:.4f} | {v['log_loss_recalibrated']:.4f} | {v['calibration_component']:.4f} | "
                 f"{v['mean_p']:.3f} | {v['auroc']:.3f} | {'–' if e is None else format(e, '.4f')} |")
    L += ["", "## 2. Clinical slope with and without the smoking edit (evaluation set; profile resample and reference refit)", "",
          "| System | All clinical edits | Without smoking | Without smoking and age | Reported class slope |", "|---|---|---|---|---:|"]
    for m, v in r["slope"].items():
        f = lambda k: f"{v[k]['slope']:.2f} ({v[k]['ci'][0]:.2f} to {v[k]['ci'][1]:.2f})"
        L.append(f"| {m} | {f('all_clinical_edits')} | {f('without_smoking')} | {f('without_smoking_and_age')} | {v['reported_class_slope']:.2f} |")
    fw = r["slope"]["jev"]["all_clinical_edits"]["field_weights"]
    L += ["", "The descriptive class slope (all clinical edits, results/eval/analysis.json classes), not the confirmatory slope "
          "(six family edits, headlines). Field weights of the class slope, all clinical edits: "
          + ", ".join(f"{k} {v:.1%}" for k, v in sorted(fw.items(), key=lambda kv: -kv[1])) + "."]
    cw = r["confirmatory_slope_without_smoking"]
    L += ["", "Confirmatory slope (six family clinical edits; the analysis's primary draws), all edits (simultaneous 95%) and "
          "without smoking (marginal percentile 95%): " +
          "; ".join(f"{m} {v['all_edits']:.2f} ({v['all_edits_ci_simultaneous'][0]:.2f} to {v['all_edits_ci_simultaneous'][1]:.2f}), "
                    f"{v['without_smoking']:.2f} ({v['without_smoking_ci'][0]:.2f} to {v['without_smoking_ci'][1]:.2f})" for m, v in cw.items()) + "."]
    L += ["", "Age-equivalents (excess in years of age, marginal 95% intervals, from results/eval/analysis.json):"]
    for e, v in r["age_equivalents"].items():
        L.append(f"- {e}: " + "; ".join(f"{m} {x['excess_in_years_of_age']:+.1f} ({x['ci'][0]:+.1f} to {x['ci'][1]:+.1f})" for m, x in v.items()))
    s = r["smoking_gradient"]
    L += ["", f"## 3. Smoking gradient in real records (current {s['n_current']:,}, never {s['n_never']:,}; observed risk "
          f"{s['observed_risk_current']:.3f} and {s['observed_risk_never']:.3f})", "",
          "| Predictor | Mean predicted log-odds, current minus never | Mean predicted, current | Mean predicted, never |", "|---|---|---:|---:|"]
    for m, v in s["systems"].items():
        L.append(f"| {m} | {v['log_odds_gap']:.2f} ({v['log_odds_gap_ci'][0]:.2f} to {v['log_odds_gap_ci'][1]:.2f}) | "
                 f"{v['mean_predicted_current']:.3f} | {v['mean_predicted_never']:.3f} |")
    L += ["", "## 5. Subgroups (fixed list; wide set)", "",
          "| Group | n | Deaths | Observed | Jev mean predicted | Jev log-loss | Chatbot log-loss (range) | Jev minus mean chatbot | Jev AUROC | Best chatbot AUROC |",
          "|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for g in r["subgroups"]:
        sy = g["systems"]; cl = [sy[c]["log_loss"] for c in CHATBOTS]; au = [sy[c]["auroc"] for c in CHATBOTS if sy[c]["auroc"] is not None]
        ci = g["jev_minus_mean_chatbot_ci"]
        L.append(f"| {g['group']} | {g['n']:,} | {g['deaths']:,} | {g['observed_risk']:.3f} | {sy['jev']['mean_predicted']:.3f} | "
                 f"{sy['jev']['log_loss']:.3f} | {min(cl):.3f}–{max(cl):.3f} | {g['jev_minus_mean_chatbot']:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f}) | "
                 f"{sy['jev']['auroc']:.3f} | {max(au):.3f} |")
    sm = r["subgroups_meta"]; nd = sm["no_diabetes_means_not_coded_yes"]
    L += ["", f"'no diabetes' is every record not coded yes ({nd['no']:,} no, {nd['borderline']:,} borderline, {nd['missing']:,} "
          f"missing). Records with income ({sm['income_missing']:,}) or smoking ({sm['smoking_missing']:,}) missing are in no "
          "income or smoking group."]
    L += ["", "## 6. Calibration by race and ethnicity (observed / expected deaths; wide run, each system's own valid answers "
          "on the 12,910 wide records, not the common prediction set)", "",
          "| Predictor | Non-Hispanic White | Non-Hispanic Black | Mexican American | Black / White ratio | Ratio, race field removed |", "|---|---:|---:|---:|---|---|"]
    for m, v in r["race_calibration"].items():
        raw = v["raw"]; rr = raw["ratio_NHB_to_NHW"]; m2 = v.get("M2", {}).get("ratio_NHB_to_NHW")
        L.append(f"| {m} | {raw['Non-Hispanic White']['O_E']:.2f} | {raw['Non-Hispanic Black']['O_E']:.2f} | {raw['Mexican American']['O_E']:.2f} | "
                 f"{rr['est']:.2f} ({rr['ci'][0]:.2f} to {rr['ci'][1]:.2f}) | " + (f"{m2['est']:.2f} ({m2['ci'][0]:.2f} to {m2['ci'][1]:.2f})" if m2 else "–") + " |")
    L += ["", "M1 instruction on the race edit (instruction minus raw, log-odds): " +
          "; ".join(f"{m} {v['diff_logit']:+.3f} ({v['diff_logit_ci'][0]:+.3f} to {v['diff_logit_ci'][1]:+.3f})" for m, v in r["m1_race_edit"].items() if m in SYSTEMS)]
    f = r["format_dependence"]
    L += ["", "## 4. Format dependence (evaluation set)", "",
          "Irrelevant edits (field order, units, prose) against asking twice: mean absolute change in percentage points.", "",
          "| System | Field order | Units | Prose (signed) | Repeat | Excess over repeat |", "|---|---:|---:|---|---:|---|"]
    for m, v in f["drift"].items():
        b = v["by_edit"]; ci = v["excess_pts_ci"]
        L.append(f"| {m} | {b['I1_field_order']['mean_abs_pts']:.1f} | {b['I2_units']['mean_abs_pts']:.1f} | "
                 f"{b['I3_prose']['mean_abs_pts']:.1f} ({b['I3_prose']['mean_signed_pts']:+.1f}) | {v['mean_abs_repeat_pts']:.1f} | "
                 f"{v['excess_pts']:+.2f} ({ci[0]:+.2f} to {ci[1]:+.2f}) |")
    bv = f["jev_bands_vs_noul"]; jc = f["jev_complement"]; jh = f["jev_complement_hundredths"]
    L += ["", "Jev, the same edited records asked two ways (the 300 evaluation records asked for risk bands; class slope of "
          "Jev's change on the supported change): " +
          "; ".join(f"{k} {c} {v['beta']:.2f} ({v['beta_ci'][0]:.2f} to {v['beta_ci'][1]:.2f})" for k, cs in bv.items() for c, v in cs.items()),
          f"Jev complement ({jh['n']} baselines): P(die) + P(alive) mean {jc['sum_mean']:.3f}, range {jc['sum_range'][0]:.2f} to "
          f"{jc['sum_range'][1]:.2f}; {jh['within_0.05_inclusive']} of {jh['n']} ({jh['share_within_0.05_inclusive']:.1%}) within 0.05 "
          f"of 1, counted in whole hundredths ({jh['at_exactly_0.95_or_1.05']} at exactly 0.95 or 1.05; "
          f"analysis.json's float test gives {jc['share_within_0.05_of_1']:.0%})."]
    L += ["", "## Supplement S1. Free and open tiers against their flagships (evaluation set; clinical slope, confirmatory edits)", "",
          "Slope: the confirmatory clinical slope (six family clinical edits, weights proportional to the squared supported "
          "change). Difference: first minus second. Per-call error: mean |system's log-odds change - supported change| over "
          "the supported record-edits of the ten family edits.", "",
          "| Contrast (first - second) | Slope, first | Slope, second | Difference | Per-call error difference |", "|---|---:|---:|---|---|"]
    for k, v in r["supp_tiers"].items():
        s, e = v["clinical_slope"], v["per_call_error_diff"]
        L.append(f"| {k} | {s['a']:.2f} | {s['b']:.2f} | {s['diff']:+.2f} ({s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}) | "
                 f"{e['diff']:+.3f} ({e['ci'][0]:+.3f} to {e['ci'][1]:+.3f}) |")
    L += ["", "## Supplement S2. Reasoning effort, high against low (fixed 200-record subset; slope and per-call error as in S1)", "",
          "| System | Slope, high | Slope, low | Difference | Per-call error difference |", "|---|---:|---:|---|---|"]
    for m, v in r["supp_reasoning_effort"].items():
        s, e = v["clinical_slope"], v["per_call_error_diff"]
        L.append(f"| {m} | {s['a']:.2f} | {s['b']:.2f} | {s['diff']:+.2f} ({s['ci'][0]:+.2f} to {s['ci'][1]:+.2f}) | "
                 f"{e['diff']:+.3f} ({e['ci'][0]:+.3f} to {e['ci'][1]:+.3f}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    res = main(a.B)
    (ROOT / "results" / "wide" / "arm1_extras.json").write_text(json.dumps(res, indent=1, default=float))
    md = tables(res)
    (ROOT / "results" / "wide" / "arm1_extras.md").write_text(md, encoding="utf-8")
    print(md)
