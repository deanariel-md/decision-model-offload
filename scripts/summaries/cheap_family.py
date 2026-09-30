"""Jev against the five cheaper systems as a tested family. Same procedure as each arm's analysis for its five main
chatbots: paired bootstrap over items stratified by the truth stratum (2,000 resamples, the arm seed, the arm's own
strat_index), Holm across the family, one-sided alpha 0.025, margin 5 points, outcome non-inferior / worse /
inconclusive (src/jevity/categorical_analysis.holm_outcomes). The main-family Holm bounds are rebuilt first and must
equal the stored ones, so the item order and resampling are the arm's. Stored answers only.

Ten-year risk (--risk): the 983 evaluation records every system answered, paired bootstrap of the log-loss difference
over participants, margin of the arm 1 baseline prediction (results/eval/analysis.json), simultaneous 95% intervals
across the five cheaper systems (max-t).

  python scripts/summaries/cheap_family.py          -> results/summaries/cheap_family.json
  python scripts/summaries/cheap_family.py --risk   (after the first command: adds sets.risk_evaluation to that file)
Keys and strata: patient messages from the item identifier and data/arm2/items.csv (vignette_type); board examination
from data/arm2_ext/items.csv; staging from data/arm3/items.csv; kidney injury from GPT-5.6 Sol's answers in the version
with definitions (scored 1.000 in results/eicu_aki/analysis_definitions.json). Settings from config/arm2.yaml,
config/arm2_ext.yaml, config/arm3.yaml and config/eicu_aki.yaml; ten-year risk from results/eval/calls.parquet and
data/cohort.parquet.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src").resolve()))
from jevity.categorical_analysis import strat_index, metric_boot, accuracy_metric, holm_outcomes  # noqa: E402
import yaml  # noqa: E402

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
J = lambda p: json.loads(Path(p).read_text())


def aki_truth():
    ck = pd.read_parquet("results/eicu_aki/calls.parquet")
    d0 = ck[(ck["variant"] == "definitions") & (ck["repeat"] == 0)]
    g1 = d0[d0["system"] == "gpt"].set_index("item_id")["answer"]
    assert J("results/eicu_aki/analysis_definitions.json")["systems"]["gpt"]["exact_accuracy"] == 1.0
    return g1


def items_for(key):
    if key == "next_step":
        it = pd.read_csv("data/arm2/items.csv", dtype=str)
        return pd.DataFrame({"item_id": it.item_id, "truth": np.where(it.item_id.str.contains("_lv_"), "B", "A"),
                             "stratum": it.vignette_type})
    if key == "board_exam":
        it = pd.read_csv("data/arm2_ext/items.csv", dtype=str)
        return pd.DataFrame({"item_id": it.item_id, "truth": it.truth, "stratum": it.source})
    if key.startswith("staging"):
        it = pd.read_csv("data/arm3/items.csv", dtype=str)
        return pd.DataFrame({"item_id": it.report_id, "truth": it.level, "stratum": it.level})
    g = aki_truth()
    return pd.DataFrame({"item_id": g.index, "truth": g.values, "stratum": g.values})


SETS = [("next_step", "results/arm2/calls.parquet", "main", "results/arm2/analysis.json", "config/arm2.yaml"),
        ("board_exam", "results/arm2_ext/calls.parquet", "medhelm", "results/arm2_ext/analysis.json", "config/arm2_ext.yaml"),
        ("staging_names", "results/arm3/calls.parquet", "names", "results/arm3/analysis.json", "config/arm3.yaml"),
        ("staging_definitions", "results/arm3/calls.parquet", "definitions", "results/arm3/analysis_definitions.json", "config/arm3.yaml"),
        ("aki_names", "results/eicu_aki/calls.parquet", "names", "results/eicu_aki/analysis.json", "config/eicu_aki.yaml"),
        ("aki_definitions", "results/eicu_aki/calls.parquet", "definitions", "results/eicu_aki/analysis_definitions.json", "config/eicu_aki.yaml")]


def categorical(key, calls, variant, afile, cfile, orders):
    an, cfg = J(afile), yaml.safe_load(Path(cfile).read_text())
    pm, seed, margin = an["primary_metric"], int(cfg["seed"]), float(cfg["analysis"].get("margin", 0.05))
    alpha = float(cfg["analysis"].get("alpha_one_sided", 0.025))
    c = pd.read_parquet(calls)
    c = c[(c["variant"] == variant) & (c["repeat"] == 0)]
    base = items_for(key)
    for order_name, it in orders(base):
        it = it.reset_index(drop=True)
        strata = it.stratum.astype(str).to_numpy()
        C = {}
        for s in ["jev"] + MAIN + CHEAP:
            g = c[c.system == s].set_index("item_id").reindex(it.item_id)
            C[s] = (g.valid.fillna(False).astype(bool) & (g.answer == it.truth.values)).to_numpy()
            assert abs(accuracy_metric(C[s], strata, pm) - an["systems"][s][pm]) < 1e-9, (key, s)
        idx = strat_index(strata, 2000, seed)
        diff = lambda s: metric_boot(C["jev"][idx], strata, pm) - metric_boot(C[s][idx], strata, pm)
        ho = holm_outcomes({s: diff(s) for s in MAIN}, margin, alpha)
        ok = all(abs(ho[s]["holm_lower_bound"] - an["comparisons"][s]["holm_lower_bound"]) < 1e-12 and
                 ho[s]["outcome"] == an["comparisons"][s]["outcome"] for s in MAIN)
        if ok:
            hc = holm_outcomes({s: diff(s) for s in CHEAP}, margin, alpha)
            out = {}
            for s in CHEAP:
                d = diff(s)
                out[s] = {"diff": accuracy_metric(C["jev"], strata, pm) - accuracy_metric(C[s], strata, pm),
                          "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                          "holm_lower_bound": hc[s]["holm_lower_bound"], "holm_upper_bound": hc[s]["holm_upper_bound"],
                          "outcome": hc[s]["outcome"], "superior": bool(hc[s]["holm_lower_bound"] > 0)}
            return {"metric": pm, "margin": margin, "alpha_one_sided": alpha, "seed": seed, "item_order": order_name,
                    "main_family_reproduced": True, "comparisons": out}
    raise AssertionError(f"{key}: main-family Holm bounds not reproduced with any item order")


def orders(base):
    yield "items file", base
    yield "sorted ids", base.sort_values("item_id")


def main():
    res = {"note": "Jev against the five cheaper systems as their own Holm family; procedure identical to each arm's "
                   "main family (reproduced first)", "sets": {}}
    for key, calls, variant, afile, cfile in SETS:
        res["sets"][key] = categorical(key, calls, variant, afile, cfile, orders)
        r = res["sets"][key]
        print(key, r["item_order"], {s: (round(100 * v["diff"], 1), round(100 * v["holm_lower_bound"], 1),
                                         round(100 * v["holm_upper_bound"], 1), v["outcome"]) for s, v in r["comparisons"].items()})
    Path("results/summaries/cheap_family.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__" and "--risk" not in sys.argv:
    main()


def risk_eval(B=2000, seed=20260927):
    """Evaluation set: Jev minus each cheaper system in log-loss on the 983 records every system answered; paired
    bootstrap over participants; simultaneous 95% intervals across the five (max-t, as the arm 1 evaluation analysis);
    non-inferior when the simultaneous upper bound is below the evaluation-set margin stored by arm 1."""
    a = J("results/eval/analysis.json")["prediction"]
    margin = a["focal_vs_llms"]["pairs"]["jev - gpt"]["noninferiority_margin"]
    c = pd.read_parquet("results/eval/calls.parquet")
    b = c[(c.edit == "baseline") & (c.variant == "raw") & (~c.annotated) & (c.repeat == 0)]
    S = ["jev"] + MAIN + CHEAP
    piv = b[b.model.isin(S)].pivot_table(index="profile", columns="model", values="p", aggfunc="first")
    val = b[b.model.isin(S)].pivot_table(index="profile", columns="model", values="valid", aggfunc="first").fillna(False).astype(bool)
    com = val[S].all(axis=1)
    P = piv.loc[com, S].clip(0.005, 0.995)
    y = pd.read_parquet("data/cohort.parquet").set_index("SEQN")["death_10y"].reindex(P.index).astype(int).to_numpy()
    L = {s: -(y * np.log(P[s].to_numpy()) + (1 - y) * np.log(1 - P[s].to_numpy())) for s in S}
    for s in S:
        assert abs(L[s].mean() - a["systems"][s]["log_loss"]) < 1e-9, s
    rng = np.random.default_rng(seed); n = len(y)
    ix = rng.integers(0, n, size=(B, n))
    D = {s: L["jev"][ix].mean(1) - L[s][ix].mean(1) for s in CHEAP}
    est = {s: float(L["jev"].mean() - L[s].mean()) for s in CHEAP}
    se = {s: float(D[s].std()) for s in CHEAP}
    tmax = np.max(np.stack([np.abs(D[s] - est[s]) / se[s] for s in CHEAP]), axis=0)
    crit = float(np.quantile(tmax, 0.95))
    out = {}
    for s in CHEAP:
        lo, hi = est[s] - crit * se[s], est[s] + crit * se[s]
        out[s] = {"diff": est[s], "ci": [float(np.quantile(D[s], 0.025)), float(np.quantile(D[s], 0.975))],
                  "ci_simultaneous": [lo, hi], "noninferior": bool(hi < margin), "worse": bool(lo > margin),
                  "outcome": "non-inferior" if hi < margin else "worse" if lo > margin else "inconclusive"}
    return {"n_records": int(n), "deaths": int(y.sum()), "margin": margin, "critical_value": crit, "B": B, "seed": seed,
            "note": "log-loss, lower is better; difference Jev minus system", "comparisons": out}


if __name__ == "__main__" and "--risk" in sys.argv:
    res = J("results/summaries/cheap_family.json")
    res["sets"]["risk_evaluation"] = risk_eval()
    Path("results/summaries/cheap_family.json").write_text(json.dumps(res, indent=1))
    print({s: (round(v["diff"], 4), [round(x, 4) for x in v["ci_simultaneous"]], v["outcome"])
           for s, v in res["sets"]["risk_evaluation"]["comparisons"].items()}, res["sets"]["risk_evaluation"]["margin"])
