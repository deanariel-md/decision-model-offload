"""Arm 3 cancer staging: summary numbers from the stored outputs.

Ranges over the five main chatbots, the definitions version, the hybrid and random-half control, cost, and one
descriptive count computed here: stage III reports Jev placed in stage 0-II, and IVC reports it placed in IVA-IVB.
Recomputed exact accuracy must equal analysis.json.

  python scripts/summaries/arm3_summary.py [--dir results/arm3] [--items data/arm3/items.csv]
    -> results/summaries/arm3_summary.json
"""
import argparse, json
from pathlib import Path
import pandas as pd

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
FREE = ["gpt_free", "claude_free", "gemini_free"]
GROUP = {"0": "0", "I": "I-II", "IIA": "I-II", "IIB": "I-II", "IIC": "I-II", "IIIA": "III", "IIIB": "III", "IIIC": "III",
         "IVA-IVB": "IV", "IVC": "IV"}


def rng(vals):
    v = list(vals)
    return {"min": min(v), "max": max(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arm3")
    ap.add_argument("--items", default="data/arm3/items.csv")
    ap.add_argument("--out", default="results/summaries/arm3_summary.json")
    a = ap.parse_args()
    d = Path(a.dir)
    an = json.loads((d / "analysis.json").read_text())
    df = json.loads((d / "analysis_definitions.json").read_text())
    assert an["variant"] == "names" and df["variant"] == "definitions"
    S, C = an["systems"], an["comparisons"]

    calls = pd.read_parquet(d / "calls.parquet")
    items = pd.read_csv(a.items, dtype=str)
    truth = dict(zip(items["report_id"], items["level"]))
    c = calls[(calls["variant"] == "names") & (calls["repeat"] == 0)].copy()
    c["truth"] = c["item_id"].map(truth)
    c["ok"] = c["valid"] & (c["answer"] == c["truth"])
    for s, g in c.groupby("system"):
        assert abs(g["ok"].mean() - S[s]["exact_accuracy"]) < 1e-9, (s, g["ok"].mean(), S[s]["exact_accuracy"])

    def cross(s):
        g = c[(c["system"] == s) & c["valid"]]
        tg, ag = g["truth"].map(GROUP), g["answer"].map(GROUP)
        return {"stage_iii_n": int((tg == "III").sum()),
                "stage_iii_to_0_ii": int(((tg == "III") & ag.isin(["0", "I-II"])).sum()),
                "stage_0_ii_to_iii": int((tg.isin(["0", "I-II"]) & (ag == "III")).sum()),
                "ivc_n": int((g["truth"] == "IVC").sum()),
                "ivc_to_iva_ivb": int(((g["truth"] == "IVC") & (g["answer"] == "IVA-IVB")).sum()),
                "stage_iv_to_lower": int(((tg == "IV") & (ag != "IV")).sum()),
                "errors": int((g["answer"] != g["truth"]).sum()),
                "errors_within_0_ii": int((tg.isin(["0", "I-II"]) & ag.isin(["0", "I-II"]) & (g["answer"] != g["truth"])).sum())}

    dcalls = calls[(calls["variant"] == "definitions") & (calls["repeat"] == 0)]
    dids = set(dcalls["item_id"])
    assert len(dids) == df["n_items"]
    names_on_d = {s: float(g.loc[g["item_id"].isin(dids), "ok"].mean()) for s, g in c.groupby("system")}
    jc = cross("jev")
    jc["stage_iii_to_0_ii_share"] = jc["stage_iii_to_0_ii"] / jc["stage_iii_n"]
    jc["ivc_to_iva_ivb_share"] = jc["ivc_to_iva_ivb"] / jc["ivc_n"]
    mc = {s: cross(s) for s in MAIN}

    hy, ro = an["hybrid_cost"], an["routing"]["systems"]
    for s in MAIN:  # the tested hybrid and the cost table agree
        hs = an["hybrid_share"]["comparisons"][s]
        assert abs(hs["exact_accuracy"] - hy[s]["exact_accuracy"]) < 1e-9 and abs(hs["diff"] - (hy[s]["exact_accuracy"] - hy[s]["chatbot_alone"])) < 1e-9
    jl = S["jev"]["cost"]["usd_list_per_1000_answers"]
    res = {
        "note": "descriptive; names version is primary; cost at standard list price",
        "n_items": an["n_items"], "n_definitions": df["n_items"],
        "jev": {"exact_accuracy": S["jev"]["exact_accuracy"], "exact_accuracy_ci": S["jev"]["exact_accuracy_ci"],
                "qwk": S["jev"]["qwk"], "definitions_exact_accuracy": df["systems"]["jev"]["exact_accuracy"],
                "definitions_exact_accuracy_ci": df["systems"]["jev"]["exact_accuracy_ci"],
                "names_on_definition_reports": names_on_d["jev"],
                "definitions_gain_pts": 100 * (df["systems"]["jev"]["exact_accuracy"] - names_on_d["jev"]),
                "top_prob_max": float(c.loc[c["system"] == "jev", "top_prob"].max()),
                "top_prob_median": float(c.loc[c["system"] == "jev", "top_prob"].median()),
                "usd_list_per_1000": jl, "by_level": S["jev"]["accuracy_by_stratum"], "crossings": jc},
        "main_chatbots": {
            "exact_accuracy": rng(S[s]["exact_accuracy"] for s in MAIN),
            "qwk": rng(S[s]["qwk"] for s in MAIN),
            "jev_minus_chatbot_pts": rng(100 * C[s]["diff"] for s in MAIN),
            "jev_minus_chatbot_abs_pts": rng(-100 * C[s]["diff"] for s in MAIN),
            "all_worse": all(C[s]["outcome"] == "worse" for s in MAIN),
            "definitions_exact_accuracy": rng(df["systems"][s]["exact_accuracy"] for s in MAIN),
            "names_on_definition_reports": rng(names_on_d[s] for s in MAIN),
            "usd_list_per_1000": rng(S[s]["cost"]["usd_list_per_1000_answers"] for s in MAIN),
            "cost_ratio_to_jev": rng(S[s]["cost"]["usd_list_per_1000_answers"] / jl for s in MAIN),
            "hybrid_exact_accuracy": rng(hy[s]["exact_accuracy"] for s in MAIN),
            "hybrid_minus_chatbot_pts": rng(100 * (hy[s]["exact_accuracy"] - hy[s]["chatbot_alone"]) for s in MAIN),
            "hybrid_minus_chatbot_abs_pts": rng(100 * (hy[s]["chatbot_alone"] - hy[s]["exact_accuracy"]) for s in MAIN),
            "hybrid_noninferior_any": any(an["hybrid_share"]["comparisons"][s]["noninferior"] for s in MAIN),
            "hybrid_all_worse": all(an["hybrid_share"]["comparisons"][s]["outcome"] == "worse" for s in MAIN),
            "hybrid_over_chatbot_cost": rng(hy[s]["hybrid_over_chatbot"] for s in MAIN),
            "confidence_minus_random_pts": rng(100 * ro[s]["exact_accuracy"]["diff"] for s in MAIN),
            "crossings": {k: sum(mc[s][k] for s in MAIN) for k in ("stage_iii_to_0_ii", "stage_0_ii_to_iii", "stage_iv_to_lower")}},
        "definitions_hybrid": {
            "exact_accuracy": rng(df["hybrid_share"]["comparisons"][s]["exact_accuracy"] for s in MAIN),
            "minus_chatbot_pts": rng(100 * df["hybrid_share"]["comparisons"][s]["diff"] for s in MAIN),
            "minus_chatbot_abs_pts": rng(-100 * df["hybrid_share"]["comparisons"][s]["diff"] for s in MAIN),
            "all_noninferior": all(df["hybrid_share"]["comparisons"][s]["noninferior"] for s in MAIN),
            "holm_lower_bound_pts": rng(100 * df["hybrid_share"]["comparisons"][s]["holm_lower_bound"] for s in MAIN),
            "over_chatbot_cost": rng(df["hybrid_share"]["comparisons"][s]["cost_list"]["hybrid_over_chatbot"] for s in MAIN),
            "confidence_minus_random_pts": rng(100 * df["routing"]["systems"][s]["exact_accuracy"]["diff"] for s in MAIN)},
        "free": {s: {"exact_accuracy": S[s]["exact_accuracy"], "definitions_exact_accuracy": df["systems"][s]["exact_accuracy"],
                     "usd_list_per_1000": S[s]["cost"]["usd_list_per_1000_answers"]} for s in FREE},
        "medical_pair": {s: {"exact_accuracy": S[s]["exact_accuracy"], "exact_accuracy_ci": S[s]["exact_accuracy_ci"],
                             "definitions_exact_accuracy": df["systems"][s]["exact_accuracy"]} for s in ("medgemma", "gemma")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
