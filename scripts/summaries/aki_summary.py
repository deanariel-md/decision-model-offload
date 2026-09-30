"""Summary numbers for the eICU AKI stage set (KDIGO creatinine criteria, eICU demo), from its stored analysis outputs.

Ranges over the five main chatbots for both versions (level names only; with the KDIGO table shown), Jev with its
intervals, the tests, the hybrid and random-half control, Jev's confident half as an error filter, and two descriptive
measures computed here from stored per-stage results: balanced accuracy (mean recall over the four stages) and recall
of severe AKI (stage 2 or 3 answered stage 2 or 3, from the confusion counts). Also the stays whose key depends on the
baseline creatinine, and the key with creatinine values before unit admission assessed too (on_arrival.json). Reads
analysis.json, analysis_definitions.json, supplement_names.json, supplement_definitions.json and on_arrival.json from
--dir and side.json from --side; no records are read.

  python scripts/summaries/aki_summary.py [--dir results/eicu_aki] [--side results/eicu_aki_side]
    -> results/summaries/aki_summary.json
"""
import argparse, json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
ALL = ["jev"] + MAIN + ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
STAGES = ["no AKI", "stage 1", "stage 2", "stage 3"]


def rng(xs):
    xs = list(xs)
    return {"min": min(xs), "max": max(xs)}


def per_system(an, sup):
    out = {}
    for s in ALL:
        S = an["systems"][s]
        rec = S["accuracy_by_stratum"]
        cm = sup["confusion"][s]["counts"]
        n_sev = sum(sum(cm[t].values()) for t in ("stage 2", "stage 3"))
        sev = sum(cm[t][a] for t in ("stage 2", "stage 3") for a in ("stage 2", "stage 3"))
        for t in STAGES:  # confusion rows agree with the stored recall
            assert abs(cm[t][t] / sum(cm[t].values()) - rec[t]) < 1e-9, (s, t)
        out[s] = {"exact_accuracy": S["exact_accuracy"], "exact_accuracy_ci": S["exact_accuracy_ci"],
                  "balanced_accuracy": sum(rec[t] for t in STAGES) / 4, "recall": rec,
                  "severe_recall": sev / n_sev, "severe_n": n_sev, "unusable": S["unusable"], "qwk": S.get("qwk"),
                  "usd_list_per_1000": S["cost"]["usd_list_per_1000_answers"]}
    return out


def version(an, sup):
    ps = per_system(an, sup)
    C, H, R = an["comparisons"], an["hybrid_share"]["comparisons"], an["routing"]["systems"]
    ch = sup["confident_half"]
    return {
        "n_items": an["n_items"], "strata": an["strata"], "systems": ps,
        "jev": ps["jev"] | {"confident_half": ch, "confusion": sup["confusion"]["jev"]["counts"],
                            "stage_1_to_2": sup["confusion"]["jev"]["counts"]["stage 1"]["stage 2"],
                            "no_aki_to_stage_1": sup["confusion"]["jev"]["counts"]["no AKI"]["stage 1"],
                            "over_staged": sum(sup["confusion"]["jev"]["counts"][t][a] for i, t in enumerate(STAGES)
                                               for a in STAGES[i + 1:]),
                            "errors_let_through": round(ch["let_through"] * ch["wrong"]), "n_errors": ch["wrong"],
                            "share_by_stage_in_confident_half": H["gpt"]["share_jev_by_stratum"]},
        "main_chatbots": {
            "exact_accuracy": rng(ps[s]["exact_accuracy"] for s in MAIN),
            "balanced_accuracy": rng(ps[s]["balanced_accuracy"] for s in MAIN),
            "severe_recall": rng(ps[s]["severe_recall"] for s in MAIN),
            "jev_minus_chatbot_abs_pts": rng(-100 * C[s]["diff"] for s in MAIN),
            "outcomes": {s: C[s]["outcome"] for s in MAIN},
            "n_worse": sum(C[s]["outcome"] == "worse" for s in MAIN),
            "hybrid_exact_accuracy": rng(H[s]["exact_accuracy"] for s in MAIN),
            "hybrid_minus_chatbot_pts": rng(100 * H[s]["diff"] for s in MAIN),
            "hybrid_minus_chatbot_abs_pts": rng(-100 * H[s]["diff"] for s in MAIN),
            "hybrid_outcomes": {s: H[s]["outcome"] for s in MAIN},
            "hybrid_all_noninferior": all(H[s]["noninferior"] for s in MAIN),
            "hybrid_all_worse": all(H[s]["outcome"] == "worse" for s in MAIN),
            "hybrid_holm_lower_bound_pts": rng(100 * H[s]["holm_lower_bound"] for s in MAIN),
            "hybrid_over_chatbot_cost": rng(H[s]["cost_list"]["hybrid_over_chatbot"] for s in MAIN),
            "confidence_minus_random_pts": rng(100 * R[s]["exact_accuracy"]["diff"] for s in MAIN),
            "usd_list_per_1000": rng(ps[s]["usd_list_per_1000"] for s in MAIN)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/eicu_aki")
    ap.add_argument("--side", default="results/eicu_aki_side")
    ap.add_argument("--out", default="results/summaries/aki_summary.json")
    a = ap.parse_args()
    d = Path(a.dir)
    J = lambda p: json.loads(Path(p).read_text())
    names, defs = J(d / "analysis.json"), J(d / "analysis_definitions.json")
    assert names["variant"] == "names" and defs["variant"] == "definitions"
    res = {"note": "descriptive summary of the AKI stage set; names version primary; cost at standard list price",
           "names": version(names, J(d / "supplement_names.json")),
           "definitions": version(defs, J(d / "supplement_definitions.json"))}
    res["definitions_gain_pts"] = 100 * (res["definitions"]["jev"]["exact_accuracy"] - res["names"]["jev"]["exact_accuracy"])
    bs = J(d / "supplement_names.json")["baseline_sensitivity"]
    res["baseline_sensitive"] = {"n": bs["n_sensitive"], "jev_accuracy_without": bs["systems"]["jev"]["accuracy_without"],
                                 "main_accuracy_without": rng(bs["systems"][s]["accuracy_without"] for s in MAIN)}
    res["baseline_sensitive"]["on_sensitive_key"] = {s: bs["systems"][s]["on_sensitive_key"] for s in ["jev"] + MAIN}
    oa = J(d / "on_arrival.json")  # values before unit admission assessed too, same rule and windows
    res["on_arrival"] = {"n_records": oa["n_records"], "n_changed": oa["n_changed"],
                         "n_changed_baseline_sensitive": oa["changed_baseline_sensitive"]}
    for v, an in (("names", names), ("definitions", defs)):
        x = oa["versions"][v]
        for s in ["jev"] + MAIN:  # the key's accuracies must be the analysis file's
            assert abs(x["accuracy"][s]["key"] - an["systems"][s]["exact_accuracy"]) < 1e-12, (v, s)
        sh = [100 * x["jev_minus"][s]["shift"] for s in MAIN]
        res["on_arrival"][v] = {"shift_pts": {"min": min(sh), "max": max(sh)},
                                "jev_minus_chatbot_pts": rng(100 * x["jev_minus"][s]["on_arrival"] for s in MAIN)}
    side = J(Path(a.side) / "side.json")["versions"]["names"]
    res["side"] = {k: {"n": v["n"], "jev": v["systems"]["jev"]["exact_accuracy"],
                       "main": rng(v["systems"][s]["exact_accuracy"] for s in MAIN)} for k, v in side.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for v in ("names", "definitions"):
        x = res[v]
        print(v, "jev", round(x["jev"]["exact_accuracy"], 3), "bal", round(x["jev"]["balanced_accuracy"], 3), "severe",
              round(x["jev"]["severe_recall"], 3), "| main", {k: x["main_chatbots"][k] for k in
              ("exact_accuracy", "hybrid_minus_chatbot_pts", "hybrid_outcomes", "severe_recall")})
        print("   jev errors through", x["jev"]["errors_let_through"], "of", x["jev"]["n_errors"])
    print("gain", res["definitions_gain_pts"], res["baseline_sensitive"], res["side"])


if __name__ == "__main__":
    main()
