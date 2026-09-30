"""Summary of the runs of Jev used as its developer documents (colon cancer staging and ten-year risk).

Reads the documented-use results (results/arm3_docuse/, results/wide_docuse/; --root points to the folder that holds
them) and the summaries in results/summaries/, and writes results/summaries/docuse_summary.json. Every number is
copied from those files; the checks below recompute a few of them from their parts and stop on any difference.
    python scripts/summaries/docuse_summary.py [--root <folder with results/arm3_docuse and results/wide_docuse>]
Run it after arm3_summary.py, cheap_family.py, chain.py, hybrid_all.py and wide_cheaper_structured.py, and once more
after hybrid_explore.py (which reads the first output): the blocks built from hybrid_explore.json are added then.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
VERSIONS = ["documented", "documented_notes"]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def rng(values) -> dict:
    v = list(values)
    return {"min": min(v), "max": max(v)}


def staging(res: Path) -> dict:
    out = {}
    for v in VERSIONS:
        d = res / "arm3_docuse" / v
        o = {}
        for chat in ["names", "definitions"]:
            a = load(d / f"analysis_{chat}.json")
            j = a["systems"]["jev"]
            assert abs(j["exact_accuracy"] - j["correct"] / j["n_items"]) < 1e-12, (v, chat)
            st = load(d / f"staging_{chat}.json")["systems"]["jev"]
            cf = load(d / f"cheap_family_{chat}.json")
            o[chat] = {
                "n_items": j["n_items"],
                "exact_accuracy": j["exact_accuracy"],
                "exact_accuracy_ci": j["exact_accuracy_ci"],
                "correct": j["correct"],
                "usable_share": j["usable_share"],
                "main_stage_accuracy": st["main_stage_accuracy"],
                "stage_iii_to_0_ii": st["stage_iii_to_0_ii"],
                "stage_iii_n": st["stage_iii_n"],
                "usd_list_per_1000": j["cost"]["usd_list_per_answer"] * 1000,
                "tokens_in_per_answer": j["cost"]["tokens_in_per_answer"],
                "main_chatbots": {
                    "exact_accuracy": rng(a["systems"][m]["exact_accuracy"] for m in MAIN),
                    "jev_minus_chatbot_pts": rng(100 * a["comparisons"][m]["diff"] for m in MAIN),
                    "outcomes": {m: a["comparisons"][m]["outcome"] for m in MAIN},
                    "n_noninferior": sum(a["comparisons"][m]["outcome"] == "non-inferior" for m in MAIN),
                    "usd_list_per_1000": rng(a["systems"][m]["cost"]["usd_list_per_answer"] * 1000 for m in MAIN),
                },
                "cheaper_systems": {
                    "outcomes": {m: cf["comparisons"][m]["outcome"] for m in CHEAP},
                    "n_noninferior": sum(cf["comparisons"][m]["outcome"] == "non-inferior" for m in CHEAP),
                },
            }
        parts = load(d / "parts.json")["main_pass"]
        o["parts_accuracy"] = {q: parts[q]["accuracy"] for q in parts}
        c = load(d / "analysis_confuser.json")
        mis = load(d / "misread.json")["types_1_4"]
        o["misleading_feature"] = {
            "n": c["systems"]["jev"]["n_items"] if "n_items" in c["systems"]["jev"] else c["n_items"],
            "exact_accuracy": c["systems"]["jev"]["exact_accuracy"],
            "by_type": {mis[t]["feature"]: {"n": mis[t]["systems"]["jev"]["n"], "correct": mis[t]["systems"]["jev"]["correct"]}
                        for t in mis},
            "main_chatbots_exact_accuracy": rng(c["systems"][m]["exact_accuracy"] for m in MAIN if m in c["systems"]),
        }
        out[v] = o
    rw = res / "arm3_docuse_reworded"
    if (rw / "summary.json").exists():
        sm, dt = load(rw / "summary.json"), load(rw / "details.json")
        out["reworded"] = {"n_reports": sm["per_variant"]["documented"]["n_reports"],
                           "exact_correct_min": min(sm["per_variant"][v]["exact_correct"] for v in VERSIONS),
                           "same_answers_min": min(dt["answers"]["same_choice_as_original"].values()),
                           "n_answers": dt["answers"]["n"],
                           "parser_tnm_right_reworded": dt["parser"]["tnm_right"]["reworded"],
                           "parser_tnm_right_original": dt["parser"]["tnm_right"]["original"]}
    t = load(res / "arm3_docuse" / "timing_sample.json")["systems"]["jev"]
    out["timing"] = {"n": t["n"], "usable": t["usable"], "median_s": t["median_s"], "p90_s": t["p90_s"]}
    return out


def risk(res: Path) -> dict:
    a = load(res / "wide_docuse" / "analysis.json")
    p = a["prediction"]
    s = p["systems"]
    j = s["jev_documented"]
    pairs = p["focal_vs_llms"]["pairs"]
    diffs = {m: pairs[f"jev_documented - {m}"] for m in MAIN}
    worse = [m for m in MAIN if diffs[m]["ci_simultaneous"][0] > diffs[m]["noninferiority_margin_fixed"]]
    noninf = [m for m in MAIN if diffs[m]["noninferior_fixed_margin"]]
    dvo = a["documented_vs_original"]
    orig = dvo["systems"]["jev"]
    d_ll = dvo["focal_vs_llms"]["pairs"]["jev_documented - jev"]
    d_re = dvo["focal_vs_llms_recalibrated"]["pairs"]["jev_documented - jev"]
    assert abs((j["log_loss"] - orig["log_loss"]) - d_ll["diff"]) < 1e-9
    agesex = p["focal_vs_age_sex"]["pairs"]["jev_documented - reference_age_sex"]
    ec = a["eval_cheap_family"]
    assert abs(ec["margin"] - a["margins"]["eval_fixed"]) < 1e-15
    return {
        "records": a["records"]["wide"]["scored"],
        "deaths": p["deaths"],
        "margin": a["margins"]["wide_fixed"],
        "jev_documented": {k: j[k] for k in ["log_loss", "log_loss_ci", "log_loss_recalibrated", "brier", "auroc",
                                             "calibration_slope", "mean_p"]},
        "jev_original": {k: orig[k] for k in ["log_loss", "log_loss_ci", "log_loss_recalibrated", "auroc", "mean_p"]},
        "documented_minus_original": {"diff": d_ll["diff"], "ci_simultaneous": d_ll["ci_simultaneous"],
                                      "reduction": -d_ll["diff"],
                                      "reduction_ci_simultaneous": [-d_ll["ci_simultaneous"][1], -d_ll["ci_simultaneous"][0]]},
        "documented_minus_original_recalibrated": {"diff": d_re["diff"], "ci_simultaneous": d_re["ci_simultaneous"]},
        "main_chatbots": {
            "log_loss": rng(s[m]["log_loss"] for m in MAIN),
            "auroc": rng(s[m]["auroc"] for m in MAIN),
            "mean_p": rng(s[m]["mean_p"] for m in MAIN),
            "diff": {m: diffs[m]["diff"] for m in MAIN},
            "diff_worse": rng(diffs[m]["diff"] for m in worse) if worse else None,
            "n_worse": len(worse), "worse": worse,
            "n_noninferior": len(noninf), "noninferior": noninf,
        },
        "vs_age_sex": {"diff": agesex["diff"], "ci_simultaneous": agesex["ci_simultaneous"],
                       "reduction": -agesex["diff"],
                       "reduction_ci_simultaneous": [-agesex["ci_simultaneous"][1], -agesex["ci_simultaneous"][0]],
                       "noninferior": agesex["noninferior_fixed_margin"]},
        "cheaper_evaluation": {"n_records": ec["n_records"], "deaths": ec["deaths"], "margin": ec["margin"],
                               "comparisons": {m: {"diff": ec["comparisons"][m]["diff"],
                                                   "ci_simultaneous": ec["comparisons"][m]["ci_simultaneous"],
                                                   "outcome": ec["comparisons"][m]["outcome"]} for m in CHEAP}},
        "latency_median_s": a["validity"]["latency_median_s"],
        "usd_list_per_1000": a["validity"]["usd_list_total"] / a["validity"]["n_calls"] * 1000,
        "usable_share": a["validity"]["usable_share"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    res = Path(a.root) / "results"
    out = {"note": "Jev used as its developer documents; numbers copied from results/arm3_docuse and results/wide_docuse; "
                   "cost at standard list price",
           "staging": staging(res), "risk": risk(res)}
    orig = load(ROOT / "results" / "summaries" / "arm3_summary.json")["jev"]
    out["staging"]["original"] = {"names_exact_accuracy": orig["exact_accuracy"],
                                  "definitions_exact_accuracy": orig["definitions_exact_accuracy"]}
    # Jev as run originally against the five cheaper systems (results/summaries/cheap_family.json), counted per system
    cf = load(ROOT / "results" / "summaries" / "cheap_family.json")["sets"]
    cat = ["next_step", "board_exam", "staging_names", "staging_definitions", "aki_names", "aki_definitions"]
    out["cheaper_original"] = {m: {"n_noninferior": sum(cf[s]["comparisons"][m]["outcome"] == "non-inferior" for s in cat),
                                   "n_worse": sum(cf[s]["comparisons"][m]["outcome"] == "worse" for s in cat),
                                   "n_sets": len(cat)} for m in CHEAP}
    out["cheaper_original"]["open_weights_noninferior"] = sum(out["cheaper_original"][m]["n_noninferior"]
                                                              for m in ["medgemma", "gemma"])
    # Three-step chain Jev -> GPT-5.6 Luna -> GPT-5.6 Sol (results/summaries/chain.json), where names came with
    # definitions or criteria, and with names alone
    ch = load(ROOT / "results" / "summaries" / "chain.json")["sets"]
    with_def = ["next_step", "board_exam", "staging_definitions", "aki_definitions"]
    names = ["staging_names", "aki_names"]
    c = {s: ch[s]["chains"]["jev>gpt_free>gpt"] for s in with_def + names}
    # Jev's most confident half plus each main chatbot (results/summaries/hybrid_all.json), sets with definitions/criteria
    ha = load(ROOT / "results" / "summaries" / "hybrid_all.json")["sets"]
    out["hybrid_main_with_definitions"] = {
        "cost_share": rng(ha[s]["partners"][m]["cost_share"] for s in with_def for m in MAIN),
        "diff_pts": rng(ha[s]["partners"][m]["diff_pts"] for s in with_def for m in MAIN),
        "diff_pts_abs_max": max(abs(ha[s]["partners"][m]["diff_pts"]) for s in with_def for m in MAIN),
    }
    lu = [ha[s]["partners"]["gpt_free"] for s in with_def]
    out["hybrid_luna_with_definitions"] = {"cost_share": rng(x["cost_share"] for x in lu),
                                           "saving": rng(1 - x["cost_share"] for x in lu),
                                           "diff_pts": rng(x["diff_pts"] for x in lu)}
    out["chain_luna_sol"] = {
        "with_definitions_minus_sol_pts": rng(-c[s]["chain_minus_expensive_pts"]["est"] for s in with_def),
        "with_definitions_cost_share": rng(c[s]["usd_per_1000"]["chain"] / c[s]["usd_per_1000"]["expensive_alone"]
                                           for s in with_def),
        "names_minus_sol_pts": rng(-c[s]["chain_minus_expensive_pts"]["est"] for s in names),
    }
    # The five cheaper systems on the full risk set (results/summaries/wide_cheaper_structured.json), Jev both ways
    wf = ROOT / "results" / "summaries" / "wide_cheaper_structured.json"
    if wf.exists():
        w = load(wf)
        assert w["records"] == out["risk"]["records"] and abs(w["jev_documented_log_loss"]
                                                              - out["risk"]["jev_documented"]["log_loss"]) < 1e-12
        cf_full = {"records": w["records"], "margin": w["margin"],
                   "log_loss": rng(w["systems"][m]["log_loss"] for m in CHEAP),
                   "auroc": rng(w["systems"][m]["auroc"] for m in CHEAP),
                   "usd_list_all_answers": sum(w["usd_list_all_answers"].values())}
        for blk in ("structured", "chatbot_prompt"):
            comp = w[blk]["comparisons"]
            cf_full[blk] = {o: [m for m in CHEAP if comp[m]["outcome"] == o]
                            for o in ("non-inferior", "worse", "inconclusive")}
            cf_full[blk]["n"] = {o: len(v) for o, v in list(cf_full[blk].items())}
            cf_full[blk]["diff"] = {m: comp[m]["diff"] for m in CHEAP}
        out["risk"]["cheaper_full"] = cf_full
    # Jev's joint confidence on the altered reports against the main 1,000 (results/summaries/hybrid_explore.json):
    # full confidence = joint confidence 1.00, the median of the main reports in both versions
    he = ROOT / "results" / "summaries" / "hybrid_explore.json"
    if he.exists():
        sn = load(he)["structured_safety_net"]["primary"]
        alt = ["misleading_feature", "nonregional_sites"]
        blk = {v: sn[f"{v}|median"] for v in VERSIONS}
        assert all(abs(blk[v]["cut"] - 1.0) < 1e-12 for v in VERSIONS)
        wrong = sum(blk[v]["sets"][s]["jev_wrong"] for v in VERSIONS for s in alt)
        below = sum(blk[v]["sets"][s]["wrong_sent"] for v in VERSIONS for s in alt)
        out["staging"]["full_confidence"] = {
            "altered_wrong": wrong, "altered_wrong_below_full": below,
            "altered_n_reports": sum(blk["documented"]["sets"][s]["n"] for s in alt),
            "main_share_full": rng(1 - blk[v]["sets"]["main"]["share_sent"] for v in VERSIONS)}
    nr = res / "arm3_nonregional_sites" / "summary.json"
    if nr.exists():
        nj = load(nr)
        ns = nj["systems"]
        out["staging"]["nonregional"] = {
            "n": ns["jev/documented"]["n_reports"], "n_sites": len(nj["sites"]),
            "jev_documented": ns["jev/documented"]["correct"], "jev_notes": ns["jev/documented_notes"]["correct"],
            "jev_other_parts_correct_min": min(ns[f"jev/{v}"]["four_questions_correct"][q] for v in VERSIONS
                                               for q in ["t_category", "regional_nodes", "tumor_deposits"]),
            "chatbots_correct": rng(ns[f"{m}/names"]["correct"] for m in MAIN)}
    # Held-out reports (results/arm3_heldout/summary.json and truth_check.json); every value in summary.json is a
    # [value, source] pair
    hp = ROOT / "results" / "arm3_heldout" / "summary.json"
    if hp.exists():
        hs = load(hp)
        v = lambda x: x[0]
        ja, cb = hs["jev_arms"], hs["chatbots"]
        chats = ["names", "definitions"]
        tc = load(ROOT / "results" / "arm3_heldout" / "truth_check.json")
        out["staging"]["heldout"] = {
            "n": hs["n_reports"], "n_kept": tc["n_kept"], "adjudicated": tc["adjudicated"],
            "reader_mismatches": sum(tc["mismatches_by_reader_and_fact"].values()),
            "documented_exact_min": min(v(ja[ver][c]["exact_accuracy"]) for ver in VERSIONS for c in chats),
            "documented_n_noninferior_min": min(sum(v(ja[ver][c]["vs_chatbots"][m]["outcome"]) == "non-inferior" for m in MAIN)
                                                for ver in VERSIONS for c in chats),
            "documented_usd_list_per_1000": v(ja["documented"]["names"]["usd_list_per_1000"]),
            "original_names_exact": v(ja["original"]["names"]["exact_accuracy"]),
            "original_definitions_exact": v(ja["original"]["definitions"]["exact_accuracy"]),
            "original_n_worse_min": min(sum(v(ja["original"][c]["vs_chatbots"][m]["outcome"]) == "worse" for m in MAIN) for c in chats),
            "chatbots_names_exact": rng(v(cb["names"][m]["exact_accuracy"]) for m in MAIN),
            "chatbots_definitions_exact": rng(v(cb["definitions"][m]["exact_accuracy"]) for m in MAIN),
        }
        assert out["staging"]["heldout"]["documented_exact_min"] == 1.0, "manuscript: 'staged every held-out report correctly'"
    # Agreement cascade: Jev and GPT-5.6 Luna answer every item, their shared answer stands, GPT-5.6 Sol answers the
    # items on which they differ (results/summaries/hybrid_explore.json)
    if he.exists():
        ac = load(he)["agreement_cascade"]
        cc = {s_: ac[s_]["cascades"]["agree(jev,gpt_free)>gpt"] for s_ in with_def + names}
        out["agreement_luna_sol"] = {
            "all_minus_sol_pts": rng(cc[s_]["minus_top_alone_pts"]["est"] for s_ in with_def + names),
            "all_sol_minus_pts": rng(-cc[s_]["minus_top_alone_pts"]["est"] + 0.0 for s_ in with_def + names),
            "with_definitions_cost_share": rng(cc[s_]["cost_share_vs_top"] for s_ in with_def),
            "names_cost_share": rng(cc[s_]["cost_share_vs_top"] for s_ in names),
            "all_cost_share": rng(cc[s_]["cost_share_vs_top"] for s_ in with_def + names),
            "share_escalated": rng(cc[s_]["share_escalated"] for s_ in with_def + names),
            "names_minus_sol_pts": rng(cc[s_]["minus_top_alone_pts"]["est"] for s_ in names),
        }
    # statements in the manuscript text; stop if the numbers behind them change
    assert out["risk"]["main_chatbots"]["n_worse"] == 4, "manuscript: 'worse than four of the five chatbots'"
    if "cheaper_full" in out["risk"]:
        c = out["risk"]["cheaper_full"]["structured"]
        assert c["non-inferior"] == ["medgemma", "gemma"] and c["worse"] == ["gpt_free", "gemini_free"], \
            "manuscript: 'no worse than the two open-weight ones and worse than GPT-5.6 Luna and Gemini 3.5 Flash-Lite'"
    if "nonregional" in out["staging"]:
        assert out["staging"]["nonregional"]["n_sites"] == 5, "manuscript: 'at five sites the notes do not name'"
        assert out["staging"]["nonregional"]["chatbots_correct"]["min"] == out["staging"]["nonregional"]["n"], \
            "manuscript: 'every chatbot on all 20'"
    if "full_confidence" in out["staging"]:
        fc = out["staging"]["full_confidence"]
        assert fc["altered_wrong"] == fc["altered_wrong_below_full"], "manuscript: 'none of Jev's wrong answers carried full confidence'"
    dst = ROOT / "results" / "summaries" / "docuse_summary.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {dst}")
