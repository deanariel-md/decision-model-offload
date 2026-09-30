"""Arm 2 second item set (MedQA and Medbullets in MedHELM's versions): summary numbers from the stored outputs.

  python scripts/summaries/arm2_ext_summary.py [--dir results/arm2_ext] -> results/summaries/arm2_ext_summary.json
All questions: analysis.json (MedHELM option counts, 1,571 questions); subgroup: analysis_medhelm_decision.json
(next-step and management questions). Jev's confident half is the tested hybrid rule (hybrid_share.jev_items); its
errors let through are counted from calls.parquet with the key in data/arm2_ext/items.csv (built by
python scripts/build_arm2_ext_items.py --download).
"""
import argparse, json
from pathlib import Path
import pandas as pd

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]


def rng(xs):
    xs = list(xs)
    return {"min": min(xs), "max": max(xs)}


def block(a):
    S, C, H, R = a["systems"], a["comparisons"], a["hybrid_share"]["comparisons"], a["routing"]["systems"]
    out = {
        "n_items": a["n_items"],
        "jev": {"accuracy": S["jev"]["accuracy"], "accuracy_ci": S["jev"]["accuracy_ci"],
                "certain_share": S["jev"]["certain"]["share"], "certain_n": S["jev"]["certain"]["n"],
                "certain_accuracy": S["jev"]["certain"]["accuracy"], "certain_accuracy_ci": S["jev"]["certain"]["accuracy_ci"],
                "accuracy_below_certain": S["jev"]["certain"]["accuracy_below"],
                "usd_list_per_1000": S["jev"]["cost"]["usd_list_per_1000_answers"]},
        "main_chatbots": {
            "accuracy": rng(S[s]["accuracy"] for s in MAIN),
            "jev_minus_chatbot_abs_pts": rng(-100 * C[s]["diff"] for s in MAIN),
            "outcomes": {s: C[s]["outcome"] for s in MAIN},
            "n_worse": sum(C[s]["outcome"] == "worse" for s in MAIN),
            "usd_list_per_1000": rng(S[s]["cost"]["usd_list_per_1000_answers"] for s in MAIN),
            "hybrid_accuracy": rng(H[s]["accuracy"] for s in MAIN),
            "hybrid_minus_chatbot_pts": rng(100 * H[s]["diff"] for s in MAIN),
            "hybrid_all_noninferior": all(H[s]["noninferior"] for s in MAIN),
            "hybrid_over_chatbot_cost": rng(H[s]["cost_list"]["hybrid_over_chatbot"] for s in MAIN),
            "hybrid_saving_share": rng(H[s]["cost_list"]["saving_share"] for s in MAIN),
            "confidence_minus_random_pts": rng(100 * R[s]["accuracy"]["diff"] for s in MAIN)},
        "other": {s: {"accuracy": S[s]["accuracy"], "usd_list_per_1000": S[s]["cost"]["usd_list_per_1000_answers"]}
                  for s in ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arm2_ext")
    ap.add_argument("--out", default="results/summaries/arm2_ext_summary.json")
    ap.add_argument("--items", default="data/arm2_ext/items.csv")
    a = ap.parse_args()
    d = Path(a.dir)
    main_ = json.loads((d / "analysis.json").read_text())
    sub = json.loads((d / "analysis_medhelm_decision.json").read_text())
    res = {"note": "descriptive summary of the second next-step item set; cost at standard list price",
           "all": block(main_), "next_step_subgroup": block(sub)}
    # Jev's confident half as an error filter: truth from data/arm2_ext/items.csv, checked against analysis.json
    items = pd.read_csv(a.items, dtype=str)
    truth = dict(zip(items["item_id"], items["truth"]))
    dec = set(items.loc[items["decision_question"].str.lower() == "true", "item_id"])
    c = pd.read_parquet(d / "calls.parquet")
    j = c[(c["system"] == "jev") & (c["variant"] == "medhelm") & (c["repeat"] == 0)].set_index("item_id")
    ok = j["valid"] & (j["answer"] == j.index.map(truth))
    assert abs(ok.mean() - main_["systems"]["jev"]["accuracy"]) < 1e-9, ok.mean()
    for key, an, ids in (("all", main_, j.index), ("next_step_subgroup", sub, [i for i in j.index if i in dec])):
        o = ok.loc[list(ids)]
        half = o.index.isin(an["hybrid_share"]["jev_items"])
        assert abs(o.mean() - an["systems"]["jev"]["accuracy"]) < 1e-9, (key, o.mean())
        res[key]["jev"]["confident_half"] = {"n": int(half.sum()), "accuracy": float(o[half].mean()),
                                             "errors_let_through": int((~o[half]).sum()), "n_errors": int((~o).sum()),
                                             "correct_kept": int(o[half].sum()), "n_correct": int(o.sum())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3500])


if __name__ == "__main__":
    main()
