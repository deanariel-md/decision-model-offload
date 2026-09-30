"""Arm 2 next-step choice: descriptive summary from the stored outputs.

Reads the arm 2 outputs (analysis.json, calls.parquet) and writes results/summaries/arm2_summary.json with:
- ranges across the five main chatbots (balanced accuracy, differences from Jev, list cost per 1,000 answers,
  hybrid cost ratio and saving);
- Jev's answers split at a reported top probability of 1.00 (share, accuracy, balanced accuracy, and the hybrid that
  hands Jev exactly those items), a threshold rule that needs no batch, and the same at lower cutoffs;
- Jev minus chatbot balanced accuracy with intervals from a bootstrap that resamples whole recommendations
  (14 clusters; sensitivity to the item-level bootstrap of the main analysis), and the tested hybrid against each
  chatbot with the same bootstrap;
- the chatbots' output tokens on the items Jev handles and on those it passes on.
Truth: items whose id contains "_lv_" are low-value (correct answer: hold off, canonical "B"); "_ac_" are indicated
(correct: the intervention, canonical "A"). Checked against analysis.json (accuracy per system must match).

  python scripts/summaries/arm2_summary.py [--arm2 results/arm2] [--out results/summaries/arm2_summary.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]


def ba(correct: pd.Series, low_value: pd.Series) -> float:
    return float((correct[low_value].mean() + correct[~low_value].mean()) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm2", default="results/arm2")
    ap.add_argument("--out", default="results/summaries/arm2_summary.json")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260922)
    a = ap.parse_args()
    arm2 = Path(a.arm2)
    an = json.loads((arm2 / "analysis.json").read_text())
    c = pd.read_parquet(arm2 / "calls.parquet")
    c = c[(c["variant"] == "main") & (c["repeat"] == 0)].copy()
    c["low_value"] = c["item_id"].str.contains("_lv_")
    c["truth"] = np.where(c["low_value"], "B", "A")
    c["correct"] = c["answer"] == c["truth"]
    for s in ["jev"] + MAIN:  # check the truth rule against the analysis
        got = c.loc[c.system == s, "correct"].mean()
        assert abs(got - an["systems"][s]["accuracy"]) < 1e-9, (s, got)
    W = c.pivot_table(index="item_id", columns="system", values="correct", aggfunc="first").astype(bool)
    L = pd.Series(W.index.str.contains("_lv_"), index=W.index)
    rec = pd.Series(W.index.str.slice(0, 6), index=W.index)
    out = {"note": "descriptive; outside the tests of the main analysis", "n_items": int(len(W))}

    S = an["systems"]; H = an["hybrid_cost"]; R = an["routing"]["systems"]; C = an["comparisons"]
    rng_ = lambda xs: {"min": float(min(xs)), "max": float(max(xs))}
    out["main_chatbots"] = {
        "balanced_accuracy": rng_([S[s]["balanced_accuracy"] for s in MAIN]),
        "jev_minus_chatbot": rng_([C[s]["diff"] for s in MAIN]),
        "jev_minus_chatbot_abs": rng_([-C[s]["diff"] for s in MAIN]),
        "usd_list_per_1000": rng_([S[s]["cost"]["usd_list_per_1000_answers"] for s in MAIN]),
        "hybrid_balanced_accuracy": rng_([H[s]["balanced_accuracy"] for s in MAIN]),
        "hybrid_over_chatbot_cost": rng_([H[s]["hybrid_over_chatbot"] for s in MAIN]),
        "hybrid_saving_share": rng_([H[s]["saving_share"] for s in MAIN]),
        "hybrid_saving_usd_per_1000": rng_([H[s]["saving_usd_per_1000_items"] for s in MAIN]),
        "confidence_minus_random_ba": rng_([R[s]["balanced_accuracy"]["diff"] for s in MAIN]),
        "outcomes": {s: C[s]["outcome"] for s in MAIN},
        "hybrid_outcomes": {s: an["hybrid_share"]["comparisons"][s]["outcome"] for s in MAIN},
    }
    mc = out["main_chatbots"]
    for k in ["jev_minus_chatbot_abs", "confidence_minus_random_ba"]:
        mc[k + "_pts"] = {m: 100 * v for m, v in mc[k].items()}
    hd = [an["hybrid_share"]["comparisons"][s]["diff"] for s in MAIN]
    mc["hybrid_minus_chatbot_pts"] = rng_([100 * x for x in hd])
    jr = S["jev"]["rates"]
    out["jev_errors"] = {"n_errors": int(round((1 - S["jev"]["accuracy"]) * S["jev"]["n_items"])),
                         "overuse_n": int(round(jr["overuse"] * an["strata"]["low_value"])),
                         "false_restraint_n": int(round(jr["false_restraint"] * an["strata"]["appropriate_care"])),
                         "overuse": jr["overuse"], "false_restraint": jr["false_restraint"],
                         "n_low_value": an["strata"]["low_value"], "n_indicated": an["strata"]["appropriate_care"]}
    out["jev_cost_ratio"] = {s: float(S[s]["cost"]["usd_list_per_1000_answers"] / S["jev"]["cost"]["usd_list_per_1000_answers"]) for s in MAIN}
    out["jev_cost_ratio_range"] = rng_(list(out["jev_cost_ratio"].values()))

    j = c[c.system == "jev"].set_index("item_id")
    top1 = j["top_prob"] >= 0.9999
    jl = j["low_value"]
    out["jev_at_probability_1"] = {
        "n": int(top1.sum()), "share": float(top1.mean()),
        "accuracy": float(j.loc[top1, "correct"].mean()),
        "balanced_accuracy": ba(j.loc[top1, "correct"], jl[top1]),
        "n_below": int((~top1).sum()), "accuracy_below": float(j.loc[~top1, "correct"].mean()),
        "hybrid": {s: {"balanced_accuracy": ba(pd.Series(np.where(W.index.isin(j.index[top1]), W["jev"], W[s]), index=W.index), L),
                       "chatbot_alone": ba(W[s], L)} for s in MAIN},
    }
    cuts = {}
    for cut in [1.0, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70]:
        keep = pd.Series(j["top_prob"].reindex(W.index).values >= cut - 1e-9, index=W.index)
        cuts[f"{cut:.2f}"] = {
            "n_jev": int(keep.sum()), "share_jev": float(keep.mean()),
            "jev_accuracy_kept": float(W.loc[keep, "jev"].mean()),
            "jev_accuracy_passed_on": float(W.loc[~keep, "jev"].mean()) if (~keep).any() else None,
            "hybrid_balanced_accuracy": {s: ba(pd.Series(np.where(keep, W["jev"], W[s]), index=W.index), L) for s in MAIN},
            "hybrid_minus_chatbot_pts": {s: 100 * (ba(pd.Series(np.where(keep, W["jev"], W[s]), index=W.index), L) - ba(W[s], L)) for s in MAIN}}
    out["jev_probability_cutoffs"] = {"note": "descriptive: Jev keeps its answer when its top probability is at or above the cutoff; the tested rule is the 50% share", "cutoffs": cuts}
    jev_items = set(an["hybrid_share"]["jev_items"])
    out["hybrid_half_within_ties"] = {"n_half": len(jev_items), "all_at_probability_1": bool(all(top1[i] for i in jev_items)),
                                      "correct_in_half": int(j.loc[list(jev_items), "correct"].sum())}

    rng = np.random.default_rng(a.seed); recs = sorted(rec.unique()); idx = {r: W.index[rec == r] for r in recs}
    clus = {}
    for s in MAIN:
        d = []
        for _ in range(a.n_boot):
            pick = rng.choice(recs, len(recs), replace=True)
            ii = np.concatenate([idx[r] for r in pick])
            d.append(ba(W.loc[ii, "jev"].reset_index(drop=True), L.loc[ii].reset_index(drop=True))
                     - ba(W.loc[ii, s].reset_index(drop=True), L.loc[ii].reset_index(drop=True)))
        clus[s] = {"diff": ba(W["jev"], L) - ba(W[s], L), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    out["cluster_bootstrap_by_recommendation"] = {"n_clusters": len(recs), "n_boot": a.n_boot, "systems": clus}

    # the tested hybrid (Jev's most confident 50%, the chatbot the rest) against each chatbot, whole recommendations
    # resampled; one-sided lower bounds at 0.025 (unadjusted) and 0.005 (Bonferroni over five, stricter than Holm)
    jin = pd.Series(W.index.isin(list(jev_items)), index=W.index)
    rng = np.random.default_rng(a.seed + 1)
    hcl = {}
    for s in MAIN:
        H_ = pd.Series(np.where(jin, W["jev"], W[s]), index=W.index)
        d = []
        for _ in range(a.n_boot):
            pick = rng.choice(recs, len(recs), replace=True)
            ii = np.concatenate([idx[r] for r in pick])
            LL = L.loc[ii].reset_index(drop=True)
            d.append(ba(H_.loc[ii].reset_index(drop=True), LL) - ba(W.loc[ii, s].reset_index(drop=True), LL))
        d = np.array(d)
        hcl[s] = {"diff": ba(H_, L) - ba(W[s], L), "lower_0025": float(np.percentile(d, 2.5)),
                  "lower_0005": float(np.percentile(d, 0.5)), "upper_0975": float(np.percentile(d, 97.5))}
    out["hybrid_cluster_bootstrap_by_recommendation"] = {
        "margin": -0.05, "n_clusters": len(recs), "n_boot": a.n_boot, "systems": hcl,
        "all_above_margin_bonferroni": bool(all(v["lower_0005"] > -0.05 for v in hcl.values())),
        "all_above_margin_unadjusted": bool(all(v["lower_0025"] > -0.05 for v in hcl.values()))}

    tok = {}
    for s in MAIN:
        x = c[c.system == s].set_index("item_id")["tokens_out"].fillna(0)
        inj = x.index.isin(jev_items)
        tok[s] = {"jev_items": float(x[inj].mean()), "passed_on": float(x[~inj].mean())}
    out["chatbot_output_tokens_by_jev_half"] = tok

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["main_chatbots", "jev_at_probability_1"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()
