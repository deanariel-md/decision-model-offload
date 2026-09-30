"""Arm 1 risk estimation (NHANES): ranges and named cells of the edit family (spline reference) and list cost.

Reads results/eval/analysis.json (confirmatory block, spline reference) and results/eval/cost_speed.json; writes
results/summaries/arm1_edit_summary.json: each system's clinical response slope, the chatbots' slope range, the cells
whose simultaneous interval excludes zero with their implied and supported odds ratios, the slope with each clinical
edit left out in turn, and list cost per 1,000 estimates (Jev, chatbot range, ratios).
  python scripts/summaries/arm1_edit_summary.py [--eval results/eval]
"""
import argparse, json, math
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/eval")
    ap.add_argument("--out", default="results/summaries/arm1_edit_summary.json")
    a = ap.parse_args()
    ev = Path(a.eval)
    an = json.loads((ev / "analysis.json").read_text())
    C = an["confirmatory"]
    rng = lambda xs: {"min": float(min(xs)), "max": float(max(xs))}
    sl = C["clinical_slope"]
    out = {"note": "descriptive ranges for the manuscript; spline reference", "n_cells": C["n_cells"], "n_tests": C["n_tests"],
           "critical_value": C["critical_value"]["D_logit"], "edits": C["edits"],
           "slope": {s: {"beta": v["beta"], "ci_simultaneous": v["ci_simultaneous"], "excludes_one": v["excludes_one"],
                         "beta_without_age": v["beta_without_age"],
                         "smoking_weight": v["field_weights"]["C5_smoking_never_to_current"]} for s, v in sl.items()},
           "slope_chatbots": {"beta": rng([sl[s]["beta"] for s in MAIN]),
                              "any_excludes_one": any(sl[s]["excludes_one"] for s in MAIN)}}
    dep = []
    for e, cells in C["cells"].items():
        for s, x in cells.items():
            lo, hi = x["D_logit_ci_simultaneous"]
            if lo > 0 or hi < 0:
                dep.append({"edit": e, "system": s, "D_logit": x["D_logit"], "ci_simultaneous": [lo, hi],
                            "or_system": x["implied_or_model"], "or_reference": x["or_reference"], "n": x["n_profiles"]})
    out["departures"] = dep
    chat = [d for d in dep if d["system"] in MAIN]
    out["chatbot_departures"] = {"n": len(chat), "n_cells": len(C["edits"]) * len(MAIN),
                                 "max_abs_D_logit": max((abs(d["D_logit"]) for d in chat), default=0.0)}
    clin = sl["jev"]["fields"]
    loo = {}
    for s_ in ["jev"] + MAIN:  # slope through the origin on the edit-level log odds ratios; reproduces clinical_slope.beta
        r = {e: math.log(C["cells"][e][s_]["or_reference"]) for e in clin}
        m = {e: math.log(C["cells"][e][s_]["implied_or_model"]) for e in clin}
        b = lambda es: sum(r[e] * m[e] for e in es) / sum(r[e] ** 2 for e in es)
        assert abs(b(clin) - sl[s_]["beta"]) < 1e-6, (s_, b(clin), sl[s_]["beta"])
        loo[s_] = {e: b([x for x in clin if x != e]) for e in clin}
    out["slope_leave_one_edit_out"] = loo
    out["n_departures"] = len(dep)
    out["departures_by_system"] = {s: sum(d["system"] == s for d in dep) for s in ["jev"] + MAIN}
    out["cells"] = {e: {s: {"or_system": x["implied_or_model"], "D_logit": x["D_logit"]} for s, x in cells.items()} | {"or_reference": cells["jev"]["or_reference"], "n": cells["jev"]["n_profiles"]}
                    for e, cells in C["cells"].items()}
    cs = json.loads((ev / "cost_speed.json").read_text())
    sysc = cs["systems"]
    key = "usd_list_per_1000_answers"
    out["cost_key"] = key
    out["usd_list_per_1000"] = {s: sysc[s][key] for s in sysc if isinstance(sysc[s], dict) and key in sysc[s]}
    out["usd_list_per_1000_chatbots"] = rng([out["usd_list_per_1000"][s] for s in MAIN])
    out["chatbot_over_jev_cost"] = rng([out["usd_list_per_1000"][s] / out["usd_list_per_1000"]["jev"] for s in MAIN])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["slope_chatbots", "n_departures", "departures_by_system", "cost_key", "usd_list_per_1000_chatbots", "chatbot_over_jev_cost"]}, indent=1))


if __name__ == "__main__":
    main()
