"""Arm 1 risk estimation (NHANES), secondary results from results/eval/analysis.json and results/wide/analysis.json:
LightGBM as the reference (clinical response slopes), reasoning effort high against low (200 participants), and the
run counts (bootstrap draws, reference refits, baseline and evaluation records).
  python scripts/summaries/arm1_secondary_summary.py -> results/summaries/arm1_secondary_summary.json"""
import json
from pathlib import Path
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
a = json.loads(Path("results/eval/analysis.json").read_text())
w = json.loads(Path("results/wide/analysis.json").read_text())
def rng(xs):
    xs = list(xs)
    return {"min": min(xs), "max": max(xs)}
s2 = a["confirmatory_second_reference"]["clinical_slope"]
s1 = a["confirmatory"]["clinical_slope"]
ra = a["reasoning_arm"]
res = {"note": "descriptive summary of results/eval/analysis.json and results/wide/analysis.json",
       "lightgbm_slope": {"jev": s2["jev"]["beta"], "jev_ci_simultaneous": s2["jev"]["ci_simultaneous"],
                          "chatbots": rng(s2[m]["beta"] for m in MAIN)},
       "spline_slope": {"jev": s1["jev"]["beta"], "chatbots": rng(s1[m]["beta"] for m in MAIN)},
       "reasoning": {"n_participants": ra["gpt"]["n_profiles"],
                     "slope_high_minus_low": rng(ra[m]["high_minus_low"]["clinical_slope"]["diff"] for m in MAIN),
                     "slope_ci_excludes_zero": sum(ra[m]["high_minus_low"]["clinical_slope"]["ci"][0] > 0 or
                                                   ra[m]["high_minus_low"]["clinical_slope"]["ci"][1] < 0 for m in MAIN),
                     "per_call_error_high_minus_low": rng(ra[m]["high_minus_low"]["per_call_error_diff"]["diff"] for m in MAIN)},
       "counts": {"bootstrap_draws": a["meta"]["B"], "reference_refits": a["meta"]["reference_refits"],
                  "baseline_records": w["prediction"]["n_records"], "baseline_deaths": w["prediction"]["deaths"],
                  "evaluation_records": a["prediction"]["n_records"]}}
Path("results/summaries/arm1_secondary_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
