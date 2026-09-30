"""Arm 1 risk estimation (NHANES), further descriptive analyses: ranges over the five main chatbots from
results/wide/arm1_extras.json (format dependence, the M1 instruction's effect on the race edit, the smoking gradient
in real records).

  python scripts/summaries/arm1_extras_summary.py -> results/summaries/arm1_extras_summary.json
"""
import json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
x = json.loads(Path("results/wide/arm1_extras.json").read_text())
fd, m1, sg = x["format_dependence"]["drift"], x["m1_race_edit"], x["smoking_gradient"]["systems"]
def rng(v):
    v = list(v)
    return {"min": min(v), "max": max(v)}
res = {"note": "ranges over the five main chatbots; descriptive, outside the fixed family",
       "format": {"jev": fd["jev"], "chatbots_excess_pts": rng(fd[s]["excess_pts"] for s in MAIN),
                  "chatbots_prose_abs_pts": rng(fd[s]["by_edit"]["I3_prose"]["mean_abs_pts"] for s in MAIN),
                  "jev_highest_excess": fd["jev"]["excess_pts"] > max(fd[s]["excess_pts"] for s in MAIN)},
       "bands": x["format_dependence"]["jev_bands_vs_noul"], "complement": x["format_dependence"]["jev_complement"],
       "m1": {"jev": m1["jev"], "chatbots_diff_logit": rng(m1[s]["diff_logit"] for s in MAIN),
              "all_chatbots_shrink": all(m1[s]["diff_logit"] < 0 for s in MAIN)},
       "smoking_real": {"jev": sg["jev"], "chatbots_gap": rng(sg[s]["log_odds_gap"] for s in MAIN),
                        "reference": sg["reference_spline_logit"],
                        "observed_current": x["smoking_gradient"]["observed_risk_current"],
                        "observed_never": x["smoking_gradient"]["observed_risk_never"]}}
Path("results/summaries/arm1_extras_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("m1",)}, indent=0)[:600], res["format"]["chatbots_excess_pts"], res["format"]["jev_highest_excess"])
