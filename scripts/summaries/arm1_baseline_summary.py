"""Arm 1 risk estimation (NHANES): baseline prediction of Jev and the references; ranges over the five main chatbots.

Reads results/wide/analysis.json (prediction section) and writes results/summaries/arm1_baseline_summary.json:
observed death share, and for Jev and the chatbots log-loss, recalibrated log-loss, Brier, AUROC, calibration and
mean predicted risk; Jev minus chatbot log-loss (raw, recalibrated) with the simultaneous intervals as min/max.
  python scripts/summaries/arm1_baseline_summary.py [--wide results/wide/analysis.json]
"""
import argparse, json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
KEYS = ["log_loss", "log_loss_recalibrated", "brier", "auroc", "calibration_slope", "calibration_intercept", "mean_p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="results/wide/analysis.json")
    ap.add_argument("--out", default="results/summaries/arm1_baseline_summary.json")
    a = ap.parse_args()
    p = json.loads(Path(a.wide).read_text())["prediction"]
    S = p["systems"]
    rng = lambda xs: {"min": float(min(xs)), "max": float(max(xs))}
    out = {"note": "descriptive ranges for the manuscript; values from results/wide/analysis.json",
           "n_records": p["n_records"], "deaths": p["deaths"], "observed_death_share": p["deaths"] / p["n_records"],
           "jev": {k: S["jev"][k] for k in KEYS}, "reference": {k: S["reference_spline_logit"][k] for k in KEYS},
           "age_sex": {k: S["reference_age_sex"][k] for k in KEYS},
           "chatbots": {k: rng([S[s][k] for s in MAIN]) for k in KEYS}}
    for sec, name in [("focal_vs_llms", "jev_minus_chatbot_log_loss"), ("focal_vs_llms_recalibrated", "jev_minus_chatbot_log_loss_recalibrated")]:
        pr = p[sec]["pairs"]
        d = [pr[f"jev - {s}"]["diff"] for s in MAIN]
        lo = [pr[f"jev - {s}"]["ci_simultaneous"][0] for s in MAIN]
        out[name] = {"diff": rng(d), "simultaneous_lower": rng(lo),
                     "noninferior_any": any(pr[f"jev - {s}"]["noninferior"] for s in MAIN),
                     "margin": pr["jev - gpt"]["noninferiority_margin"]}
    out["jev_minus_age_sex_log_loss"] = S["jev"]["log_loss"] - S["reference_age_sex"]["log_loss"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
