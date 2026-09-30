"""eICU replication: baseline prediction summary from results/eicu/analysis.json (Jev, APACHE IVa, ranges over the five
main chatbots and the three references; every predictor scored on the stays all of them scored; no records are read).

  python scripts/summaries/eicu_summary.py -> results/summaries/eicu_summary.json
"""
import json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
REF = ["reference_spline_logit", "reference_lightgbm", "reference_xgboost"]
KEYS = ["log_loss", "auroc", "brier", "calibration_slope", "mean_predicted"]

a = json.loads(Path("results/eicu/analysis.json").read_text())
b = a["baseline_prediction"]
P = b["predictors"]
rng = lambda ss, k: {"min": min(P[s][k] for s in ss), "max": max(P[s][k] for s in ss)}
res = {"note": "descriptive; every predictor scored on the stays all of them scored",
       "n_records": b["n_records"], "deaths": b["deaths"], "observed": P["jev"]["observed"],
       "evaluation_stays": a["meta"]["evaluation_stays"],
       "jev": {k: P["jev"][k] for k in KEYS} | {"log_loss_ci": P["jev"]["log_loss_ci"], "mean_predicted_ci": P["jev"]["mean_predicted_ci"]},
       "apache_iva": {k: P["apache_iva"][k] for k in KEYS},
       "main_chatbots": {k: rng(MAIN, k) for k in KEYS},
       "references": {k: rng(REF, k) for k in KEYS},
       "jev_highest_log_loss": P["jev"]["log_loss"] == max(P[s]["log_loss"] for s in P),
       "jev_lowest_auroc": P["jev"]["auroc"] == min(P[s]["auroc"] for s in P)}
Path("results/summaries").mkdir(parents=True, exist_ok=True)
Path("results/summaries/eicu_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
