"""Arm 1 risk estimation (NHANES): the hybrid against each of the five main chatbots, from results/wide/hybrid.json.
  python scripts/summaries/arm1_hybrid_summary.py -> results/summaries/arm1_hybrid_summary.json"""
import json
from pathlib import Path
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
h = json.loads(Path("results/wide/hybrid.json").read_text())
C = h["chatbots"]
rng = lambda xs: {"min": min(xs), "max": max(xs)}
res = {"note": "descriptive summary of results/wide/hybrid.json", "margin": h["margin"], "n_records": h["n_records"],
       "hybrid_minus_chatbot_log_loss": rng([C[s]["diff"] for s in MAIN]),
       "all_worse": all(C[s]["outcome"] == "worse" for s in MAIN),
       "confidence_minus_random_log_loss": rng([C[s]["confidence_minus_random"] for s in MAIN]),
       "outcomes": {s: C[s]["outcome"] for s in MAIN}}
Path("results/summaries/arm1_hybrid_summary.json").write_text(json.dumps(res, indent=1))
print(res)
