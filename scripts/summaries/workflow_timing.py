"""Time per report for the recommended-form workflow in cancer staging (stored timings only).

On the reports timed for both Jev in the recommended form (results/arm3_docuse/timing_sample.parquet, --docuse folder)
and each main LLM with stage names (results/arm3/timing_sample.parquet): time = Jev's latency (as timing_sample.py), plus the LLM's latency (as timing_sample.py)
when Jev's answer fell short of full confidence (joint top probability >= 1 - 1e-9 in the main run, workflow_checks.py's cut).
Reports: mean and median per system, and workflow mean / LLM mean.

  python scripts/summaries/workflow_timing.py [--docuse .] -> results/summaries/workflow_timing.json
--docuse: the folder holding results/arm3_docuse (default: the working directory, the repository root).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
TOL = 1e-9

ap = argparse.ArgumentParser(); ap.add_argument("--docuse", default="."); a = ap.parse_args()
c = pd.read_parquet(f"{a.docuse}/results/arm3_docuse/calls.parquet")
c = c[(c["variant"] == "documented") & (c["system"] == "jev") & (c["repeat"] == 0)].set_index("item_id")
full = c["top_prob"].astype(float) >= 1 - TOL
jt = pd.read_parquet(f"{a.docuse}/results/arm3_docuse/timing_sample.parquet")
jt = jt[(jt["system"] == "jev") & jt["valid"]].set_index("item_id")["latency_s"]
lt = pd.read_parquet("results/arm3/timing_sample.parquet")
out = {"note": __doc__.strip().splitlines()[0], "jev_recommended_form": {"n": int(len(jt)), "median_s": float(jt.median()), "mean_s": float(jt.mean()),
       "share_full_confidence": float(full.reindex(jt.index).mean())}, "systems": {}}
for s in MAIN:
    ls = lt[(lt["system"] == s) & lt["valid"]].set_index("item_id")["latency_s"]
    idx = jt.index.intersection(ls.index)
    f = full.reindex(idx).fillna(False).to_numpy()
    wf = jt[idx].to_numpy() + np.where(f, 0.0, ls[idx].to_numpy())
    out["systems"][s] = {"n": int(len(idx)), "llm_median_s": float(ls[idx].median()), "llm_mean_s": float(ls[idx].mean()),
                         "workflow_median_s": float(np.median(wf)), "workflow_mean_s": float(wf.mean()),
                         "mean_ratio": float(wf.mean() / ls[idx].mean())}
r = [v["mean_ratio"] for v in out["systems"].values()]
out["summary"] = {"mean_ratio": {"min": min(r), "max": max(r)},
                  "llm_median_s": {"min": min(v["llm_median_s"] for v in out["systems"].values()), "max": max(v["llm_median_s"] for v in out["systems"].values())},
                  "workflow_median_s": {"min": min(v["workflow_median_s"] for v in out["systems"].values()), "max": max(v["workflow_median_s"] for v in out["systems"].values())}}
Path("results/summaries").mkdir(parents=True, exist_ok=True)
Path("results/summaries/workflow_timing.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["summary"]), json.dumps(out["jev_recommended_form"]))
