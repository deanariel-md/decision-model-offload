"""Median and 90th-percentile response time per system from the timing samples (stored outputs only).

  python scripts/summaries/timing_summary.py -> results/summaries/timing_summary.json
Arm 1 risk estimation: results/eval/cost_speed.json (timing_median_s, timing_p90_s); arm 2 next-step choice:
results/arm2/timing_sample.json (median_s, p90_s); board examination: results/arm2_ext/timing_sample.json; arm 3
staging: results/arm3/timing_sample.json (same fields; 100 reports per system in one window).
"""
import json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
A2 = Path("results/arm2/timing_sample.json")
t1 = json.loads(Path("results/eval/cost_speed.json").read_text())["systems"]
t2 = json.loads(A2.read_text())["systems"]
res = {"note": "median seconds per answer, standard route, timing sample of 100 items per system",
       "risk": {s: {"median_s": t1[s]["timing_median_s"], "p90_s": t1[s]["timing_p90_s"]} for s in t1 if t1[s].get("timing_median_s") is not None},
       "next_step": {s: {"median_s": v["median_s"], "p90_s": v["p90_s"]} for s, v in t2.items()}}
for task, f in (("board_exam", "results/arm2_ext/timing_sample.json"), ("staging", "results/arm3/timing_sample.json")):
    res[task] = {s: {"median_s": v["median_s"], "p90_s": v["p90_s"]}
                 for s, v in json.loads(Path(f).read_text())["systems"].items()}
for task in ("risk", "next_step", "board_exam", "staging"):
    m = [res[task][s]["median_s"] for s in MAIN]
    res[task + "_main_range"] = {"min": min(m), "max": max(m)}
    m4 = [res[task][s]["median_s"] for s in MAIN if s != "glm"]
    res[task + "_main_range_without_glm"] = {"min": min(m4), "max": max(m4)}
    res[task + "_jev_ms"] = 1000 * res[task]["jev"]["median_s"]
TASKS = ("risk", "next_step", "board_exam", "staging")
res["all_tasks_jev_s"] = {"min": min(res[k]["jev"]["median_s"] for k in TASKS), "max": max(res[k]["jev"]["median_s"] for k in TASKS)}
res["all_tasks_main_range"] = {"min": min(res[k + "_main_range"]["min"] for k in TASKS),
                               "max": max(res[k + "_main_range"]["max"] for k in TASKS)}
Path("results/summaries/timing_summary.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k.endswith(("range", "glm", "_ms"))})
print("glm", res["risk"]["glm"], res["next_step"]["glm"])
