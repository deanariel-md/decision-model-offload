"""Arm 2 next-step choice, further descriptive analyses from results/arm2/analysis.json: Jev's overuse errors by
recommendation (total, the four largest groups, the thyroid group), the analysis that resamples whole recommendations,
and the count of Jev's certain answers that were wrong.

  python scripts/summaries/arm2_further_summary.py [--analysis results/arm2/analysis.json]
    -> results/summaries/arm2_further_summary.json
"""
import argparse, json
from pathlib import Path

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
ap = argparse.ArgumentParser()
ap.add_argument("--analysis", default="results/arm2/analysis.json")
ap.add_argument("--out", default="results/summaries/arm2_further_summary.json")
a = ap.parse_args()
ph = json.loads(Path(a.analysis).read_text())["post_hoc"]
g = ph["rates_by_group"]["groups"]
over = sorted(((v["rates"]["jev"]["overuse"]["n"], k) for k, v in g.items()), reverse=True)
top4 = over[:4]
cs = ph["cluster_sensitivity"]["comparisons"]
res = {"note": "descriptive; added after the results",
       "jev_overuse_total": sum(n for n, _ in over),
       "jev_overuse_top4": sum(n for n, _ in top4), "top4": [k for _, k in top4],
       "thyroid": g["rec_04"]["rates"]["jev"]["overuse"],
       "n_recommendations": ph["cluster_sensitivity"]["n_groups"],
       "cluster_outcomes": {s: cs[s]["outcome"] for s in MAIN},
       "cluster_all_inconclusive": all(cs[s]["outcome"] == "inconclusive" for s in MAIN),
       "cluster_holm_upper_pts": {"min": min(100 * cs[s]["holm_upper_bound"] for s in MAIN),
                                  "max": max(100 * cs[s]["holm_upper_bound"] for s in MAIN)},
       "certain_errors": {"n_certain": ph["certain_errors"]["n_certain"], "n_wrong": ph["certain_errors"]["n_wrong"]}}
Path(a.out).write_text(json.dumps(res, indent=1))
print(res)
