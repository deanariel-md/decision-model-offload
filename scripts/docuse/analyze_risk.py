"""Score Jev used as documented (results/wide_docuse/calls.parquet) with the code that produced the original results.
First the same code runs on Jev's ORIGINAL answers and must reproduce results/wide/analysis.json (analysis.run_wide,
whole file), the evaluation prediction block of results/eval/analysis.json and the five-cheaper-systems block of
results/summaries/cheap_family.json; the script stops on any difference. Then, with only Jev's answers changed:
  records                  the 12,697 records every system answered in the original analysis (and the 983 evaluation
                           records), intersected with the records documented Jev answered: counts and dropped ids
  prediction               those records: log-loss (95% CI), Brier, AUROC, calibration slope, mean predicted,
                           cross-fitted recalibrated log-loss; Jev used as documented minus each main chatbot and minus
                           the references; non-inferiority at the fixed analysis.wide_margin (noninferior_fixed_margin)
                           beside prediction_block's margin on the records scored (noninferior)
  documented_vs_original   Jev used as documented minus Jev as run originally, same records
  eval_prediction          the evaluation records, as the evaluation analysis, fixed analysis.eval_margin
  eval_cheap_family        the evaluation records, against the five cheaper systems, fixed analysis.eval_margin
  validity                 usable answers, parse categories, providers, tokens, list cost, latency
No model call. Writes results/wide_docuse/analysis.json.
  python scripts/docuse/analyze_risk.py [--calls results/wide_docuse/calls.parquet] [--reproduce-only]"""
import argparse, json, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import docuse_risk as D


def _shown(p: Path, root: Path) -> str:
    """A path as written into the output: relative to the repository root when it is inside it."""
    p = Path(p)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(p)


def main(calls_path: Path, out: Path, reproduce_only: bool = False, root: Path = ROOT) -> dict:
    inp = D.inputs(root)
    rep = D.reproduce(inp, root)
    shown = {k: v for k, v in rep.items() if k != "_state"}
    print("reproduced on Jev's original answers:", json.dumps(shown, indent=1))
    if reproduce_only:
        return shown
    calls = pd.read_parquet(calls_path)
    res = D.score(calls, inp, rep, root)
    res = {"meta": {"calls": _shown(calls_path, root), "config": "config/docuse_risk.yaml",
                    "config_sha256": D.config_sha256(), "variant": D.VARIANT,
                    "written": time.strftime("%Y-%m-%d %H:%M:%S"), "reproduction": shown}, **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    pr = res["prediction"]
    print(f"wide: {pr['n_records']:,} records, {pr['deaths']:,} deaths")
    for k, v in pr["systems"].items():
        print(f"  {k:24s} log-loss {v['log_loss']:.4f}  recal {v['log_loss_recalibrated']:.4f}  AUROC {v['auroc']:.3f}  "
              f"slope {v['calibration_slope']:.2f}")
    for nm, r in res["records"].items():
        print(f"  records {nm}: {r['scored']:,} scored of {r['original']:,}; dropped {r['dropped']}")
    for blk in ("focal_vs_llms", "focal_vs_age_sex"):
        for k, v in pr.get(blk, {}).get("pairs", {}).items():
            print(f"  {k:32s} diff log-loss {v['diff']:+.4f} {v['ci_simultaneous']} non-inferior at the fixed margin "
                  f"{v['noninferiority_margin_fixed']:.6f}: {v['noninferior_fixed_margin']}")
    for k, v in res["eval_cheap_family"]["comparisons"].items():
        print(f"  eval, cheaper: {k:12s} {v['diff']:+.4f} {v['ci_simultaneous']} {v['outcome']}")
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=str(ROOT / "results" / D.CFG["name"] / "calls.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / D.CFG["name"] / "analysis.json"))
    ap.add_argument("--reproduce-only", action="store_true")
    a = ap.parse_args()
    main(Path(a.calls), Path(a.out), a.reproduce_only)
