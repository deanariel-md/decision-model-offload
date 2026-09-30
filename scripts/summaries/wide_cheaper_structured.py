"""Jev, both ways of asking, against the five cheaper systems on the full ten-year-risk set.

The documented-use scoring code (jevity.docuse_risk: wide_predictions, cheap_family) on results/wide/calls.parquet plus
the cheaper systems' wide calls (results/wide_cheaper/calls.parquet), with the wide margin and the cheaper-family
resampling of config/docuse_risk.yaml (cheap_B, cheap_seed). Records: the 12,697 of results/wide/analysis.json. Before
scoring, each cheaper system's log-loss on these records must equal results/wide_cheaper/analysis.json. No model call.

    python scripts/summaries/wide_cheaper_structured.py --root . --cheaper results/wide_cheaper
Writes results/summaries/wide_cheaper_structured.json. --root: the repository root (src/, data/cohort.parquet,
data/reference_oof.parquet, results/wide/ and results/wide_docuse/); --cheaper: the folder holding the cheaper
systems' calls.parquet, analysis.json and summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cheaper", required=True)
    ap.add_argument("--out", default=str(HERE / "results" / "summaries" / "wide_cheaper_structured.json"))
    a = ap.parse_args()
    root, cdir = Path(a.root), Path(a.cheaper)
    sys.path.insert(0, str(root / "src"))
    from jevity import analysis as A
    from jevity import docuse_risk as D

    ac = D.CFG["analysis"]
    wm, B, seed, cheap = float(ac["wide_margin"]), int(ac["cheap_B"]), int(ac["cheap_seed"]), list(ac["cheap_family"])
    wide = pd.read_parquet(root / "results" / "wide" / "calls.parquet")
    cc = pd.read_parquet(cdir / "calls.parquet")
    assert list(cc.columns) == list(wide.columns), "calls tables differ in columns"
    assert not (set(cc.model) & set(wide.model)), "a system is in both calls tables"
    calls = pd.concat([wide, cc[cc.model.isin(cheap)]], ignore_index=True)
    cohort = pd.read_parquet(root / "data" / "cohort.parquet")
    oof = pd.read_parquet(root / "data" / "reference_oof.parquet")
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    wj = json.loads((root / "results" / "wide" / "analysis.json").read_text(encoding="utf-8"))
    cj = json.loads((cdir / "analysis.json").read_text(encoding="utf-8"))
    sm = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))

    base, _ = D.wide_predictions(wide, oof)
    common = D.common_records(base, y)
    assert len(common) == wj["prediction"]["n_records"] == cj["prediction"]["n_records"], "record sets differ"
    yc = y.reindex(common)

    jd = D.documented_answers(pd.read_parquet(root / "results" / "wide_docuse" / "calls.parquet")).dropna()
    assert set(common) <= set(jd.index), "documented Jev did not answer every record"
    orig, _ = D.wide_predictions(calls, oof)
    doc, _ = D.wide_predictions(calls, oof, jev=jd, name="jev_documented")

    check = {}
    for m in cheap:
        ll = float(A._ll(orig[m].loc[common].to_numpy(float), yc.to_numpy(float)).mean())
        ref = cj["prediction"]["systems"][m]["log_loss"]
        assert abs(ll - ref) < 1e-12, (m, ll, ref)
        check[m] = {"log_loss": ll, "stored": ref}
    ll_doc = float(A._ll(doc["jev_documented"].loc[common].to_numpy(float), yc.to_numpy(float)).mean())

    out = {"note": "log-loss, lower is better; difference Jev minus system; simultaneous 95% intervals (max-t over the "
                   "five cheaper systems); non-inferior when the upper bound is below the fixed wide margin, worse when "
                   "the lower bound is above it",
           "records": len(common), "deaths": int(yc.sum()), "margin": wm, "B": B, "seed": seed,
           "check_cheaper_log_loss": check, "jev_documented_log_loss": ll_doc,
           "systems": {m: {k: cj["prediction"]["systems"][m][k] for k in
                           ["log_loss", "log_loss_ci", "brier", "auroc", "calibration_slope", "mean_p", "log_loss_recalibrated"]}
                       for m in cheap},
           "usd_list_all_answers": {m: sm["per_system"][m]["usd_all_answers_list"] for m in cheap},
           "calls": {m: {"planned": sm["per_system"][m]["planned"], "usable": sm["per_system"][m]["usable"]} for m in cheap},
           "structured": D.cheap_family({k: doc[k] for k in ["jev_documented"] + cheap}, yc, "jev_documented", cheap,
                                        wm, B, seed, label="Jev, structured input"),
           "chatbot_prompt": D.cheap_family({k: orig[k] for k in ["jev"] + cheap}, yc, "jev", cheap, wm, B, seed,
                                            label="Jev, chatbot prompt")}
    for blk in ("structured", "chatbot_prompt"):
        assert out[blk]["n_records"] == len(common), blk
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)),
                           encoding="utf-8")
    for blk in ("structured", "chatbot_prompt"):
        for k, v in out[blk]["comparisons"].items():
            print(f"{blk:15s} {k:12s} {v['diff']:+.4f} [{v['ci_simultaneous'][0]:+.4f}, {v['ci_simultaneous'][1]:+.4f}] "
                  f"{v['outcome']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
