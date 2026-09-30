"""Descriptive supplement tables (src/jevity/supplement.py): the reference change per confirmatory edit, answers at
exactly 0 or 1, baseline log-loss under three clipping bounds, run-to-run spread from the repeat calls, repeatability
on the repeat set (ICC(2,1) and the share answered identically, as arm 3 computes them), and answers at the output cap
(framing: the cap table only). Reads the inputs of scripts/analyze.py; run after it. Stops if its log-loss at the
analysis's clipping bound or its reference odds ratios differ from results/<split>/analysis.json. Writes
results/<split>/supplement_descriptives.json. No network.
  python scripts/supplement_descriptives.py eval
  python scripts/supplement_descriptives.py wide
  python scripts/supplement_descriptives.py framing"""
import argparse, json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import analysis as A
from jevity import supplement as S
from jevity.reference import phenoage_mortality_10y

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze import jsonable     # noqa: E402

TOL = 1e-9


def comparators(root: Path, cohort: pd.DataFrame, split: str) -> dict:
    """The comparator predictions of scripts/analyze.py (the prevalence-only comparator is not in the prediction block)."""
    ev = cohort[cohort.split == split].set_index(cohort.loc[cohort.split == split, "SEQN"].astype(int))
    comps = {}
    for n in ("spline_logit", "lightgbm", "survival_logit", "age_sex"):
        f = root / "data" / f"reference_{n}.joblib"
        if f.exists():
            comps[f"reference_{n}"] = pd.Series(joblib.load(f).predict(ev), index=ev.index)
    comps["phenoage_10y"] = pd.Series(np.clip(phenoage_mortality_10y(ev).to_numpy(), 1e-6, 1 - 1e-6), index=ev.index)
    return comps


def check(res: dict, reported: dict | None, split: str) -> list[str]:
    """Agreement with results/<split>/analysis.json: log-loss and its interval at the analysis's clipping bound, and
    (eval) the primary reference's odds ratio per edit."""
    if reported is None:
        return [f"results/{split}/analysis.json missing: nothing checked"]
    bad = []
    blk = res["log_loss_clipping"]["by_clip"][str(A.CLIP[0])]["systems"]
    for nm, v in reported["prediction"]["systems"].items():
        mine = blk.get(nm)
        if mine is None or abs(mine["log_loss"] - v["log_loss"]) > TOL or \
                any(abs(a - b) > TOL for a, b in zip(mine["log_loss_ci"], v["log_loss_ci"])):
            bad.append(f"log-loss {nm}")
    if reported["prediction"]["n_records"] != res["log_loss_clipping"]["n_records"]:
        bad.append("prediction records")
    if split == "eval":
        for e, orr in reported["confirmatory"].get("reference_or_all_supported", {}).items():
            mine = res["reference_change"].get("spline_logit", {}).get(e, {}).get("odds_ratio")
            if mine is None or abs(mine - orr) > 1e-9 * max(1.0, orr):
                bad.append(f"reference odds ratio {e}")
    if bad:
        raise SystemExit(f"differs from results/{split}/analysis.json: " + ", ".join(bad))
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["eval", "wide", "framing"])
    ap.add_argument("--B", type=int, default=2000, help="draws for the reference-change intervals (eval)")
    ap.add_argument("--B_pred", type=int, default=1000,
                    help="record resamples of the prediction block (analysis.prediction_block: 1,000 for eval and wide)")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)      # tests: a synthetic tree
    a = ap.parse_args()
    root = Path(a.root)
    cohort = pd.read_parquet(root / "data" / "cohort.parquet")
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    calls = pd.read_parquet(root / "results" / a.split / "calls.parquet")
    reported_f = root / "results" / a.split / "analysis.json"
    reported = json.loads(reported_f.read_text()) if reported_f.exists() else None
    models = S.systems(calls)
    prim = [m for m in models if A.is_primary(m)]
    res = {"meta": {"split": a.split, "note": "descriptive supplement tables; outside the confirmatory family",
                    "systems": models, "primary_systems": prim, "clips": list(S.CLIPS), "fixed_clip": A.CLIP[0]}}
    res["output_cap"] = S.output_cap(calls, models)
    if a.split == "framing":
        out = root / "results" / "framing" / "supplement_descriptives.json"
        out.write_text(json.dumps(jsonable(res), indent=1))
        print(f"framing: output cap table; wrote {out}")
        sys.exit(0)
    if a.split == "wide":
        oof = pd.read_parquet(root / "data" / "reference_oof.parquet").set_index("SEQN")
        profiles = sorted(set(oof.index))
        models = A.full_set_systems(models)
        preds = {m: calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & A._variant(calls, "raw")]
                 .drop_duplicates("profile").set_index("profile")["p"].reindex(profiles) for m in models}
        for col, name in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                          ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
            if col in oof:
                preds[name] = oof[col].reindex(profiles)
        yv = y.reindex(profiles).dropna()
    else:
        inst = pd.read_parquet(root / "data" / "instances_eval.parquet")
        refits = {r: A.Refits.load(f) for r in S.REFERENCE_ORDER
                  if (f := root / "data" / f"refit_d_{r}_eval.npz").exists()}
        profiles = sorted(cohort.loc[cohort.split == "eval", "SEQN"].astype(int))
        res["reference_change"] = S.reference_change(inst, refits, profiles, B=a.B)
        res["repeat_spread"] = S.repeat_spread(calls, prim)
        res["repeatability"] = S.repeatability(calls, prim, B=a.B)     # the analysis's number of draws and seed
        preds = A.baseline_predictions(calls, models, profiles, comparators(root, cohort, "eval"))
        yv = y.reindex(profiles).dropna()
    res["extreme_answers"] = S.extreme_answers(calls, models)
    res["log_loss_clipping"] = S.log_loss_clipping(preds, yv, versus=[m for m in prim if m != "jev"] + ["reference_age_sex"],
                                                   B=a.B_pred)
    res["meta"]["checked_against_analysis_json"] = not check(res, reported, a.split)
    out = root / "results" / a.split / "supplement_descriptives.json"
    out.write_text(json.dumps(jsonable(res), indent=1))
    ll = res["log_loss_clipping"]
    print(f"{a.split}: {ll['n_records']:,} records; checked against analysis.json: {res['meta']['checked_against_analysis_json']}")
    print(f"wrote {out}")
