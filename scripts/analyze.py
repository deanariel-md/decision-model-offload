"""Analyse a split. Writes results/<split>/analysis.json (every analysis output of the split).
  python scripts/analyze.py eval [--B 2000]
  python scripts/analyze.py wide"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import analysis as A
from jevity.reference import phenoage_mortality_10y

ROOT = Path(__file__).resolve().parents[1]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def complete_blocks(calls: pd.DataFrame, inst: pd.DataFrame, block: int = 50):
    """For a run that stopped early: keep the blocks (runner.block_order) in which every planned call was attempted; a
    call never attempted is a missing batch answer or a refusal for spending (HTTP 402)."""
    from jevity.runner import Call, block_order
    from jevity.clients import BATCH_MISSING
    err = calls["error"].fillna("").astype(str)
    unattempted = (err == BATCH_MISSING) | err.str.contains("402")
    order = list(dict.fromkeys(c.profile for c in block_order([Call("x", int(p), "", False, 0, "") for p in
                                                              sorted(calls.profile.unique())], block=block)))
    blk = {p: i // block for i, p in enumerate(order)}
    bad = {blk[int(p)] for p in calls.loc[unattempted, "profile"]}
    keep = {p for p, b in blk.items() if b not in bad}
    print(f"complete blocks: {len(set(blk.values())) - len(bad)} of {len(set(blk.values()))} ({len(keep)} records)")
    return calls[calls.profile.isin(keep)], inst[inst.profile.isin(keep)], sorted(set(blk.values()) - bad)


def check_refits(refits: dict, inst: pd.DataFrame, cohort: pd.DataFrame, synthetic: bool) -> None:
    """The evaluation analysis stops unless every required refit set exists, is large enough and covers every
    supported instance exactly once; a sampling-only (degenerate) reference is never used for an interval."""
    import yaml
    need = yaml.safe_load((ROOT / "config" / "analysis.yaml").read_text())["refits_required"]
    weights = "wt_mec_10y" in cohort and cohort["wt_mec_10y"].notna().any()
    sup = inst[inst.applicable & inst.supported & (inst.klass != "irrelevant")]
    keys = set(zip(sup.profile.astype(int), sup.edit.astype(str)))
    for ref, R_need in need.items():
        if ref == "survival_logit" and "d_logit_survival_logit" not in inst:
            continue
        if ref == "spline_logit_weighted" and not weights:
            continue
        if ref not in refits:
            raise SystemExit(f"data/refit_d_{ref}_*.npz missing: run scripts/refit_references.py first")
        r = refits[ref]
        if r.R < R_need and not synthetic:
            raise SystemExit(f"{ref}: {r.R} refits, {R_need} required")
        if len(r.index) != r.d_logit.shape[0]:
            raise SystemExit(f"{ref}: duplicated refit keys")
        if not (np.isfinite(r.d_logit).all() and np.isfinite(r.d_prob).all()):
            raise SystemExit(f"{ref}: non-finite refit contrasts")
        miss = keys - set(r.index)
        if miss:
            raise SystemExit(f"{ref}: {len(miss)} supported instances have no refit row")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["eval", "wide"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--B_desc", type=int, default=500)
    ap.add_argument("--complete-blocks", action="store_true",
                    help="for a run that stopped early: keep the blocks of 50 records answered by every system")
    ap.add_argument("--synthetic", action="store_true",
                    help="dry run only: accept fewer refits than required; results are marked synthetic")
    a = ap.parse_args()
    cohort = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    if a.split == "wide":
        calls = pd.read_parquet(ROOT / "results" / "wide" / "calls.parquet")
        oof = pd.read_parquet(ROOT / "data" / "reference_oof.parquet")
        res = A.run_wide(calls, cohort, oof, B=min(a.B, 1000))
        (ROOT / "results" / "wide" / "analysis.json").write_text(json.dumps(jsonable(res), indent=1))
        pr = res["prediction"]
        print(f"wide: {pr['n_records']:,} records, {pr['deaths']:,} deaths")
        for k, v in pr["systems"].items():
            print(f"  {k:24s} log-loss {v['log_loss']:.4f}  recal {v['log_loss_recalibrated']:.4f}  AUROC {v['auroc']:.3f}  slope {v['calibration_slope']:.2f}")
        for k, v in pr.get("focal_vs_llms", {}).get("pairs", {}).items():
            print(f"  {k:24s} diff log-loss {v['diff']:+.4f} {v['ci_simultaneous']}")   # ASCII: cp1252 consoles
        sys.exit(0)
    inst = pd.read_parquet(ROOT / "data" / f"instances_{a.split}.parquet")
    calls = pd.read_parquet(ROOT / "results" / a.split / "calls.parquet")
    kept_blocks = None
    if a.complete_blocks:
        calls, inst, kept_blocks = complete_blocks(calls, inst)
    refits = {}
    for ref in ("spline_logit", "lightgbm", "survival_logit", "spline_logit_weighted"):
        f = ROOT / "data" / f"refit_d_{ref}_{a.split}.npz"
        if f.exists():
            refits[ref] = A.Refits.load(f)
    check_refits(refits, inst, cohort, synthetic=a.synthetic)
    profiles = sorted(cohort.loc[cohort.split == a.split, "SEQN"].astype(int))
    if kept_blocks is not None:
        profiles = sorted(set(profiles) & set(calls.profile.astype(int)))
    ev = cohort[cohort.split == a.split].set_index(cohort.loc[cohort.split == a.split, "SEQN"].astype(int))
    comps = {}
    for n in ("spline_logit", "lightgbm", "survival_logit", "age_sex"):
        f = ROOT / "data" / f"reference_{n}.joblib"
        if f.exists():
            comps[f"reference_{n}"] = pd.Series(joblib.load(f).predict(ev), index=ev.index)
    comps["phenoage_10y"] = pd.Series(np.clip(phenoage_mortality_10y(ev).to_numpy(), 1e-6, 1 - 1e-6), index=ev.index)
    fit_prev = cohort.loc[cohort.split == "fit", "death_10y"].mean()
    comps["prevalence_only"] = pd.Series(np.full(len(ev), fit_prev) + 1e-9 * np.arange(len(ev)), index=ev.index)
    res = A.run_all(calls, inst, cohort, refits, profiles, B=a.B, B_desc=a.B_desc, comparator_preds=comps)
    res["meta"]["synthetic"] = bool(a.synthetic)
    res["meta"]["complete_blocks"] = kept_blocks
    res["meta"]["refits_R"] = {k: v.R for k, v in refits.items()}
    out = ROOT / "results" / a.split / "analysis.json"
    out.write_text(json.dumps(jsonable(res), indent=1))
    fam = res["confirmatory"]
    print(f"confirmatory family: {fam['n_cells']} edit x model cells, critical value {fam['critical_value'].get('D_logit', float('nan')):.2f}")
    for e, cells in fam["cells"].items():
        print(f"  {e:30s} " + "  ".join(f"{m} {r['D_logit']:+.2f}{'*' if m in fam['simultaneous_excludes_zero'][e] else ' '}"
                                       for m, r in cells.items()))
    for m, c in res["classes"].items():
        rho = c.get("rho", {})
        print(f"  {m:9s} beta_clin {c.get('clinical', {}).get('beta', float('nan')):.2f} {c.get('clinical', {}).get('beta_ci')}  "
              f"beta_label {c.get('label', {}).get('beta', float('nan')):.2f}  rho {rho.get('rho')}  drift excess {res['drift'][m].get('excess_pts')}")
    print(f"wrote {out}")
