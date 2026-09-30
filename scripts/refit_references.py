"""Outer loop of the two-level bootstrap: refit each reference on bootstrap resamples of the fitting split and
recompute d (log-odds and probability) for every supported instance of a split, in batch.
Writes data/refit_d_<ref>_<split>.npz with arrays d_logit, d_prob (n_instances x R) and the instance keys.
  python scripts/refit_references.py eval --R 300
spline_logit_weighted uses the same seed as spline_logit, so its refits are paired with the primary's (shift intervals)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.prior import settings, load_prior

ROOT = Path(__file__).resolve().parents[1]


def run(split: str, n_refits: int, ref: str, root: Path = ROOT, seed: int = 0):
    cohort = pd.read_parquet(root / "data" / "cohort.parquet")
    fit = cohort[cohort.split == "fit"].reset_index(drop=True)
    if ref == "spline_logit_weighted" and (R.SURVEY_WEIGHT not in fit or fit[R.SURVEY_WEIGHT].isna().all()):
        print("no examination weights in the cohort: weighted refits skipped"); return
    ed = pd.read_parquet(root / "data" / f"edited_{split}.parquet")
    base, edited = aligned_pairs(ed)
    keys = pd.DataFrame({"profile": base["SEQN"].astype(int), "edit": base["_edit"]})
    DL = np.zeros((len(base), n_refits)); DP = np.zeros_like(DL)
    t0 = time.time()
    prior = load_prior(root)
    a0 = settings(root)["a0"] if prior is not None else 0.0
    for r, m in enumerate(R.bootstrap_refits(fit, n=n_refits, seed=seed, kind=ref, prior_df=prior, a0=a0)):
        c = m.contrast(base, edited)
        DL[:, r], DP[:, r] = c["d_logit"].to_numpy(), c["d_prob"].to_numpy()
        if (r + 1) % 10 == 0:
            print(f"{ref}: {r + 1}/{n_refits} refits, {time.time() - t0:.0f}s")
    assert np.isfinite(DL).all() and np.isfinite(DP).all(), f"{ref}: non-finite refit contrasts"
    np.savez_compressed(root / "data" / f"refit_d_{ref}_{split}.npz", d_logit=DL, d_prob=DP,
                        profile=keys.profile.to_numpy(), edit=keys.edit.to_numpy().astype(str),
                        meta=np.array(json.dumps({"ref": ref, "split": split, "R": n_refits, "seed": seed, "a0": a0})))


def aligned_pairs(ed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline and edited rows joined on (SEQN, edit), one to one; never by position."""
    k = ["SEQN", "_edit"]
    base, edited = ed[ed._role == "base"], ed[ed._role == "edited"]
    for name, t in (("base", base), ("edited", edited)):
        dup = t.duplicated(k).sum()
        if dup:
            raise SystemExit(f"{dup} duplicated (SEQN, edit) keys among {name} rows")
    base, edited = base.sort_values(k).reset_index(drop=True), edited.sort_values(k).reset_index(drop=True)
    if len(base) != len(edited) or not (base[k].to_numpy() == edited[k].to_numpy()).all():
        raise SystemExit("baseline and edited rows do not pair one to one on (SEQN, edit)")
    return base, edited


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("--R", type=int, default=None,
                                                 help="default: the count in config/analysis.yaml")
    ap.add_argument("--refs", default="spline_logit,lightgbm,survival_logit,spline_logit_weighted")
    a = ap.parse_args()
    import yaml
    need = yaml.safe_load((ROOT / "config" / "analysis.yaml").read_text())["refits_required"]
    for ref in a.refs.split(","):
        run(a.split, a.R or int(need[ref]), ref)
