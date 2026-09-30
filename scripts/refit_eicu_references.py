"""eICU replication, step 4 (local machine): bootstrap refits of each reference on the full fitting set, recomputing d for every
applicable instance of a demo split. Writes data/eicu/refit_d_<ref>_<split>.npz in the NHANES format.
  python scripts/refit_eicu_references.py eval [--R 1000] [--refs spline_logit,lightgbm]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import eicu as E, eicu_reference as ER
sys.path.insert(0, str(Path(__file__).resolve().parent))
from refit_references import aligned_pairs

ROOT = Path(__file__).resolve().parents[1]


def run(split: str, n: int, ref: str, root: Path = ROOT, seed: int = 0):
    d = root / E.CFG["out_dir"]
    fit = pd.read_parquet(d / "full_fit.parquet").reset_index(drop=True)
    base, edited = aligned_pairs(pd.read_parquet(d / f"edited_{split}.parquet"))
    DL = np.zeros((len(base), n)); DP = np.zeros_like(DL); t0 = time.time()
    for r, m in enumerate(ER.bootstrap_refits(fit, n=n, seed=seed, kind=ref)):
        c = m.contrast(base, edited)
        DL[:, r], DP[:, r] = c["d_logit"].to_numpy(), c["d_prob"].to_numpy()
        if (r + 1) % 10 == 0:
            print(f"{ref}: {r + 1}/{n} refits, {time.time() - t0:.0f}s")
    assert np.isfinite(DL).all() and np.isfinite(DP).all(), f"{ref}: non-finite refit contrasts"
    np.savez_compressed(d / f"refit_d_{ref}_{split}.npz", d_logit=DL, d_prob=DP, profile=base["SEQN"].astype(int).to_numpy(),
                        edit=base["_edit"].to_numpy().astype(str),
                        meta=np.array(json.dumps({"ref": ref, "split": split, "R": n, "seed": seed, "population": "eicu_demo"})))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--refs", default="spline_logit,lightgbm")
    a = ap.parse_args()
    for ref in a.refs.split(","):
        run(a.split, a.R or int(E.CFG["reference"]["refits"][ref]), ref)
