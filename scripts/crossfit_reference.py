"""Out-of-fold outcome-reference predictions for every complete-case record of the fitting and evaluation splits (the
'wide' baseline set): 10-fold within the fitting split, full-fit for evaluation records. Also PhenoAge's ten-year
probability.
Writes data/reference_oof.parquet. Needed before analysing the wide baseline run (never score a record with a model
fitted on it)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.prior import settings, load_prior

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    fit = co[co.split == "fit"]
    wide = co[co.split.isin(["fit", "eval"]) & co.get("complete9", pd.Series(True, index=co.index))]
    prior = load_prior(); a0 = settings()["a0"] if prior is not None else 0.0
    oof = R.crossfit_predictions(fit, wide, k=10, prior_df=prior, a0=a0)
    ph = pd.Series(np.clip(R.phenoage_mortality_10y(wide).to_numpy(), 1e-6, 1 - 1e-6), index=wide.SEQN.astype(int).to_numpy())
    oof["p_phenoage_10y"] = oof.SEQN.map(ph)
    oof.to_parquet(ROOT / "data" / "reference_oof.parquet", index=False)
    print(f"out-of-fold reference predictions for {len(oof):,} records")
