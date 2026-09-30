"""Step 2. Fit the references on the fitting split (NHANES 1999-2008 outcomes only: a0 = 0 in config/nhanes3.yaml)
and audit them on the held-out splits before they anchor anything.
Writes data/reference_<name>.joblib and results/reference_performance.json.
  primary      spline_logit          (current era only)
  second       lightgbm              (current era only)
  secondary    survival_logit        (discrete-time hazard on the primary basis, follow-up to 20 years)
  sensitivity  spline_logit_prior (NHANES III power prior a0 = 0.5 on clinical effects),
               spline_logit_shared (labels borrowed too), spline_logit_weighted (examination weights on the NHANES
               1999-2008 rows; the cohort is analysed unweighted)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.analysis import performance
from jevity.prior import settings, load_prior

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    prior = load_prior(); cfg = settings()
    a0 = cfg["a0"] if prior is not None else 0.0
    fit, held = df[df.split == "fit"], df[df.split != "fit"]
    y = held[R.OUTCOME].to_numpy()
    out = {"n_fit": int(len(fit)), "deaths_fit": int(fit[R.OUTCOME].sum()), "n_heldout": int(len(held)), "deaths_heldout": int(y.sum()),
           "prior": {"used": prior is not None, "a0": a0, "n_prior": int(len(prior)) if prior is not None else 0,
                     "deaths_prior": int(prior[R.OUTCOME].sum()) if prior is not None else 0}}
    fits = {"spline_logit": lambda: R.fit_spline_logit(fit, prior_df=prior, a0=a0),
            "lightgbm": lambda: R.fit_lgbm(fit, prior_df=prior, a0=a0),
            "survival_logit": lambda: R.fit_survival_logit(fit, prior_df=prior, a0=a0),
            "age_sex": lambda: R.fit_age_sex(fit)}
    if R.SURVEY_WEIGHT in fit and fit[R.SURVEY_WEIGHT].notna().any():
        fits["spline_logit_weighted"] = lambda: R.fit_spline_logit(fit, prior_df=prior, a0=a0, weighted=True)
    if prior is not None:
        for name, v in cfg["sensitivities"].items():
            fits[name] = (lambda v=v: R.fit_spline_logit(fit, prior_df=prior, a0=v["a0"], era_specific=v["era_specific"]))
    for name, fn in fits.items():
        m = fn()
        joblib.dump(m, ROOT / "data" / f"reference_{name}.joblib")
        out[name] = performance(m.predict(held), y, B=300)
        print(name, {k: round(v["est"], 3) for k, v in out[name].items() if isinstance(v, dict)})
    out["phenoage_10y"] = performance(np.clip(R.phenoage_mortality_10y(held).to_numpy(), 1e-6, 1 - 1e-6), y, B=300)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "reference_performance.json").write_text(json.dumps(out, indent=1))
